#!/usr/bin/env python3
"""Small load generator for the OpenAI-compatible server: C clients at once, each a different prompt.

    python3 tools/serve_load.py http://127.0.0.1:8080 --clients 1,2,4,8 --rounds 3 --max-tokens 256

For every client count it sends C streamed chat requests at the same moment (each prompt starts with its own nonce, so
the prompt cache cannot help), and reports per round and as medians: total tokens/s (all generated tokens over the
wall time of the burst), per-request end-to-end tokens/s, time to first token (p50 / p95 over all requests of that count), and
the request latency (p50 / p95).  With --save DIR it writes every answer, so a later run can be compared byte for
byte with `--compare DIR` (greedy answers of a request must not depend on who else ran).
Configured API keys and bearer credentials are redacted before saving or comparing; comparison covers that redacted text.

Token counts come only from reported completion-token usage; missing counts stay unknown.
Per-request throughput includes prompt processing and transport time, not inferred decode time.
Standard library only.  The prompts are fixed texts, so two runs of the tool send the same requests.
"""
import argparse, json, os, re, sys, threading, time, urllib.error, urllib.request

from bench_metrics import finite_nonnegative, measured_median, measured_ratio, validated_metrics

TOPICS = [
    "the history of lighthouses and how their lenses worked",
    "how a mechanical watch keeps time",
    "why bridges are built as arches and as suspension spans",
    "the water cycle and what a drought does to a river valley",
    "how bread rises, from yeast to crust",
    "the design of a medieval castle and why walls were stepped",
    "how a transistor switches, explained for a curious teenager",
    "the life of a honey bee colony through one year",
]


def prompt_for(i: int, words: int, nonce: str = "") -> str:
    topic = TOPICS[i % len(TOPICS)]
    return (f"[request {i}{nonce}] Write a clear, well organised essay of about {words} words on {topic}. "
            f"Use plain prose, no lists.")


def public_text(text, key):
    """Keep authentication material out of saved replies and comparison output."""
    if key:
        text = text.replace(key, "<key>")
    return re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{8,}", r"\1<key>", text)


def sse_data(response):
    """Read complete SSE events, including multiline data and comment keep-alives."""
    parts = []
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if parts:
                yield "\n".join(parts)
                parts = []
        elif line.startswith("data:"):
            parts.append(line[5:].removeprefix(" "))
    if parts:
        raise ValueError("unterminated SSE event")


def one(url, key, model, i, words, max_tokens, res, extra, think, nonce=""):
    body = {"model": model, "stream": True, "stream_options": {"include_usage": True},
            "max_tokens": max_tokens, "temperature": 0,
            "messages": [{"role": "user", "content": prompt_for(i, words, nonce)}]}
    if not think:
        body["reasoning_effort"] = "none"
        body["chat_template_kwargs"] = {"enable_thinking": False}
    body.update(extra)
    t0 = time.perf_counter()
    first, n, text = None, None, []
    err, finish, done = None, None, False
    try:
        req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": "Bearer " + key} if key else {})})
        with urllib.request.urlopen(req, timeout=900) as r:
            for data in sse_data(r):
                if data == "[DONE]":
                    done = True
                    break
                j = json.loads(data)
                if not isinstance(j, dict):
                    raise ValueError("SSE event must be an object")
                if "error" in j:
                    err = "server reported a stream error"
                    break
                usage, _, errors = validated_metrics(j.get("usage"), j.get("timings"))
                if errors:
                    err = "; ".join(errors)
                    break
                if j.get("usage") is not None:
                    n = usage.get("completion_tokens")
                choices = j.get("choices", [])
                if not isinstance(choices, list) or len(choices) > 1:
                    raise ValueError("expected one streamed choice")
                for ch in choices:
                    if not isinstance(ch, dict) or type(ch.get("index", 0)) is not int or ch.get("index", 0) != 0:
                        raise ValueError("invalid streamed choice")
                    dl = ch.get("delta", {})
                    if not isinstance(dl, dict):
                        raise ValueError("invalid delta")
                    content, reasoning = dl.get("content"), dl.get("reasoning_content")
                    if any(value is not None and not isinstance(value, str) for value in (content, reasoning)):
                        raise ValueError("invalid content")
                    piece = (content or "") + (reasoning or "")
                    if piece:
                        if finish is not None:
                            raise ValueError("content after finish_reason")
                        if first is None:
                            first = time.perf_counter() - t0
                        text.append(piece)
                    reason = ch.get("finish_reason")
                    if reason is not None:
                        if not isinstance(reason, str) or reason not in ("stop", "length", "tool_calls"):
                            raise ValueError("invalid finish_reason")
                        if finish is not None and finish != reason:
                            raise ValueError("conflicting finish_reason")
                        finish = reason
    except urllib.error.HTTPError as e:
        err = f"HTTP {e.code}"
    except Exception as e:                          # noqa: BLE001
        # Exceptions can echo URLs, authorization values or response bodies.
        err = f"stream failed ({type(e).__name__})"
    if err is None and (not done or finish is None):
        err = "incomplete SSE stream: requires finish_reason and [DONE]"
    res[i] = {"first": first, "tokens": n if err is None else None, "total": time.perf_counter() - t0,
              "text": public_text("".join(text), key), "err": err, "finish_reason": finish}


def pct(v, p):
    v = sorted(v)
    if not v or not all(finite_nonnegative(x) for x in v):
        return None
    k = (len(v) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    value = v[lo] + (v[hi] - v[lo]) * (k - lo)
    return value if finite_nonnegative(value) else None


def show(value, digits):
    return f"{value:.{digits}f}" if value is not None else "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--clients", default="1,2,4,8")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--words", type=int, default=800, help="the essay length asked for (longer = longer prompt? no: the answer)")
    ap.add_argument("--model", default="strata")
    ap.add_argument("--key", default=os.environ.get("STRATA_KEY", ""))
    ap.add_argument("--nonce", default=str(int(time.time())), help="prefix of the per-request nonce (every round reads fresh prompts); '' = identical prompts every round; fix it to compare two runs")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--stagger", type=float, default=0.0, help="seconds between the starts of the clients of a burst")
    ap.add_argument("--save", help="write every answer (clients/round/request) here")
    ap.add_argument("--compare", help="compare every answer with the files a --save run wrote")
    ap.add_argument("--json", help="write the table as JSON")
    a = ap.parse_args()
    try:
        clients = [int(x) for x in a.clients.split(",")]
    except ValueError:
        ap.error("--clients must be comma-separated positive integers")
    if not clients or any(c <= 0 for c in clients) or a.rounds <= 0:
        ap.error("clients and rounds must be positive")
    if not finite_nonnegative(a.stagger):
        ap.error("--stagger must be finite and nonnegative")
    extra = {}
    out = []
    bad = 0
    for c in clients:
        rows = []
        for rd in range(a.rounds):
            res = {}
            th = []
            t0 = time.perf_counter()
            for i in range(c):
                t = threading.Thread(target=one, args=(a.url, a.key, a.model, i, a.words, a.max_tokens, res, extra, a.think, f" {a.nonce}-{c}-{rd}" if a.nonce else ""))
                t.start()
                th.append(t)
                if a.stagger:
                    time.sleep(a.stagger)
            for t in th:
                t.join()
            wall = time.perf_counter() - t0
            errs = [r["err"] for r in res.values() if r["err"]]
            if len(res) != c:
                errs.append("missing client result")
            good = [r for r in res.values() if not r["err"]]
            known = not errs and all(r["tokens"] is not None for r in good)
            toks = sum(r["tokens"] for r in good) if known else None
            tps = measured_ratio(toks, wall)
            if known and tps is None:
                errs.append("aggregate token count or throughput is not representable")
                if not finite_nonnegative(toks):
                    toks = None
            firsts = [r["first"] for r in good if r["first"] is not None]
            tot = [r["total"] for r in good]
            per = []
            for r in good:
                rate = measured_ratio(r["tokens"], r["total"])
                if r["tokens"] is not None and rate is None:
                    errs.append("per-request throughput is not representable")
                per.append(rate)
            rows.append({"wall": wall, "toks": toks, "tps": tps, "first": firsts, "lat": tot, "per": per, "err": errs})
            if a.save:
                d = os.path.join(a.save, f"c{c}", f"r{rd}")
                os.makedirs(d, exist_ok=True)
                for i, r in res.items():
                    if not r["err"]:
                        with open(os.path.join(d, f"{i}.txt"), "w", encoding="utf-8") as f:
                            f.write(r["text"])
            if a.compare:
                for i, r in res.items():
                    if r["err"]:
                        continue
                    p = os.path.join(a.compare, f"c{c}", f"r{rd}", f"{i}.txt")
                    if os.path.exists(p):
                        with open(p, encoding="utf-8") as f:
                            ref = public_text(f.read(), a.key)
                        if ref != r["text"]:
                            n = next((k for k, (x, y) in enumerate(zip(ref, r["text"])) if x != y), min(len(ref), len(r["text"])))
                            print(f"  DIFF c={c} round={rd} req={i}: first difference at char {n}")
                            bad += 1
                    else:
                        print(f"  MISSING comparison c={c} round={rd} req={i}")
                        bad += 1
            print(f"c={c} round={rd}: {toks if toks is not None else 'unknown'} tokens in {wall:.1f}s = {show(tps, 1)} tok/s total"
                  + (f"  ERRORS {errs[:2]}" if errs else ""), flush=True)
        allf = [x for r in rows for x in r["first"]]
        alll = [x for r in rows for x in r["lat"]]
        allp = [x for r in rows for x in r["per"]]
        s = {"clients": c, "total_tps": measured_median(r["tps"] for r in rows),
             "per_req_tps": measured_median(allp),
             "first_p50": pct(allf, .5), "first_p95": pct(allf, .95),
             "lat_p50": pct(alll, .5), "lat_p95": pct(alll, .95), "errors": sum(len(r["err"]) for r in rows),
             "rounds": rows, "per_req_tps_basis": "completion tokens / end-to-end request seconds"}
        out.append(s)
    print("\nclients  total tok/s  per-req tok/s  first p50/p95 (s)  latency p50/p95 (s)  errors")
    for s in out:
        print(f"{s['clients']:>7}  {show(s['total_tps'], 1):>11}  {show(s['per_req_tps'], 1):>13}  "
              f"{show(s['first_p50'], 2):>7}/{show(s['first_p95'], 2):<7}  "
              f"{show(s['lat_p50'], 2):>8}/{show(s['lat_p95'], 2):<8}  {s['errors']}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, allow_nan=False)
    failed = any(s["errors"] for s in out)
    if a.compare:
        print("compare:", "INCOMPLETE" if failed else ("ALL IDENTICAL" if not bad else f"{bad} DIFFERENT OR MISSING"))
    return 1 if bad or failed else 0


if __name__ == "__main__":
    sys.exit(main())
