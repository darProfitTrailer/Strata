#!/usr/bin/env python3
"""Strata vs llama.cpp head-to-head (stream 1402-h).  Measurement only.

Both engines are driven through their OpenAI-compatible HTTP servers (/v1/chat/completions), with the same prompt
text, same sampler settings and the same GGUF.  Server-side timings (llama.cpp's names, which Strata's server also
emits) give prompt tok/s and decode tok/s. prompt_ms is server prompt-processing time, not client TTFT;
non-streaming requests do not measure first-token latency. Memory peaks are observed from sampler start after
process spawn through each record, including loading, warm-up and earlier cells, not isolated per-cell measurements.
VRAM is total device usage of the first nvidia-smi device, not process usage or a verified engine-selected GPU.
Baselines are observations before process start, not a guarantee of an idle machine. RAM comes from the
engine process tree (sum of psutil RSS, including mmap'd pages; shared pages may be counted more than once)
and the system's used-memory delta.

    python bench_vs_llama.py prompts  --out DIR --tokenizer DIR --corpus FILE...     # build 4K / 32K prompt files
    python bench_vs_llama.py run      --machine CFG.json --out DIR --rounds N [--configs a,b] [--cells ...]
    python bench_vs_llama.py quality  --machine CFG.json --out DIR [--configs a,b]
    python bench_vs_llama.py summary  --out DIR

CFG.json: {"gguf": ..., "strata_cfg": ..., "llama_server": ..., "configs": [{"name","kind":"llama|strata","args":[...]}]}
Interleaving: engines cannot be loaded together (12 GB card, 64 GB RAM), so one ROUND runs every config once
(start, warm-up, cells, stop), and rounds alternate the config order.  Medians are taken over rounds.
The caller holds the machine's GPU lock.  Processes are stopped by PID only.
"""
import argparse, json, os, subprocess, sys, threading, time, urllib.request
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from bench_metrics import validated_metrics, measured_median, measured_ratio, finite_nonnegative

CELLS = [("4k", 4000, 2), ("32k", 32000, 1)]       # (name, prompt tokens, reps per round)
TEMPS = [0.0, 0.7]
GEN = 256
PORT = 8190


def http(url, body=None, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def memory_value(value):
    return value if finite_nonnegative(value) else None


def vram_used_mib():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=20)
        if result.returncode != 0:
            return None
        return memory_value(int(result.stdout.split()[0]))
    except Exception:
        return None


def rss_used_bytes(pid):
    """A complete process-tree observation, or unknown if any process cannot be read."""
    try:
        parent = psutil.Process(pid)
        readings = [memory_value(p.memory_info().rss) for p in [parent] + parent.children(recursive=True)]
        return memory_value(sum(readings)) if all(value is not None for value in readings) else None
    except (psutil.Error, OSError):
        return None


def system_used_bytes():
    try:
        return memory_value(psutil.virtual_memory().used)
    except (psutil.Error, OSError):
        return None


class Sampler(threading.Thread):
    def __init__(self, pid):
        super().__init__(daemon=True)
        self.pid, self.stop = pid, False
        self.vram = self.rss = self.sysused = None
        self.started = time.monotonic()
        self.last_sample_s = None
        self.attempts = 0
        self.valid = dict(vram=0, rss=0, system=0)
        self.lock = threading.Lock()

    def sample(self):
        readings = dict(vram=vram_used_mib(), rss=rss_used_bytes(self.pid), system=system_used_bytes())
        completed = time.monotonic() - self.started
        with self.lock:
            self.attempts += 1
            for name, attr in (("vram", "vram"), ("rss", "rss"), ("system", "sysused")):
                value = memory_value(readings[name])
                if value is not None:
                    self.valid[name] += 1
                    previous = getattr(self, attr)
                    setattr(self, attr, value if previous is None else max(previous, value))
            self.last_sample_s = completed

    def snapshot(self):
        with self.lock:
            return dict(vram=self.vram, rss=self.rss, system=self.sysused,
                        observations={"attempts": self.attempts, "valid": self.valid.copy()},
                        last_sample_s=self.last_sample_s, snapshot_s=time.monotonic() - self.started)

    def run(self):
        while not self.stop:
            self.sample()
            time.sleep(1.0)


def memory_record(engine):
    sampler = engine.sampler
    snapshot = getattr(sampler, "snapshot", None)
    if callable(snapshot):
        values = snapshot()
        scope = "sampler_start_to_record"
    else:
        # Compatibility with callers supplying the earlier attribute-only sampler.
        values = dict(vram=sampler.vram or None, rss=sampler.rss or None, system=sampler.sysused or None,
                      observations=None, last_sample_s=None, snapshot_s=None)
        scope = "legacy_run_unknown_coverage"

    def gib(value):
        value = memory_value(value)
        return round(value / 2**30, 2) if value is not None else None

    base_used, system_peak = memory_value(engine.base_used), memory_value(values["system"])
    delta = round((system_peak - base_used) / 2**30, 2) if system_peak is not None and base_used is not None else None
    return dict(vram_peak_mib=memory_value(values["vram"]), vram_base_mib=memory_value(engine.base_vram),
                rss_peak_gib=gib(values["rss"]), sys_used_delta_gib=delta,
                sys_used_peak_gib=gib(system_peak), sys_used_base_gib=gib(base_used),
                memory_scope=scope, memory_observations=values["observations"],
                memory_last_poll_completed_s=values["last_sample_s"], memory_snapshot_s=values["snapshot_s"],
                memory_baseline_scope="before_process_start", memory_poll_delay_s=1.0,
                vram_scope="first nvidia-smi device, total device usage; engine GPU not verified")


def memory_summary(rows, field, sensor, digits, baseline=None):
    values, partial, legacy = [], False, False
    for row in rows:
        value = memory_value(row.get(field))
        metadata = row.get("memory_observations")
        covered = row.get("memory_scope") == "sampler_start_to_record" and isinstance(metadata, dict)
        if covered:
            attempts = metadata.get("attempts", 0)
            valid_counts = metadata.get("valid")
            valid = valid_counts.get(sensor, 0) if isinstance(valid_counts, dict) else 0
            if type(attempts) is not int or type(valid) is not int or not 0 <= valid <= attempts:
                value, partial = None, True
            else:
                partial |= valid < attempts or valid == 0
                if valid == 0:
                    value = None
        else:
            legacy = True
            # Old zero-initialized samplers cannot distinguish no observation from zero.
            if value == 0:
                value = None
        if baseline:
            base = memory_value(row.get(baseline))
            if value is None or base is None:
                value = None
            else:
                value -= base  # A negative device delta is a valid observation.
        if value is None:
            partial = True
        else:
            values.append(value)
    if not values:
        return "unknown"
    value = max(values)
    shown = f"{value:+.{digits}f}" if baseline else f"{value:.{digits}f}"
    return shown + (" (partial)" if partial else "") + (" (coverage unknown)" if legacy else "")


class Engine:
    def __init__(self, machine, cfg):
        self.m, self.cfg, self.proc, self.log = machine, cfg, None, None

    def command(self):
        c = self.cfg
        if c["kind"] == "llama":
            return [self.m["llama_server"], "-m", self.m["gguf"], "--port", str(PORT), "--host", "127.0.0.1",
                    "-np", "1", "--no-webui", "--jinja"] + c["args"]
        scfg = json.load(open(self.m["strata_cfg"], encoding="utf-8-sig"))
        scfg["port"] = PORT
        scfg.pop("vision", None)
        args = [a for a in scfg["args"]]
        for k, v in c.get("override", {}).items():          # replace "--flag value" pairs
            if k in args:
                args[args.index(k) + 1] = v
            else:
                args += [k, v]
        scfg["args"] = args + c["args"]
        if self.m.get("strata_exe"):
            scfg["exe"] = self.m["strata_exe"]
        p = Path(self.out) / ("strata-cfg-%s.json" % c["name"])
        p.write_text(json.dumps(scfg, indent=1))
        return [sys.executable, str(Path(self.m["serve"]) / "server.py"), "--engine", "strata", "--config", str(p),
                "--port", str(PORT)]

    def start(self, out):
        self.out = out = str(Path(out).resolve())
        self.base_vram, self.base_used = vram_used_mib(), system_used_bytes()
        cmd = self.command()
        self.cmdline = " ".join(cmd)
        self.log = open(Path(out) / ("server-%s-%d.log" % (self.cfg["name"], int(time.time()))), "w", encoding="utf-8")
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, env=env,
                                     cwd=self.m.get("cwd"))
        self.t0 = time.time()
        self.sampler = Sampler(self.proc.pid)
        self.sampler.start()
        health = "http://127.0.0.1:%d/health" % PORT
        while time.time() - self.t0 < 1500:
            if self.proc.poll() is not None:
                raise RuntimeError("server exited rc=%s (see %s)" % (self.proc.returncode, self.log.name))
            try:
                r = http(health, timeout=5)
                if r.get("status", "ok") == "ok":
                    break
            except Exception:
                pass
            time.sleep(2)
        else:
            raise RuntimeError("server not ready in 1500 s")
        self.load_s = time.time() - self.t0

    def stop(self):
        if self.proc:
            self.sampler.stop = True
            try:
                p = psutil.Process(self.proc.pid)
                kids = p.children(recursive=True)
                for c in kids + [p]:
                    try:
                        c.terminate()
                    except psutil.Error:
                        pass
                psutil.wait_procs(kids + [p], timeout=30)
                for c in kids + [p]:
                    try:
                        if c.is_running():
                            c.kill()
                    except psutil.Error:
                        pass
            except psutil.Error:
                pass
            self.log.close()
            self.proc = None
            time.sleep(5)

    def chat(self, prompt, temp, max_tokens=GEN, seed=1):
        body = {"model": "x", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                "temperature": temp, "top_k": 20, "top_p": 0.95, "seed": seed, "stream": False,
                "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}}
        if temp == 0:
            body.pop("top_p"), body.pop("top_k")
        t = time.time()
        r = http("http://127.0.0.1:%d/v1/chat/completions" % PORT, body)
        wall = time.time() - t
        u, tm, errors = validated_metrics(r.get("usage"), r.get("timings"))
        if errors:
            raise ValueError("invalid benchmark metrics: " + "; ".join(errors))
        return {"wall_s": round(wall, 2), "prompt_n": tm.get("prompt_n", u.get("prompt_tokens")),
                "prompt_ms": tm.get("prompt_ms"), "prompt_tps": tm.get("prompt_per_second"),
                "pred_n": tm.get("predicted_n", u.get("completion_tokens")), "pred_tps": tm.get("predicted_per_second"),
                "draft_n": tm.get("draft_n"), "draft_acc": tm.get("draft_n_accepted"),
                "text": (r["choices"][0]["message"].get("content") or "")}


def cmd_prompts(a):
    sys.path.insert(0, str(Path(a.repo) / "tools"))
    import strata_tokenizer as ST
    tp = Path(a.tokenizer)
    vocab = json.loads((tp / "vocab.json").read_text(encoding="utf-8"))
    toks = [None] * len(vocab)
    for t, i in vocab.items():
        toks[i] = t
    tok = ST.Tokenizer(toks, (tp / "merges.txt").read_text(encoding="utf-8").split("\n"),
                       json.loads((tp / "token_type.json").read_text()))
    corpus = "\n\n".join(Path(f).read_text(encoding="utf-8", errors="replace") for f in a.corpus)
    print("corpus chars", len(corpus))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    allids = tok.encode(corpus)
    print("corpus tokens", len(allids))
    for name, n, _ in CELLS:
        if len(allids) < n:
            raise SystemExit("corpus too small for %s" % name)
        (out / ("prompt-%s.txt" % name)).write_text(tok.decode(allids[:n]), encoding="utf-8")
        print(name, n, "tokens")


def load_prompt(out, name):
    return (Path(out) / ("prompt-%s.txt" % name)).read_text(encoding="utf-8")


def cmd_run(a):
    m = json.load(open(a.machine))
    out = Path(a.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    cfgs = [c for c in m["configs"] if not a.configs or c["name"] in a.configs.split(",")]
    cells = [c for c in CELLS if not a.cells or c[0] in a.cells.split(",")]
    prompts = {c[0]: load_prompt(out, c[0]) for c in cells}
    if not cfgs or not cells or a.rounds < 1:
        raise ValueError("select at least one config, cell and round")
    failed = False
    for rnd in range(a.start_round, a.start_round + a.rounds):
        order = cfgs if rnd % 2 == 0 else cfgs[::-1]
        for cfg in order:
            eng = Engine(m, cfg)
            rec_path = out / "raw" / ("%s.jsonl" % cfg["name"])
            try:
                eng.start(str(out))
                eng.chat("Say hello.", 0.0, 16)                              # warm-up, discarded
                eng.chat(prompts["4k"][:6000] if "4k" in prompts else "hi", 0.0, 32)
                for cname, n, reps in cells:
                    for t in TEMPS:
                        for k in range(reps):
                            nonce = "[bench r%d %s t%.1f k%d]\n" % (rnd, cname, t, k)
                            r = eng.chat(nonce + prompts[cname] + "\n\nSummarize the text above in three sentences.", t)
                            r.update(config=cfg["name"], round=rnd, cell=cname, temp=t, rep=k,
                                     load_s=round(eng.load_s, 1), cmd=eng.cmdline if (rnd == a.start_round and k == 0) else None)
                            r.update(memory_record(eng))
                            r["text"] = r["text"][:300]
                            with open(rec_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps(r) + "\n")
                            print(cfg["name"], rnd, cname, t, "pp %s tg %s draft %s/%s prompt_ms %s" % (
                                r["prompt_tps"], r["pred_tps"], r["draft_acc"], r["draft_n"], r["prompt_ms"]), flush=True)
            except Exception as e:
                failed = True
                print("ERROR", cfg["name"], rnd, repr(e), flush=True)
                with open(out / "raw" / "errors.log", "a") as f:
                    f.write("%s r%d %r\n" % (cfg["name"], rnd, e))
            finally:
                eng.stop()
    return 1 if failed else 0


QPROMPT = "Write a short story about a lighthouse keeper."


def cmd_quality(a):
    m = json.load(open(a.machine))
    out = Path(a.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    cfgs = [c for c in m["configs"] if not a.configs or c["name"] in a.configs.split(",")]
    res = {}
    for cfg in cfgs:
        eng = Engine(m, cfg)
        try:
            eng.start(str(out))
            eng.chat("Say hello.", 0.0, 16)
            for q in (QPROMPT, "Write a Python function that parses a CSV file into a list of dicts, with tests."):
                r = eng.chat(q, 0.0, 200)
                res.setdefault(cfg["name"], []).append(r["text"])
        finally:
            eng.stop()
    (out / "raw" / "quality.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


def cmd_summary(a):
    out = Path(a.out)
    rows = {}
    for f in sorted((out / "raw").glob("*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            _, _, errors = validated_metrics(None, {
                "prompt_per_second": r.get("prompt_tps"), "predicted_per_second": r.get("pred_tps"),
                "prompt_ms": r.get("prompt_ms"), "prompt_n": r.get("prompt_n"), "predicted_n": r.get("pred_n"),
                "draft_n": r.get("draft_n"), "draft_n_accepted": r.get("draft_acc")})
            if errors:
                raise ValueError("invalid saved benchmark metrics: " + "; ".join(errors))
            rows.setdefault((r["config"], r["cell"], r["temp"]), []).append(r)
    def med(xs, digits):
        value = measured_median(xs)
        return f"{value:.{digits}f}" if value is not None else "not measured"
    print("Memory columns show observed run-to-record peaks, including loading, warm-up and earlier cells; not per-cell peaks.")
    print("Polling may miss peaks. Partial means missing observations; legacy coverage is unknown and old zero peaks are ambiguous.")
    print("VRAM is first-device total usage; baseline is pre-start. Peak and delta columns are separate maxima of available snapshots.")
    print("RSS sums the process tree and may count shared pages more than once.")
    print("| config | ctx | temp | n | prompt tok/s | decode tok/s | Prompt processing s | draft acc | Observed run VRAM MiB / delta | Observed run RSS GiB |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for (c, cell, t), rs in sorted(rows.items()):
        pp = [r["prompt_tps"] for r in rs if r["prompt_tps"] is not None]
        tg = [r["pred_tps"] for r in rs if r["pred_tps"] is not None]
        tt = [r["prompt_ms"] / 1000 for r in rs if r["prompt_ms"] is not None]
        da = [ratio for r in rs if (ratio := measured_ratio(r.get("draft_acc"), r.get("draft_n"))) is not None]
        print("| %s | %s | %.1f | %d | %s | %s | %s | %s | %s / %s | %s |" % (
            c, cell, t, len(rs), med(pp, 1), med(tg, 2), med(tt, 2), med(da, 2) if da else "-",
            memory_summary(rs, "vram_peak_mib", "vram", 0),
            memory_summary(rs, "vram_peak_mib", "vram", 0, baseline="vram_base_mib"),
            memory_summary(rs, "rss_peak_gib", "rss", 1)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prompts", "run", "quality", "summary"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--machine")
    ap.add_argument("--configs")
    ap.add_argument("--cells")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--start-round", type=int, default=0)
    ap.add_argument("--tokenizer")
    ap.add_argument("--repo")
    ap.add_argument("--corpus", nargs="*")
    a = ap.parse_args()
    sys.exit({"prompts": cmd_prompts, "run": cmd_run, "quality": cmd_quality, "summary": cmd_summary}[a.mode](a))
