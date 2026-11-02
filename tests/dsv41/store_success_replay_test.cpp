// CPU-only replay of ExpertStore against a small generated file; no model or GPU.
#include "ds4_store.hpp"
#include <stdexcept>
using namespace strata::ds4;
#define check(ok) do { if (!(ok)) { std::fprintf(stderr, "failed line %d: %s\n", __LINE__, #ok); throw std::runtime_error("replay assertion failed"); } } while(0)
int main(int argc, char** argv) {
    check(argc == 2);
    auto fd = os::open_par(argv[1]);
    check(fd != os::kBadFd);
    for (auto policy : {ExpertStore::kLru, ExpertStore::kLfu}) {
        ExpertStore st;
        ExpertStore::LayerGeo g;
        g.n_exp = 8;
        for (int r = 0; r < 3; ++r) {
            g.file[r] = 0; g.off[r] = r * 8 * 4096; g.bytes[r] = g.stride[r] = 4096;
            g.tr[r] = r * (4096 + ExpertStore::kSlack);
        }
        g.blob = 3 * (4096 + ExpertStore::kSlack);
        ExpertStore::Cfg c;
        c.budget_bytes = 2 * (g.blob + 2 * ExpertStore::kSlack);
        c.floor_bytes = 0; c.threads = 1; c.policy = policy;
        check(st.init(c, {g}, {os::kBadFd}, {fd}));
        unsigned checked = 0;
        auto demand = [&](int e) {
            std::vector<char> ok;
            check(st.ensure(0, &e, 1, ok)); check(ok.size() == 1 && ok[0]);
            int s = st.find(0, e); check(s >= 0);
            for (int r = 0; r < 3; ++r) {
                auto p = st.mat(s, r);
                for (int i = 0; i < 4096; ++i) check(p[i] == (uint8_t)(r * 8 + e));
                ++checked;
            }
            st.release(0, &e, 1, ok);
        };
        demand(0); demand(1);
        st.prefetch(0, 2);
        st.preload({}, 0); // drain actual reads without a demand encounter
        check(st.find(0, 0) == -1); check(st.find(0, 1) >= 0); check(st.find(0, 2) >= 0);
        demand(2);
        check(st.preload({{0,3}}, 0) == 1);
        auto dst = st.put_begin(0, 4); check(dst != nullptr);
        for (int r = 0; r < 3; ++r) std::memset(dst + g.tr[r], r * 8 + 4, 4096);
        st.put_done(0,4,true); demand(4);
        int es[2] = {-1,-1}, count = 0;
        for (int e = 0; e < 5; ++e) if (st.find(0,e) >= 0) es[count++] = e;
        check(count == 2); std::vector<char> ok;
        check(st.ensure(0, es, 2, ok)); check(ok[0] && ok[1]);
        for (int e : es) {
            int slot = st.find(0, e); check(slot >= 0);
            for (int r = 0; r < 3; ++r) {
                auto p = st.mat(slot, r);
                for (int i = 0; i < 4096; ++i) check(p[i] == (uint8_t)(r * 8 + e));
                ++checked;
            }
        }
        st.prefetch(0, 5); // all slots pinned: observe actual no-slot rejection
        st.release(0, es, 2, ok);
        check(checked == 18);

        std::printf("policy=%d checked=%u evictions=%lld misses=%lld logical_requested=%lld\n", (int)policy, checked,
                    (long long)st.stats().evictions.load(), (long long)st.stats().misses.load(),
                    (long long)st.stats().bytes_read.load());
    }
}
