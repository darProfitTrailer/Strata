// Standalone Linux CPU reproducer: g++ -std=c++17 -pthread -Isrc/dsv41
// tests/dsv41/store_recovery_repro.cpp -o /tmp/store-recovery && /tmp/store-recovery
// Uses an anonymous temporary file; no model or external fixture.
#include "ds4_store.hpp"
#include <cstdio>
using namespace strata::ds4;
int main() {
    FILE* file = std::tmpfile();
    if (!file) return 2;
    int result = 0;
    {
        ExpertStore::LayerGeo g; g.n_exp = 2;
        for (int r=0;r<3;++r) {
            g.file[r]=0; g.off[r]=r*2*4096; g.bytes[r]=g.stride[r]=4096;
            g.tr[r]=r*(4096+ExpertStore::kSlack);
        }
        g.blob=3*(4096+ExpertStore::kSlack);
        ExpertStore::Cfg c; c.floor_bytes=0; c.threads=1;
        c.budget_bytes=g.blob+2*ExpertStore::kSlack;
        ExpertStore st;
        if (!st.init(c,{g},{os::kBadFd},{fileno(file)})) return 2;
        st.prefetch(0,0);
        int e=0; std::vector<char> ok;
        if (st.ensure(0,&e,1,ok) || ok[0] || st.stats().fails.load()!=1) return 2;
        // All failed reads have completed. Populate the same file, then retry.
        unsigned char buf[4096];
        for (int i=0;i<6;++i) {
            std::memset(buf,i,sizeof(buf));
            if (std::fwrite(buf,1,sizeof(buf),file)!=sizeof(buf)) return 2;
        }
        if (std::fflush(file)!=0) return 2;
        if (!st.ensure(0,&e,1,ok) || !ok[0]) {
            std::printf("FAIL: retry rejected; fails=%lld no_slot=%lld\n",
                (long long)st.stats().fails.load(),(long long)st.stats().no_slot.load());
            result=1;
        } else {
            int s=st.find(0,e);
            if (s<0) return 2;
            for(int r=0;r<3;++r) {
                std::memset(buf,r*2,sizeof(buf));
                if(std::memcmp(st.mat(s,r),buf,sizeof(buf))!=0) return 2;
            }
            st.release(0,&e,1,ok);
            std::puts("PASS: failed-prefetch retry recovered; 3 matrices verified");
        }
    }
    std::fclose(file);
    return result;
}
