#include "ds4_store.hpp"
#include <stdexcept>
using namespace strata::ds4;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"line %d: %s\n",__LINE__,#x); throw std::runtime_error("recovery path"); } } while(0)
static unsigned checked=0;
static std::unique_ptr<ExpertStore> make_store(ExpertStore::Policy policy, int slots=1, bool mapped=false) {
    ExpertStore::LayerGeo g; g.n_exp=80;
    for(int r=0;r<3;++r) { g.file[r]=0; g.off[r]=r*80*4096; g.bytes[r]=g.stride[r]=4096; g.tr[r]=r*(4096+ExpertStore::kSlack); }
    g.blob=3*(4096+ExpertStore::kSlack);
    ExpertStore::Cfg c; c.policy=policy; c.floor_bytes=0; c.reserve=0;
    c.budget_bytes=slots*(g.blob+2*ExpertStore::kSlack);
    if(mapped) c.map_chunk=[](void* p,size_t){return p;};
    auto st=std::make_unique<ExpertStore>(); CHECK(st->init(c,{g},{os::kBadFd},{os::kBadFd})); return st;
}
static void read(ExpertStore& st,int e) {
    std::vector<char> ok; CHECK(st.ensure(0,&e,1,ok) && ok[0]); int slot=st.find(0,e); CHECK(slot>=0);
    for(int r=0;r<3;++r) { auto p=st.mat(slot,r); for(int i=0;i<4096;++i) CHECK(p[i]==(uint8_t)(r*80+e)); ++checked; }
    st.release(0,&e,1,ok);
}
static void fail_io() { auto& io=*IoQueue::current; io.dispatch(); io.complete(true); io.drain(); }
static void reject(ExpertStore& st,int e) {
    auto before=st.stats().evictions.load(); std::vector<char> ok;
    CHECK(!st.ensure(0,&e,1,ok) && !ok[0]);
    st.prefetch(0,e); CHECK(st.preload({{0,e}},0)==0); CHECK(st.put_begin(0,e)==nullptr);
    CHECK(IoQueue::current->queued()==0 && st.stats().evictions.load()==before);
}
static void duplicate_waiters(ExpertStore::Policy policy) {
    auto st=make_store(policy); int es[2]={0,0}; std::vector<char> ok; std::vector<int> sl;
    st->begin(0,es,2,ok,sl); CHECK(sl[0]==sl[1] && sl[0]>=0); fail_io();
    // Exercises the post-finish free-slot return before either waiter drains.
    reject(*st,1);
    CHECK(!st->wait_slots(sl,ok,0,1) && !ok[0]); reject(*st,1);
    CHECK(!st->wait_slots(sl,ok,1,2) && !ok[1]); read(*st,1);
}
static void failed_put(ExpertStore::Policy policy) {
    auto st=make_store(policy); CHECK(st->put_begin(0,0)!=nullptr);
    int e=0; std::vector<char> ok; std::vector<int> sl; st->begin(0,&e,1,ok,sl);
    st->put_done(0,0,false); reject(*st,1); // free_hint rewound, but demand pin remains
    CHECK(!st->wait_slots(sl,ok,0,1) && !ok[0]); read(*st,1);
}
static void cleanup(ExpertStore::Policy policy,int path) {
    auto st=make_store(policy); st->prefetch(0,0); fail_io();
    if(path==0) { int e=0; std::vector<char> ok; CHECK(!st->ensure(0,&e,1,ok)&&!ok[0]); }
    if(path==1) CHECK(st->preload({},0)==0);
    if(path==2) { std::vector<ExpertStore::Vis> ev; CHECK(st->sync(ev,[](int,int){return false;})==0); }
    if(path==3) CHECK(st->shrink_chunks(1)==1);
    // path 4 deliberately lets take_slot itself clean the completed failure.
    read(*st,1); CHECK(st->stats().fails.load()==1 && st->stats().evictions.load()==0);
}
static void guards(ExpertStore::Policy policy,bool visible) {
    auto st=make_store(policy,1,visible); read(*st,0);
    int e=0; std::vector<char> ok;
    if(visible) { std::vector<ExpertStore::Vis> ev; CHECK(st->sync(ev,[](int,int){return true;})==1); }
    else CHECK(st->ensure(0,&e,1,ok)&&ok[0]);
    reject(*st,1);
    if(visible) st->unmark(0,0); else st->release(0,&e,1,ok);
    read(*st,1); CHECK(st->stats().evictions.load()==1);
}
static void order_and_growth(ExpertStore::Policy policy) {
    auto st=make_store(policy,65); CHECK(st->slots_allocated()==64);
    st->prefetch(0,0); fail_io(); CHECK(st->preload({},0)==0);
    for(int e=1;e<64;++e) { read(*st,e); CHECK(st->find(0,e)==e); } // preserve suffix-first
    read(*st,64); CHECK(st->find(0,64)==0);
    CHECK(st->slots_allocated()==64 && st->stats().evictions.load()==0);
}
static void loading_guard(ExpertStore::Policy policy) {
    auto st=make_store(policy); st->prefetch(0,0);
    // Queued reads must never be stolen. Here reject()'s queued==0 is inapplicable.
    int e=1; std::vector<char> ok; CHECK(!st->ensure(0,&e,1,ok)&&!ok[0]);
    st->prefetch(0,1); CHECK(st->put_begin(0,1)==nullptr); CHECK(st->preload({{0,1}},0)==0);
    // preload drained the original jobs, so the original resident now becomes ready.
    read(*st,0); CHECK(st->stats().evictions.load()==0);
}
int main() {
    for(auto p:{ExpertStore::kLru,ExpertStore::kLfu}) {
        duplicate_waiters(p); failed_put(p);
        for(int path=0;path<5;++path) cleanup(p,path);
        guards(p,false); guards(p,true); order_and_growth(p); loading_guard(p);
    }
    CHECK(checked==456);
    std::printf("allocator paths passed; %u matrices verified\n",checked);
}
