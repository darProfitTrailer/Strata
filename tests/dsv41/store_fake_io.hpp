// Test-only substitute for ds4_pool.hpp in a temporary copy of ds4_store.hpp.
// No disk, threads, clocks, or GPU. Models queue dispatch, cancellation at dispatch,
// and completion separately; it is not an implementation/performance model of IoQueue.
#pragma once
#include "ds4_os.hpp"
#include <atomic>
#include <cstdint>
#include <cstring>
#include <deque>
#include <stdexcept>
namespace strata::ds4 {
class IoQueue {
public:
    struct Job {
        os::fd_t fd = os::kBadFd, dfd = os::kBadFd;
        uint64_t off = 0, len = 0;
        uint8_t* dst = nullptr;
        uint8_t* keep = nullptr;
        std::atomic<int>* left = nullptr;
        std::atomic<bool>* bad = nullptr;
        std::atomic<bool>* cancel = nullptr;
    };
    inline static IoQueue* current = nullptr;
    unsigned completed_ok = 0, dropped = 0, failed = 0;
    explicit IoQueue(int) {
        if (current) throw std::runtime_error("one controlled store at a time");
        current = this;
    }
    ~IoQueue() { current = nullptr; }
    void push(const Job* jobs, int n, bool low = false) {
        auto& q = low ? low_ : high_;
        for (int i = 0; i < n; ++i) q.push_back(jobs[i]);
    }
    size_t queued() const { return high_.size() + low_.size(); }
    bool active() const { return active_; }
    void dispatch() {
        if (active_ || !queued()) throw std::runtime_error("invalid dispatch");
        auto& q = high_.empty() ? low_ : high_;
        job_ = q.front(); q.pop_front(); active_ = true;
        cancelled_ = job_.cancel && job_.cancel->load(std::memory_order_acquire);
    }
    void complete(bool fail = false) {
        if (!active_) throw std::runtime_error("no dispatched job");
        if (cancelled_ || fail) {
            if (cancelled_) ++dropped; else ++failed;
            job_.bad->store(true, std::memory_order_release);
        } else {
            ++completed_ok;
            std::memset(job_.keep, (int)(job_.off / 4096), (size_t)job_.len);
        }
        job_.left->fetch_sub(1, std::memory_order_acq_rel);
        active_ = false;
    }
    void drain() { while (active_ || queued()) { if (!active_) dispatch(); complete(); } }
    void wait_any(int) { if (!active_) dispatch(); complete(); }
    double busy_ms() const { return 0; }
private:
    std::deque<Job> high_, low_;
    Job job_;
    bool active_ = false, cancelled_ = false;
};
}
