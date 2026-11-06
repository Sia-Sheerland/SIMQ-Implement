#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>

// A separate reporting thread also emits a heartbeat during a slow operation.
// ETA is for this stage only, extrapolated from its average completion rate.
class ProgressReporter {
    using Clock = std::chrono::steady_clock;
    std::string name_;
    uint64_t total_;
    std::atomic<uint64_t> completed_{0};
    Clock::time_point start_ = Clock::now();
    std::mutex mutex_;
    std::condition_variable wake_;
    bool stopped_ = false;
    std::thread reporter_;

    void print() const {
        const auto n = completed_.load(std::memory_order_relaxed);
        const double elapsed = std::chrono::duration<double>(Clock::now() - start_).count();
        std::ostringstream out;
        out << "[progress] " << name_ << std::fixed << std::setprecision(1);
        if (total_) out << " " << n << "/" << total_ << " (" << 100.0 * n / total_ << "%)";
        out << " elapsed=" << elapsed << "s";
        if (total_ && n && elapsed > 0) {
            out << " rate=" << n / elapsed << "/s stage_ETA="
                << (n >= total_ ? 0.0 : elapsed * (total_ - n) / n) << "s";
        } else {
            out << " stage_ETA=unknown";
        }
        std::cerr << out.str() << std::endl;
    }

public:
    explicit ProgressReporter(std::string name, uint64_t total = 0)
        : name_(std::move(name)), total_(total) {
        print();
        reporter_ = std::thread([this] {
            std::unique_lock<std::mutex> lock(mutex_);
            while (!wake_.wait_for(lock, std::chrono::seconds(5), [this] { return stopped_; }))
                print();
        });
    }
    void advance() { completed_.fetch_add(1, std::memory_order_relaxed); }
    void finish() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopped_) return;
            stopped_ = true;
        }
        wake_.notify_one();
        reporter_.join();
        print();
    }
    ~ProgressReporter() { finish(); }
    ProgressReporter(const ProgressReporter&) = delete;
    ProgressReporter& operator=(const ProgressReporter&) = delete;
};
