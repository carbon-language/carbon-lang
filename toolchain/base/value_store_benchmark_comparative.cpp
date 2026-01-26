// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <vector>
#include <iostream>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <iomanip>

namespace Carbon::Testing {
namespace {

// Simulate Chunk (4KB alloc) - heavy construction.
struct Chunk {
    static constexpr int CapacityBytes = 4 * 1024;
    static constexpr int Capacity = CapacityBytes / sizeof(int);
    Chunk() { data_ = new int[Capacity]; std::memset(data_, 0, CapacityBytes); }
    Chunk(const Chunk& other) { data_ = new int[Capacity]; std::memcpy(data_, other.data_, CapacityBytes); }
    Chunk(Chunk&& other) noexcept : data_(other.data_) { other.data_ = nullptr; }
    ~Chunk() { delete[] data_; }
    void Add(int v) { if (count_ < Capacity) data_[count_++] = v; }
    int* data_ = nullptr;
    int count_ = 0;
};

// 1. Baseline Implementation (Eager Resize, No pre-alloc in usage).
struct ValueStoreBaseline {
    std::vector<Chunk> chunks_;
    int size_ = 0;
    // Eager reserve logic.
    void Reserve(int size) {
        if (size <= size_) return;
        int chunks_needed = (size + Chunk::Capacity - 1) / Chunk::Capacity;
        if (static_cast<size_t>(chunks_needed) > chunks_.size()) chunks_.resize(chunks_needed);
    }
    void Add(int v) {
        if (static_cast<size_t>(size_ / Chunk::Capacity) == chunks_.size()) chunks_.emplace_back();
        chunks_[size_ / Chunk::Capacity].Add(v);
        size_++;
    }
};

// 2. Lazy Reserve Implementation (Optimized).
struct ValueStoreLazy {
    std::vector<Chunk> chunks_;
    int size_ = 0;
    // Lazy reserve logic.
    void Reserve(int size) {
        if (size <= size_) return;
        int chunks_needed = (size + Chunk::Capacity - 1) / Chunk::Capacity;
        if (static_cast<size_t>(chunks_needed) > chunks_.capacity()) chunks_.reserve(chunks_needed);
    }
    void Add(int v) {
        if (static_cast<size_t>(size_ / Chunk::Capacity) == chunks_.size()) chunks_.emplace_back();
        chunks_[size_ / Chunk::Capacity].Add(v);
        size_++;
    }
};

long Measure(const char* name, auto func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    long us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return us;
}

void RunBenchmark() {
    constexpr int Runs = 100;
    constexpr int Items = 100000;

    std::cout << "Benchmark Comparison (" << Runs << " runs, " << Items << " items)\n";
    std::cout << "------------------------------------------------------------\n";

    // Scenario A: Baseline Code (Eager Reserve) + No Pre-allocation call.
    long time_baseline = Measure("Baseline (No Opts)", [&]() {
        for(int i=0; i<Runs; ++i) {
            ValueStoreBaseline store;
            for(int k=0; k<Items; ++k) store.Add(k);
        }
    });

    // Scenario B: Optimization 1 Only (Lazy Reserve implemented, but NOT called aggressively).
    long time_opt1 = Measure("Opt 1 Only (Lazy impl, incremental usage)", [&]() {
        for(int i=0; i<Runs; ++i) {
            ValueStoreLazy store;
            for(int k=0; k<Items; ++k) store.Add(k);
        }
    });

    // Scenario C: Optimization 2 Only (Simulated: Eager Reserve called aggressively).
    long time_opt2 = Measure("Opt 2 Only (Eager impl, Aggressive Reserve)", [&]() {
        for(int i=0; i<Runs; ++i) {
            ValueStoreBaseline store;
            store.Reserve(Items);
            for(int k=0; k<Items; ++k) store.Add(k);
        }
    });

    // Scenario D: Combined (Lazy Reserve + Aggressive Usage).
    long time_final = Measure("Final (Lazy impl + Aggressive Reserve)", [&]() {
        for(int i=0; i<Runs; ++i) {
            ValueStoreLazy store;
            store.Reserve(Items);
            for(int k=0; k<Items; ++k) store.Add(k);
        }
    });

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Baseline: " << time_baseline << " us\n";
    std::cout << "Opt 1 (Lazy Logic): " << time_opt1 << " us\n";
    std::cout << "Opt 2 (Pre-alloc Eager): " << time_opt2 << " us\n";
    std::cout << "Final (Lazy + Pre-alloc): " << time_final << " us\n";

    std::cout << "\nSpeedups:\n";
    if (time_final > 0) {
        std::cout << "Final vs Baseline: " << (double)time_baseline / time_final << "x\n";
        std::cout << "Final vs Opt 2 (Why lazy matters): " << (double)time_opt2 / time_final << "x\n";
    }
}

}  // namespace
}  // namespace Carbon::Testing

int main() {
    Carbon::Testing::RunBenchmark();
    return 0;
}
