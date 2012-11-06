//===-- tsan_clock.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//
#ifndef TSAN_CLOCK_H
#define TSAN_CLOCK_H

#include "tsan_defs.h"
#include "tsan_vector.h"

namespace __tsan {

// The clock that lives in sync variables (mutexes, atomics, etc).
class SyncClock {
 public:
  SyncClock();

  uptr size() const {
    return clk_.Size();
  }

  void Reset() {
    clk_.Reset();
  }

 private:
  Vector<u64> clk_;
  friend struct ThreadClock;
};

// The clock that lives in threads.
struct ThreadClock {
 public:
  ThreadClock();

  u64 get(unsigned tid) const {
    DCHECK_LT(tid, kMaxTidInClock);
    return clk_[tid];
  }

  void set(unsigned tid, u64 v) {
    DCHECK_LT(tid, kMaxTid);
    DCHECK(v >= clk_[tid] || disabled_);
    clk_[tid] = v;
    if (nclk_ <= tid)
      nclk_ = tid + 1;
  }

  void tick(unsigned tid) {
    DCHECK_LT(tid, kMaxTid);
    clk_[tid]++;
    if (nclk_ <= tid)
      nclk_ = tid + 1;
  }

  void Disable(unsigned tid);

  uptr size() const {
    return nclk_;
  }

  void acquire(const SyncClock *src);
  void release(SyncClock *dst) const;
  void acq_rel(SyncClock *dst);
  void ReleaseStore(SyncClock *dst) const;

 private:
  uptr nclk_;
  bool disabled_;
  u64 clk_[kMaxTidInClock];
};

}  // namespace __tsan

#endif  // TSAN_CLOCK_H
