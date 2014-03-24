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

const u64 kClkMask = (1ULL << kClkBits) - 1;

// The clock that lives in sync variables (mutexes, atomics, etc).
class SyncClock {
 public:
  SyncClock();

  uptr size() const {
    return clk_.Size();
  }

  u64 get(unsigned tid) const {
    DCHECK_LT(tid, clk_.Size());
    return clk_[tid] & kClkMask;
  }

  void Reset();

  void DebugDump(int(*printf)(const char *s, ...));

 private:
  u64 release_seq_;
  unsigned release_store_tid_;
  static const uptr kDirtyTids = 2;
  unsigned dirty_tids_[kDirtyTids];
  mutable Vector<u64> clk_;
  friend struct ThreadClock;
};

// The clock that lives in threads.
struct ThreadClock {
 public:
  explicit ThreadClock(unsigned tid);

  u64 get(unsigned tid) const {
    DCHECK_LT(tid, kMaxTidInClock);
    DCHECK_EQ(clk_[tid], clk_[tid] & kClkMask);
    return clk_[tid];
  }

  void set(unsigned tid, u64 v);

  void set(u64 v) {
    DCHECK_GE(v, clk_[tid_]);
    clk_[tid_] = v;
  }

  void tick() {
    clk_[tid_]++;
  }

  uptr size() const {
    return nclk_;
  }

  void acquire(const SyncClock *src);
  void release(SyncClock *dst) const;
  void acq_rel(SyncClock *dst);
  void ReleaseStore(SyncClock *dst) const;

  void DebugDump(int(*printf)(const char *s, ...));

 private:
  static const uptr kDirtyTids = SyncClock::kDirtyTids;
  const unsigned tid_;
  u64 last_acquire_;
  uptr nclk_;
  u64 clk_[kMaxTidInClock];

  bool IsAlreadyAcquired(const SyncClock *src) const;
  void UpdateCurrentThread(SyncClock *dst) const;
};

}  // namespace __tsan

#endif  // TSAN_CLOCK_H
