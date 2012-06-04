//===-- tsan_clock.cc -----------------------------------------------------===//
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
#include "tsan_clock.h"
#include "tsan_rtl.h"

// It's possible to optimize clock operations for some important cases
// so that they are O(1). The cases include singletons, once's, local mutexes.
// First, SyncClock must be re-implemented to allow indexing by tid.
// It must not necessarily be a full vector clock, though. For example it may
// be a multi-level table.
// Then, each slot in SyncClock must contain a dirty bit (it's united with
// the clock value, so no space increase). The acquire algorithm looks
// as follows:
// void acquire(thr, tid, thr_clock, sync_clock) {
//   if (!sync_clock[tid].dirty)
//     return;  // No new info to acquire.
//              // This handles constant reads of singleton pointers and
//              // stop-flags.
//   acquire_impl(thr_clock, sync_clock);  // As usual, O(N).
//   sync_clock[tid].dirty = false;
//   sync_clock.dirty_count--;
// }
// The release operation looks as follows:
// void release(thr, tid, thr_clock, sync_clock) {
//   // thr->sync_cache is a simple fixed-size hash-based cache that holds
//   // several previous sync_clock's.
//   if (thr->sync_cache[sync_clock] >= thr->last_acquire_epoch) {
//     // The thread did no acquire operations since last release on this clock.
//     // So update only the thread's slot (other slots can't possibly change).
//     sync_clock[tid].clock = thr->epoch;
//     if (sync_clock.dirty_count == sync_clock.cnt
//         || (sync_clock.dirty_count == sync_clock.cnt - 1
//           && sync_clock[tid].dirty == false))
//       // All dirty flags are set, bail out.
//       return;
//     set all dirty bits, but preserve the thread's bit.  // O(N)
//     update sync_clock.dirty_count;
//     return;
//   }
//   release_impl(thr_clock, sync_clock);  // As usual, O(N).
//   set all dirty bits, but preserve the thread's bit.
//   // The previous step is combined with release_impl(), so that
//   // we scan the arrays only once.
//   update sync_clock.dirty_count;
// }

namespace __tsan {

ThreadClock::ThreadClock() {
  nclk_ = 0;
  for (uptr i = 0; i < (uptr)kMaxTidInClock; i++)
    clk_[i] = 0;
}

void ThreadClock::acquire(const SyncClock *src) {
  DCHECK(nclk_ <= kMaxTid);
  DCHECK(src->clk_.Size() <= kMaxTid);

  const uptr nclk = src->clk_.Size();
  if (nclk == 0)
    return;
  nclk_ = max(nclk_, nclk);
  for (uptr i = 0; i < nclk; i++) {
    if (clk_[i] < src->clk_[i])
      clk_[i] = src->clk_[i];
  }
}

void ThreadClock::release(SyncClock *dst) const {
  DCHECK(nclk_ <= kMaxTid);
  DCHECK(dst->clk_.Size() <= kMaxTid);

  if (dst->clk_.Size() < nclk_)
    dst->clk_.Resize(nclk_);
  for (uptr i = 0; i < nclk_; i++) {
    if (dst->clk_[i] < clk_[i])
      dst->clk_[i] = clk_[i];
  }
}

void ThreadClock::acq_rel(SyncClock *dst) {
  acquire(dst);
  release(dst);
}

SyncClock::SyncClock()
  : clk_(MBlockClock) {
}
}  // namespace __tsan
