//===-- asan_thread_registry.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// ASan-private header for asan_thread_registry.cc
//===----------------------------------------------------------------------===//

#ifndef ASAN_THREAD_REGISTRY_H
#define ASAN_THREAD_REGISTRY_H

#include "asan_stack.h"
#include "asan_stats.h"
#include "asan_thread.h"
#include "sanitizer_common/sanitizer_mutex.h"

namespace __asan {

// Stores summaries of all created threads, returns current thread,
// thread by tid, thread by stack address. There is a single instance
// of AsanThreadRegistry for the whole program.
// AsanThreadRegistry is thread-safe.
class AsanThreadRegistry {
 public:
  explicit AsanThreadRegistry(LinkerInitialized);
  void Init();
  void RegisterThread(AsanThread *thread);
  void UnregisterThread(AsanThread *thread);

  AsanThread *GetMain();

  // Returns stats for GetCurrentThread(), or stats for
  // T0 if GetCurrentThread() returns 0.
  AsanStats &GetCurrentThreadStats();
  // Flushes all thread-local stats to accumulated stats, and makes
  // a copy of accumulated stats.
  void GetAccumulatedStats(AsanStats *stats);
  uptr GetCurrentAllocatedBytes();
  uptr GetHeapSize();
  uptr GetFreeBytes();
  void FillMallocStatistics(AsanMallocStats *malloc_stats);

  AsanThreadSummary *FindByTid(u32 tid);
  AsanThread *FindThreadByStackAddress(uptr addr);

 private:
  void UpdateAccumulatedStatsUnlocked();
  // Adds values of all counters in "stats" to accumulated stats,
  // and fills "stats" with zeroes.
  void FlushToAccumulatedStatsUnlocked(AsanStats *stats);

  static const u32 kMaxNumberOfThreads = (1 << 22);  // 4M
  AsanThreadSummary *thread_summaries_[kMaxNumberOfThreads];
  AsanThread main_thread_;
  AsanThreadSummary main_thread_summary_;
  AsanStats accumulated_stats_;
  // Required for malloc_zone_statistics() on OS X. This can't be stored in
  // per-thread AsanStats.
  uptr max_malloced_memory_;
  u32 n_threads_;
  BlockingMutex mu_;
  bool inited_;
};

// Returns a single instance of registry.
AsanThreadRegistry &asanThreadRegistry();

}  // namespace __asan

#endif  // ASAN_THREAD_REGISTRY_H
