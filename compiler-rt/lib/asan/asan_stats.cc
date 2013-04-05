//===-- asan_stats.cc -----------------------------------------------------===//
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
// Code related to statistics collected by AddressSanitizer.
//===----------------------------------------------------------------------===//
#include "asan_interceptors.h"
#include "asan_internal.h"
#include "asan_stats.h"
#include "asan_thread.h"
#include "sanitizer_common/sanitizer_mutex.h"
#include "sanitizer_common/sanitizer_stackdepot.h"

namespace __asan {

AsanStats::AsanStats() {
  CHECK(REAL(memset));
  REAL(memset)(this, 0, sizeof(AsanStats));
}

static void PrintMallocStatsArray(const char *prefix,
                                  uptr (&array)[kNumberOfSizeClasses]) {
  Printf("%s", prefix);
  for (uptr i = 0; i < kNumberOfSizeClasses; i++) {
    if (!array[i]) continue;
    Printf("%zu:%zu; ", i, array[i]);
  }
  Printf("\n");
}

void AsanStats::Print() {
  Printf("Stats: %zuM malloced (%zuM for red zones) by %zu calls\n",
             malloced>>20, malloced_redzones>>20, mallocs);
  Printf("Stats: %zuM realloced by %zu calls\n", realloced>>20, reallocs);
  Printf("Stats: %zuM freed by %zu calls\n", freed>>20, frees);
  Printf("Stats: %zuM really freed by %zu calls\n",
             really_freed>>20, real_frees);
  Printf("Stats: %zuM (%zuM-%zuM) mmaped; %zu maps, %zu unmaps\n",
             (mmaped-munmaped)>>20, mmaped>>20, munmaped>>20,
             mmaps, munmaps);

  PrintMallocStatsArray("  mmaps   by size class: ", mmaped_by_size);
  PrintMallocStatsArray("  mallocs by size class: ", malloced_by_size);
  PrintMallocStatsArray("  frees   by size class: ", freed_by_size);
  PrintMallocStatsArray("  rfrees  by size class: ", really_freed_by_size);
  Printf("Stats: malloc large: %zu small slow: %zu\n",
             malloc_large, malloc_small_slow);
}

static BlockingMutex print_lock(LINKER_INITIALIZED);

static void PrintAccumulatedStats() {
  AsanStats stats;
  GetAccumulatedStats(&stats);
  // Use lock to keep reports from mixing up.
  BlockingMutexLock lock(&print_lock);
  stats.Print();
  StackDepotStats *stack_depot_stats = StackDepotGetStats();
  Printf("Stats: StackDepot: %zd ids; %zdM mapped\n",
         stack_depot_stats->n_uniq_ids, stack_depot_stats->mapped >> 20);
  PrintInternalAllocatorStats();
}

static AsanStats unknown_thread_stats(LINKER_INITIALIZED);
static AsanStats accumulated_stats(LINKER_INITIALIZED);
// Required for malloc_zone_statistics() on OS X. This can't be stored in
// per-thread AsanStats.
static uptr max_malloced_memory;
static BlockingMutex acc_stats_lock(LINKER_INITIALIZED);

static void FlushToAccumulatedStatsUnlocked(AsanStats *stats) {
  acc_stats_lock.CheckLocked();
  uptr *dst = (uptr*)&accumulated_stats;
  uptr *src = (uptr*)stats;
  uptr num_fields = sizeof(*stats) / sizeof(uptr);
  for (uptr i = 0; i < num_fields; i++) {
    dst[i] += src[i];
    src[i] = 0;
  }
}

static void FlushThreadStats(ThreadContextBase *tctx_base, void *arg) {
  AsanThreadContext *tctx = static_cast<AsanThreadContext*>(tctx_base);
  if (AsanThread *t = tctx->thread)
    FlushToAccumulatedStatsUnlocked(&t->stats());
}

static void UpdateAccumulatedStatsUnlocked() {
  acc_stats_lock.CheckLocked();
  {
    ThreadRegistryLock l(&asanThreadRegistry());
    asanThreadRegistry().RunCallbackForEachThreadLocked(FlushThreadStats, 0);
  }
  FlushToAccumulatedStatsUnlocked(&unknown_thread_stats);
  // This is not very accurate: we may miss allocation peaks that happen
  // between two updates of accumulated_stats_. For more accurate bookkeeping
  // the maximum should be updated on every malloc(), which is unacceptable.
  if (max_malloced_memory < accumulated_stats.malloced) {
    max_malloced_memory = accumulated_stats.malloced;
  }
}

void FlushToAccumulatedStats(AsanStats *stats) {
  BlockingMutexLock lock(&acc_stats_lock);
  FlushToAccumulatedStatsUnlocked(stats);
}

void GetAccumulatedStats(AsanStats *stats) {
  BlockingMutexLock lock(&acc_stats_lock);
  UpdateAccumulatedStatsUnlocked();
  internal_memcpy(stats, &accumulated_stats, sizeof(accumulated_stats));
}

void FillMallocStatistics(AsanMallocStats *malloc_stats) {
  BlockingMutexLock lock(&acc_stats_lock);
  UpdateAccumulatedStatsUnlocked();
  malloc_stats->blocks_in_use = accumulated_stats.mallocs;
  malloc_stats->size_in_use = accumulated_stats.malloced;
  malloc_stats->max_size_in_use = max_malloced_memory;
  malloc_stats->size_allocated = accumulated_stats.mmaped;
}

AsanStats &GetCurrentThreadStats() {
  AsanThread *t = GetCurrentThread();
  return (t) ? t->stats() : unknown_thread_stats;
}

}  // namespace __asan

// ---------------------- Interface ---------------- {{{1
using namespace __asan;  // NOLINT

uptr __asan_get_current_allocated_bytes() {
  BlockingMutexLock lock(&acc_stats_lock);
  UpdateAccumulatedStatsUnlocked();
  uptr malloced = accumulated_stats.malloced;
  uptr freed = accumulated_stats.freed;
  // Return sane value if malloced < freed due to racy
  // way we update accumulated stats.
  return (malloced > freed) ? malloced - freed : 1;
}

uptr __asan_get_heap_size() {
  BlockingMutexLock lock(&acc_stats_lock);
  UpdateAccumulatedStatsUnlocked();
  return accumulated_stats.mmaped - accumulated_stats.munmaped;
}

uptr __asan_get_free_bytes() {
  BlockingMutexLock lock(&acc_stats_lock);
  UpdateAccumulatedStatsUnlocked();
  uptr total_free = accumulated_stats.mmaped
                  - accumulated_stats.munmaped
                  + accumulated_stats.really_freed
                  + accumulated_stats.really_freed_redzones;
  uptr total_used = accumulated_stats.malloced
                  + accumulated_stats.malloced_redzones;
  // Return sane value if total_free < total_used due to racy
  // way we update accumulated stats.
  return (total_free > total_used) ? total_free - total_used : 1;
}

uptr __asan_get_unmapped_bytes() {
  return 0;
}

void __asan_print_accumulated_stats() {
  PrintAccumulatedStats();
}
