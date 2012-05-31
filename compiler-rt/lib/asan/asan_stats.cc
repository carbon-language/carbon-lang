//===-- asan_stats.cc -------------------------------------------*- C++ -*-===//
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
#include "asan_interface.h"
#include "asan_internal.h"
#include "asan_lock.h"
#include "asan_stats.h"
#include "asan_thread_registry.h"

namespace __asan {

AsanStats::AsanStats() {
  CHECK(REAL(memset) != NULL);
  REAL(memset)(this, 0, sizeof(AsanStats));
}

static void PrintMallocStatsArray(const char *prefix,
                                  size_t (&array)[kNumberOfSizeClasses]) {
  Printf("%s", prefix);
  for (size_t i = 0; i < kNumberOfSizeClasses; i++) {
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
  Printf("Stats: %zuM (%zu full pages) mmaped in %zu calls\n",
         mmaped>>20, mmaped / kPageSize, mmaps);

  PrintMallocStatsArray("  mmaps   by size class: ", mmaped_by_size);
  PrintMallocStatsArray("  mallocs by size class: ", malloced_by_size);
  PrintMallocStatsArray("  frees   by size class: ", freed_by_size);
  PrintMallocStatsArray("  rfrees  by size class: ", really_freed_by_size);
  Printf("Stats: malloc large: %zu small slow: %zu\n",
         malloc_large, malloc_small_slow);
}

static AsanLock print_lock(LINKER_INITIALIZED);

static void PrintAccumulatedStats() {
  AsanStats stats = asanThreadRegistry().GetAccumulatedStats();
  // Use lock to keep reports from mixing up.
  ScopedLock lock(&print_lock);
  stats.Print();
}

}  // namespace __asan

// ---------------------- Interface ---------------- {{{1
using namespace __asan;  // NOLINT

uptr __asan_get_current_allocated_bytes() {
  return asanThreadRegistry().GetCurrentAllocatedBytes();
}

uptr __asan_get_heap_size() {
  return asanThreadRegistry().GetHeapSize();
}

uptr __asan_get_free_bytes() {
  return asanThreadRegistry().GetFreeBytes();
}

uptr __asan_get_unmapped_bytes() {
  return 0;
}

void __asan_print_accumulated_stats() {
  PrintAccumulatedStats();
}
