//===-- asan_stats.h --------------------------------------------*- C++ -*-===//
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
// ASan-private header for statistics.
//===----------------------------------------------------------------------===//
#ifndef ASAN_STATS_H
#define ASAN_STATS_H

#include "asan_allocator.h"
#include "asan_internal.h"

namespace __asan {

// AsanStats struct is NOT thread-safe.
// Each AsanThread has its own AsanStats, which are sometimes flushed
// to the accumulated AsanStats.
struct AsanStats {
  // AsanStats must be a struct consisting of uptr fields only.
  // When merging two AsanStats structs, we treat them as arrays of uptr.
  uptr mallocs;
  uptr malloced;
  uptr malloced_redzones;
  uptr frees;
  uptr freed;
  uptr real_frees;
  uptr really_freed;
  uptr really_freed_redzones;
  uptr reallocs;
  uptr realloced;
  uptr mmaps;
  uptr mmaped;
  uptr mmaped_by_size[kNumberOfSizeClasses];
  uptr malloced_by_size[kNumberOfSizeClasses];
  uptr freed_by_size[kNumberOfSizeClasses];
  uptr really_freed_by_size[kNumberOfSizeClasses];

  uptr malloc_large;
  uptr malloc_small_slow;

  // Ctor for global AsanStats (accumulated stats and main thread stats).
  explicit AsanStats(LinkerInitialized) { }
  // Default ctor for thread-local stats.
  AsanStats();

  // Prints formatted stats to stderr.
  void Print();
};

// A cross-platform equivalent of malloc_statistics_t on Mac OS.
struct AsanMallocStats {
  uptr blocks_in_use;
  uptr size_in_use;
  uptr max_size_in_use;
  uptr size_allocated;
};

}  // namespace __asan

#endif  // ASAN_STATS_H
