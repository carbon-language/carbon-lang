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
  // AsanStats must be a struct consisting of size_t fields only.
  // When merging two AsanStats structs, we treat them as arrays of size_t.
  size_t mallocs;
  size_t malloced;
  size_t malloced_redzones;
  size_t frees;
  size_t freed;
  size_t real_frees;
  size_t really_freed;
  size_t really_freed_redzones;
  size_t reallocs;
  size_t realloced;
  size_t mmaps;
  size_t mmaped;
  size_t mmaped_by_size[kNumberOfSizeClasses];
  size_t malloced_by_size[kNumberOfSizeClasses];
  size_t freed_by_size[kNumberOfSizeClasses];
  size_t really_freed_by_size[kNumberOfSizeClasses];

  size_t malloc_large;
  size_t malloc_small_slow;

  // Ctor for global AsanStats (accumulated stats and main thread stats).
  explicit AsanStats(LinkerInitialized) { }
  // Default ctor for thread-local stats.
  AsanStats();

  // Prints formatted stats to stderr.
  void Print();
};

}  // namespace __asan

#endif  // ASAN_STATS_H
