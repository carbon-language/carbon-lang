//=-- lsan_common.h -------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of LeakSanitizer.
// Private LSan header.
//
//===----------------------------------------------------------------------===//

#ifndef LSAN_COMMON_H
#define LSAN_COMMON_H

#include "sanitizer_common/sanitizer_allocator.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_platform.h"
#include "sanitizer_common/sanitizer_symbolizer.h"

#if SANITIZER_LINUX && defined(__x86_64__)
#define CAN_SANITIZE_LEAKS 1
#else
#define CAN_SANITIZE_LEAKS 0
#endif

namespace __lsan {

// Chunk tags.
enum ChunkTag {
  kDirectlyLeaked = 0,  // default
  kIndirectlyLeaked = 1,
  kReachable = 2,
  kIgnored = 3
};

struct Flags {
  uptr pointer_alignment() const {
    return use_unaligned ? 1 : sizeof(uptr);
  }

  // Print addresses of leaked objects after main leak report.
  bool report_objects;
  // Aggregate two objects into one leak if this many stack frames match. If
  // zero, the entire stack trace must match.
  int resolution;
  // The number of leaks reported.
  int max_leaks;
  // If nonzero kill the process with this exit code upon finding leaks.
  int exitcode;

  // Flags controlling the root set of reachable memory.
  // Global variables (.data and .bss).
  bool use_globals;
  // Thread stacks.
  bool use_stacks;
  // Thread registers.
  bool use_registers;
  // TLS and thread-specific storage.
  bool use_tls;

  // Consider unaligned pointers valid.
  bool use_unaligned;

  // User-visible verbosity.
  int verbosity;

  // Debug logging.
  bool log_pointers;
  bool log_threads;
};

extern Flags lsan_flags;
inline Flags *flags() { return &lsan_flags; }

struct Leak {
  uptr hit_count;
  uptr total_size;
  u32 stack_trace_id;
  bool is_directly_leaked;
};

// Aggregates leaks by stack trace prefix.
class LeakReport {
 public:
  LeakReport() : leaks_(1) {}
  void Add(u32 stack_trace_id, uptr leaked_size, ChunkTag tag);
  void PrintLargest(uptr max_leaks);
  void PrintSummary();
  bool IsEmpty() { return leaks_.size() == 0; }
 private:
  InternalMmapVector<Leak> leaks_;
};

typedef InternalMmapVector<uptr> Frontier;

// Platform-specific functions.
void InitializePlatformSpecificModules();
void ProcessGlobalRegions(Frontier *frontier);
void ProcessPlatformSpecificAllocations(Frontier *frontier);

void ScanRangeForPointers(uptr begin, uptr end,
                          Frontier *frontier,
                          const char *region_type, ChunkTag tag);

enum IgnoreObjectResult {
  kIgnoreObjectSuccess,
  kIgnoreObjectAlreadyIgnored,
  kIgnoreObjectInvalid
};

// Functions called from the parent tool.
void InitCommonLsan();
void DoLeakCheck();
bool DisabledInThisThread();

// The following must be implemented in the parent tool.

void ForEachChunk(ForEachChunkCallback callback, void *arg);
// Returns the address range occupied by the global allocator object.
void GetAllocatorGlobalRange(uptr *begin, uptr *end);
// Wrappers for allocator's ForceLock()/ForceUnlock().
void LockAllocator();
void UnlockAllocator();
// Wrappers for ThreadRegistry access.
void LockThreadRegistry();
void UnlockThreadRegistry();
bool GetThreadRangesLocked(uptr os_id, uptr *stack_begin, uptr *stack_end,
                           uptr *tls_begin, uptr *tls_end,
                           uptr *cache_begin, uptr *cache_end);
// If p points into a chunk that has been allocated to the user, returns its
// user-visible address. Otherwise, returns 0.
uptr PointsIntoChunk(void *p);
// Returns address of user-visible chunk contained in this allocator chunk.
uptr GetUserBegin(uptr chunk);
// Helper for __lsan_ignore_object().
IgnoreObjectResult IgnoreObjectLocked(const void *p);
// Wrapper for chunk metadata operations.
class LsanMetadata {
 public:
  // Constructor accepts address of user-visible chunk.
  explicit LsanMetadata(uptr chunk);
  bool allocated() const;
  ChunkTag tag() const;
  void set_tag(ChunkTag value);
  uptr requested_size() const;
  u32 stack_trace_id() const;
 private:
  void *metadata_;
};

}  // namespace __lsan

#endif  // LSAN_COMMON_H
