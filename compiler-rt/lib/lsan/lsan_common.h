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
  kReachable = 2
};

// Sources of pointers.
// Global variables (.data and .bss).
const uptr kSourceGlobals = 1 << 0;
// Thread stacks.
const uptr kSourceStacks = 1 << 1;
// TLS and thread-specific storage.
const uptr kSourceTLS = 1 << 2;
// Thread registers.
const uptr kSourceRegisters = 1 << 3;
// Unaligned pointers.
const uptr kSourceUnaligned = 1 << 4;

// Aligned pointers everywhere.
const uptr kSourceAllAligned =
    kSourceGlobals | kSourceStacks | kSourceTLS | kSourceRegisters;

struct Flags {
  bool use_registers() const { return sources & kSourceRegisters; }
  bool use_globals() const { return sources & kSourceGlobals; }
  bool use_stacks() const { return sources & kSourceStacks; }
  bool use_tls() const { return sources & kSourceTLS; }
  uptr pointer_alignment() const {
    return (sources & kSourceUnaligned) ? 1 : sizeof(uptr);
  }

  uptr sources;
  // Print addresses of leaked blocks after main leak report.
  bool report_blocks;
  // Aggregate two blocks into one leak if this many stack frames match. If
  // zero, the entire stack trace must match.
  int resolution;
  // The number of leaks reported.
  int max_leaks;
  // If nonzero kill the process with this exit code upon finding leaks.
  int exitcode;

  // Debug logging.
  bool log_pointers;
  bool log_threads;
};

extern Flags lsan_flags;
inline Flags *flags() { return &lsan_flags; }

void InitCommonLsan();
// Testing interface. Find leaked chunks and dump their addresses to vector.
void ReportLeaked(InternalVector<void *> *leaked, uptr sources);
// Normal leak check. Find leaks and print a report according to flags.
void DoLeakCheck();

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
  InternalVector<Leak> leaks_;
};

// Platform-specific functions.
void InitializePlatformSpecificModules();
void ProcessGlobalRegions(InternalVector<uptr> *frontier);
void ProcessPlatformSpecificAllocations(InternalVector<uptr> *frontier);

void ScanRangeForPointers(uptr begin, uptr end, InternalVector<uptr> *frontier,
                          const char *region_type, ChunkTag tag);

// Callables for iterating over chunks. Those classes are used as template
// parameters in ForEachChunk, so we must expose them here to allow for explicit
// template instantiation.

// Identifies unreachable chunks which must be treated as reachable. Marks them
// as reachable and adds them to the frontier.
class ProcessPlatformSpecificAllocationsCb {
 public:
  explicit ProcessPlatformSpecificAllocationsCb(InternalVector<uptr> *frontier)
      : frontier_(frontier) {}
  void operator()(void *p) const;
 private:
  InternalVector<uptr> *frontier_;
};

// Prints addresses of unreachable chunks.
class PrintLeakedCb {
 public:
  void operator()(void *p) const;
};

// Aggregates unreachable chunks into a LeakReport.
class CollectLeaksCb {
 public:
  explicit CollectLeaksCb(LeakReport *leak_report)
      : leak_report_(leak_report) {}
  void operator()(void *p) const;
 private:
  LeakReport *leak_report_;
};

// Dumps addresses of unreachable chunks to a vector (for testing).
class ReportLeakedCb {
 public:
  explicit ReportLeakedCb(InternalVector<void *> *leaked) : leaked_(leaked) {}
  void operator()(void *p) const;
 private:
  InternalVector<void *> *leaked_;
};

// Resets each chunk's tag to default (kDirectlyLeaked).
class ClearTagCb {
 public:
  void operator()(void *p) const;
};

// Scans each leaked chunk for pointers to other leaked chunks, and marks each
// of them as indirectly leaked.
class MarkIndirectlyLeakedCb {
 public:
  void operator()(void *p) const;
};

// The following must be implemented in the parent tool.

template<typename Callable> void ForEachChunk(Callable const &callback);
// The address range occupied by the global allocator object.
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
// If p points into a chunk that has been allocated to the user, return its
// user-visible address. Otherwise, return 0.
void *PointsIntoChunk(void *p);
// Return address of user-visible chunk contained in this allocator chunk.
void *GetUserBegin(void *p);
// Wrapper for chunk metadata operations.
class LsanMetadata {
 public:
  // Constructor accepts pointer to user-visible chunk.
  explicit LsanMetadata(void *chunk);
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
