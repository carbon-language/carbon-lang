//=-- lsan_common.cc ------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of LeakSanitizer.
// Implementation of common leak checking functionality.
//
//===----------------------------------------------------------------------===//

#include "lsan_common.h"

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_stackdepot.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "sanitizer_common/sanitizer_stoptheworld.h"

#if CAN_SANITIZE_LEAKS
namespace __lsan {

// This mutex is used to prevent races between DoLeakCheck and SuppressObject.
BlockingMutex global_mutex(LINKER_INITIALIZED);

THREADLOCAL int disable_counter;
bool DisabledInThisThread() { return disable_counter > 0; }

Flags lsan_flags;

static void InitializeFlags() {
  Flags *f = flags();
  // Default values.
  f->report_objects = false;
  f->resolution = 0;
  f->max_leaks = 0;
  f->exitcode = 23;
  f->use_registers = true;
  f->use_globals = true;
  f->use_stacks = true;
  f->use_tls = true;
  f->use_unaligned = false;
  f->verbosity = 0;
  f->log_pointers = false;
  f->log_threads = false;

  const char *options = GetEnv("LSAN_OPTIONS");
  if (options) {
    ParseFlag(options, &f->use_registers, "use_registers");
    ParseFlag(options, &f->use_globals, "use_globals");
    ParseFlag(options, &f->use_stacks, "use_stacks");
    ParseFlag(options, &f->use_tls, "use_tls");
    ParseFlag(options, &f->use_unaligned, "use_unaligned");
    ParseFlag(options, &f->report_objects, "report_objects");
    ParseFlag(options, &f->resolution, "resolution");
    CHECK_GE(&f->resolution, 0);
    ParseFlag(options, &f->max_leaks, "max_leaks");
    CHECK_GE(&f->max_leaks, 0);
    ParseFlag(options, &f->verbosity, "verbosity");
    ParseFlag(options, &f->log_pointers, "log_pointers");
    ParseFlag(options, &f->log_threads, "log_threads");
    ParseFlag(options, &f->exitcode, "exitcode");
  }
}

void InitCommonLsan() {
  InitializeFlags();
  InitializePlatformSpecificModules();
}

static inline bool CanBeAHeapPointer(uptr p) {
  // Since our heap is located in mmap-ed memory, we can assume a sensible lower
  // boundary on heap addresses.
  const uptr kMinAddress = 4 * 4096;
  if (p < kMinAddress) return false;
#ifdef __x86_64__
  // Accept only canonical form user-space addresses.
  return ((p >> 47) == 0);
#else
  return true;
#endif
}

// Scan the memory range, looking for byte patterns that point into allocator
// chunks. Mark those chunks with tag and add them to the frontier.
// There are two usage modes for this function: finding reachable or ignored 
// chunks (tag = kReachable or kIgnored) and finding indirectly leaked chunks
// (tag = kIndirectlyLeaked). In the second case, there's no flood fill,
// so frontier = 0.
void ScanRangeForPointers(uptr begin, uptr end,
                          Frontier *frontier,
                          const char *region_type, ChunkTag tag) {
  const uptr alignment = flags()->pointer_alignment();
  if (flags()->log_pointers)
    Report("Scanning %s range %p-%p.\n", region_type, begin, end);
  uptr pp = begin;
  if (pp % alignment)
    pp = pp + alignment - pp % alignment;
  for (; pp + sizeof(void *) <= end; pp += alignment) {
    void *p = *reinterpret_cast<void**>(pp);
    if (!CanBeAHeapPointer(reinterpret_cast<uptr>(p))) continue;
    void *chunk = PointsIntoChunk(p);
    if (!chunk) continue;
    LsanMetadata m(chunk);
    // Reachable beats ignored beats leaked.
    if (m.tag() == kReachable) continue;
    if (m.tag() == kIgnored && tag != kReachable) continue;
    m.set_tag(tag);
    if (flags()->log_pointers)
      Report("%p: found %p pointing into chunk %p-%p of size %zu.\n", pp, p,
             chunk, reinterpret_cast<uptr>(chunk) + m.requested_size(),
             m.requested_size());
    if (frontier)
      frontier->push_back(reinterpret_cast<uptr>(chunk));
  }
}

// Scan thread data (stacks and TLS) for heap pointers.
static void ProcessThreads(SuspendedThreadsList const &suspended_threads,
                           Frontier *frontier) {
  InternalScopedBuffer<uptr> registers(SuspendedThreadsList::RegisterCount());
  uptr registers_begin = reinterpret_cast<uptr>(registers.data());
  uptr registers_end = registers_begin + registers.size();
  for (uptr i = 0; i < suspended_threads.thread_count(); i++) {
    uptr os_id = static_cast<uptr>(suspended_threads.GetThreadID(i));
    if (flags()->log_threads) Report("Processing thread %d.\n", os_id);
    uptr stack_begin, stack_end, tls_begin, tls_end, cache_begin, cache_end;
    bool thread_found = GetThreadRangesLocked(os_id, &stack_begin, &stack_end,
                                              &tls_begin, &tls_end,
                                              &cache_begin, &cache_end);
    if (!thread_found) {
      // If a thread can't be found in the thread registry, it's probably in the
      // process of destruction. Log this event and move on.
      if (flags()->log_threads)
        Report("Thread %d not found in registry.\n", os_id);
      continue;
    }
    uptr sp;
    bool have_registers =
        (suspended_threads.GetRegistersAndSP(i, registers.data(), &sp) == 0);
    if (!have_registers) {
      Report("Unable to get registers from thread %d.\n");
      // If unable to get SP, consider the entire stack to be reachable.
      sp = stack_begin;
    }

    if (flags()->use_registers && have_registers)
      ScanRangeForPointers(registers_begin, registers_end, frontier,
                           "REGISTERS", kReachable);

    if (flags()->use_stacks) {
      if (flags()->log_threads)
        Report("Stack at %p-%p, SP = %p.\n", stack_begin, stack_end, sp);
      if (sp < stack_begin || sp >= stack_end) {
        // SP is outside the recorded stack range (e.g. the thread is running a
        // signal handler on alternate stack). Again, consider the entire stack
        // range to be reachable.
        if (flags()->log_threads)
          Report("WARNING: stack_pointer not in stack_range.\n");
      } else {
        // Shrink the stack range to ignore out-of-scope values.
        stack_begin = sp;
      }
      ScanRangeForPointers(stack_begin, stack_end, frontier, "STACK",
                           kReachable);
    }

    if (flags()->use_tls) {
      if (flags()->log_threads) Report("TLS at %p-%p.\n", tls_begin, tls_end);
      if (cache_begin == cache_end) {
        ScanRangeForPointers(tls_begin, tls_end, frontier, "TLS", kReachable);
      } else {
        // Because LSan should not be loaded with dlopen(), we can assume
        // that allocator cache will be part of static TLS image.
        CHECK_LE(tls_begin, cache_begin);
        CHECK_GE(tls_end, cache_end);
        if (tls_begin < cache_begin)
          ScanRangeForPointers(tls_begin, cache_begin, frontier, "TLS",
                               kReachable);
        if (tls_end > cache_end)
          ScanRangeForPointers(cache_end, tls_end, frontier, "TLS", kReachable);
      }
    }
  }
}

static void FloodFillTag(Frontier *frontier, ChunkTag tag) {
  while (frontier->size()) {
    uptr next_chunk = frontier->back();
    frontier->pop_back();
    LsanMetadata m(reinterpret_cast<void *>(next_chunk));
    ScanRangeForPointers(next_chunk, next_chunk + m.requested_size(), frontier,
                         "HEAP", tag);
  }
}

// Mark leaked chunks which are reachable from other leaked chunks.
void MarkIndirectlyLeakedCb::operator()(void *p) const {
  p = GetUserBegin(p);
  LsanMetadata m(p);
  if (m.allocated() && m.tag() != kReachable) {
    ScanRangeForPointers(reinterpret_cast<uptr>(p),
                         reinterpret_cast<uptr>(p) + m.requested_size(),
                         /* frontier */ 0, "HEAP", kIndirectlyLeaked);
  }
}

void CollectIgnoredCb::operator()(void *p) const {
  p = GetUserBegin(p);
  LsanMetadata m(p);
  if (m.allocated() && m.tag() == kIgnored)
    frontier_->push_back(reinterpret_cast<uptr>(p));
}

// Set the appropriate tag on each chunk.
static void ClassifyAllChunks(SuspendedThreadsList const &suspended_threads) {
  // Holds the flood fill frontier.
  Frontier frontier(GetPageSizeCached());

  if (flags()->use_globals)
    ProcessGlobalRegions(&frontier);
  ProcessThreads(suspended_threads, &frontier);
  FloodFillTag(&frontier, kReachable);
  // The check here is relatively expensive, so we do this in a separate flood
  // fill. That way we can skip the check for chunks that are reachable
  // otherwise.
  ProcessPlatformSpecificAllocations(&frontier);
  FloodFillTag(&frontier, kReachable);

  if (flags()->log_pointers)
    Report("Scanning ignored chunks.\n");
  CHECK_EQ(0, frontier.size());
  ForEachChunk(CollectIgnoredCb(&frontier));
  FloodFillTag(&frontier, kIgnored);

  // Iterate over leaked chunks and mark those that are reachable from other
  // leaked chunks.
  if (flags()->log_pointers)
    Report("Scanning leaked chunks.\n");
  ForEachChunk(MarkIndirectlyLeakedCb());
}

static void PrintStackTraceById(u32 stack_trace_id) {
  CHECK(stack_trace_id);
  uptr size = 0;
  const uptr *trace = StackDepotGet(stack_trace_id, &size);
  StackTrace::PrintStack(trace, size, common_flags()->symbolize,
                         common_flags()->strip_path_prefix, 0);
}

void CollectLeaksCb::operator()(void *p) const {
  p = GetUserBegin(p);
  LsanMetadata m(p);
  if (!m.allocated()) return;
  if (m.tag() == kDirectlyLeaked || m.tag() == kIndirectlyLeaked) {
    uptr resolution = flags()->resolution;
    if (resolution > 0) {
      uptr size = 0;
      const uptr *trace = StackDepotGet(m.stack_trace_id(), &size);
      size = Min(size, resolution);
      leak_report_->Add(StackDepotPut(trace, size), m.requested_size(),
                        m.tag());
    } else {
      leak_report_->Add(m.stack_trace_id(), m.requested_size(), m.tag());
    }
  }
}

static void CollectLeaks(LeakReport *leak_report) {
  ForEachChunk(CollectLeaksCb(leak_report));
}

void PrintLeakedCb::operator()(void *p) const {
  p = GetUserBegin(p);
  LsanMetadata m(p);
  if (!m.allocated()) return;
  if (m.tag() == kDirectlyLeaked || m.tag() == kIndirectlyLeaked) {
    Printf("%s leaked %zu byte object at %p.\n",
           m.tag() == kDirectlyLeaked ? "Directly" : "Indirectly",
           m.requested_size(), p);
  }
}

static void PrintLeaked() {
  Printf("\n");
  Printf("Reporting individual objects:\n");
  ForEachChunk(PrintLeakedCb());
}

struct DoLeakCheckParam {
  bool success;
  LeakReport leak_report;
};

static void DoLeakCheckCallback(const SuspendedThreadsList &suspended_threads,
                                void *arg) {
  DoLeakCheckParam *param = reinterpret_cast<DoLeakCheckParam *>(arg);
  CHECK(param);
  CHECK(!param->success);
  CHECK(param->leak_report.IsEmpty());
  ClassifyAllChunks(suspended_threads);
  CollectLeaks(&param->leak_report);
  if (!param->leak_report.IsEmpty() && flags()->report_objects)
    PrintLeaked();
  param->success = true;
}

void DoLeakCheck() {
  BlockingMutexLock l(&global_mutex);
  static bool already_done;
  CHECK(!already_done);
  already_done = true;

  DoLeakCheckParam param;
  param.success = false;
  LockThreadRegistry();
  LockAllocator();
  StopTheWorld(DoLeakCheckCallback, &param);
  UnlockAllocator();
  UnlockThreadRegistry();

  if (!param.success) {
    Report("LeakSanitizer has encountered a fatal error.\n");
    Die();
  }
  if (!param.leak_report.IsEmpty()) {
    Printf("\n================================================================="
           "\n");
    Report("ERROR: LeakSanitizer: detected memory leaks\n");
    param.leak_report.PrintLargest(flags()->max_leaks);
    param.leak_report.PrintSummary();
    if (flags()->exitcode)
      internal__exit(flags()->exitcode);
  }
}

///// LeakReport implementation. /////

// A hard limit on the number of distinct leaks, to avoid quadratic complexity
// in LeakReport::Add(). We don't expect to ever see this many leaks in
// real-world applications.
// FIXME: Get rid of this limit by changing the implementation of LeakReport to
// use a hash table.
const uptr kMaxLeaksConsidered = 1000;

void LeakReport::Add(u32 stack_trace_id, uptr leaked_size, ChunkTag tag) {
  CHECK(tag == kDirectlyLeaked || tag == kIndirectlyLeaked);
  bool is_directly_leaked = (tag == kDirectlyLeaked);
  for (uptr i = 0; i < leaks_.size(); i++)
    if (leaks_[i].stack_trace_id == stack_trace_id &&
        leaks_[i].is_directly_leaked == is_directly_leaked) {
      leaks_[i].hit_count++;
      leaks_[i].total_size += leaked_size;
      return;
    }
  if (leaks_.size() == kMaxLeaksConsidered) return;
  Leak leak = { /* hit_count */ 1, leaked_size, stack_trace_id,
                is_directly_leaked };
  leaks_.push_back(leak);
}

static bool IsLarger(const Leak &leak1, const Leak &leak2) {
  return leak1.total_size > leak2.total_size;
}

void LeakReport::PrintLargest(uptr max_leaks) {
  CHECK(leaks_.size() <= kMaxLeaksConsidered);
  Printf("\n");
  if (leaks_.size() == kMaxLeaksConsidered)
    Printf("Too many leaks! Only the first %zu leaks encountered will be "
           "reported.\n",
           kMaxLeaksConsidered);
  if (max_leaks > 0 && max_leaks < leaks_.size())
    Printf("The %zu largest leak(s):\n", max_leaks);
  InternalSort(&leaks_, leaks_.size(), IsLarger);
  max_leaks = max_leaks > 0 ? Min(max_leaks, leaks_.size()) : leaks_.size();
  for (uptr i = 0; i < max_leaks; i++) {
    Printf("%s leak of %zu byte(s) in %zu object(s) allocated from:\n",
           leaks_[i].is_directly_leaked ? "Direct" : "Indirect",
           leaks_[i].total_size, leaks_[i].hit_count);
    PrintStackTraceById(leaks_[i].stack_trace_id);
    Printf("\n");
  }
  if (max_leaks < leaks_.size()) {
    uptr remaining = leaks_.size() - max_leaks;
    Printf("Omitting %zu more leak(s).\n", remaining);
  }
}

void LeakReport::PrintSummary() {
  CHECK(leaks_.size() <= kMaxLeaksConsidered);
  uptr bytes = 0, allocations = 0;
  for (uptr i = 0; i < leaks_.size(); i++) {
      bytes += leaks_[i].total_size;
      allocations += leaks_[i].hit_count;
  }
  Printf(
      "SUMMARY: LeakSanitizer: %zu byte(s) leaked in %zu allocation(s).\n\n",
      bytes, allocations);
}
}  // namespace __lsan
#endif  // CAN_SANITIZE_LEAKS

using namespace __lsan;  // NOLINT

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE
void __lsan_ignore_object(const void *p) {
#if CAN_SANITIZE_LEAKS
  // Cannot use PointsIntoChunk or LsanMetadata here, since the allocator is not
  // locked.
  BlockingMutexLock l(&global_mutex);
  IgnoreObjectResult res = IgnoreObjectLocked(p);
  if (res == kIgnoreObjectInvalid && flags()->verbosity >= 1)
    Report("__lsan_ignore_object(): no heap object found at %p", p);
  if (res == kIgnoreObjectAlreadyIgnored && flags()->verbosity >= 1)
    Report("__lsan_ignore_object(): "
           "heap object at %p is already being ignored\n", p);
  if (res == kIgnoreObjectSuccess && flags()->verbosity >= 2)
    Report("__lsan_ignore_object(): ignoring heap object at %p\n", p);
#endif  // CAN_SANITIZE_LEAKS
}

SANITIZER_INTERFACE_ATTRIBUTE
void __lsan_disable() {
#if CAN_SANITIZE_LEAKS
  __lsan::disable_counter++;
#endif
}

SANITIZER_INTERFACE_ATTRIBUTE
void __lsan_enable() {
#if CAN_SANITIZE_LEAKS
  if (!__lsan::disable_counter) {
    Report("Unmatched call to __lsan_enable().\n");
    Die();
  }
  __lsan::disable_counter--;
#endif
}
}  // extern "C"
