//===-- tsan_rtl.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
// Main internal TSan header file.
//
// Ground rules:
//   - C++ run-time should not be used (static CTORs, RTTI, exceptions, static
//     function-scope locals)
//   - All functions/classes/etc reside in namespace __tsan, except for those
//     declared in tsan_interface.h.
//   - Platform-specific files should be used instead of ifdefs (*).
//   - No system headers included in header files (*).
//   - Platform specific headres included only into platform-specific files (*).
//
//  (*) Except when inlining is critical for performance.
//===----------------------------------------------------------------------===//

#ifndef TSAN_RTL_H
#define TSAN_RTL_H

#include "sanitizer_common/sanitizer_allocator.h"
#include "sanitizer_common/sanitizer_allocator_internal.h"
#include "sanitizer_common/sanitizer_asm.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_deadlock_detector_interface.h"
#include "sanitizer_common/sanitizer_libignore.h"
#include "sanitizer_common/sanitizer_suppressions.h"
#include "sanitizer_common/sanitizer_thread_registry.h"
#include "sanitizer_common/sanitizer_vector.h"
#include "tsan_clock.h"
#include "tsan_defs.h"
#include "tsan_flags.h"
#include "tsan_ignoreset.h"
#include "tsan_mman.h"
#include "tsan_mutexset.h"
#include "tsan_platform.h"
#include "tsan_report.h"
#include "tsan_shadow.h"
#include "tsan_stack_trace.h"
#include "tsan_sync.h"
#include "tsan_trace.h"

#if SANITIZER_WORDSIZE != 64
# error "ThreadSanitizer is supported only on 64-bit platforms"
#endif

namespace __tsan {

#if !SANITIZER_GO
struct MapUnmapCallback;
#if defined(__mips64) || defined(__aarch64__) || defined(__powerpc__)

struct AP32 {
  static const uptr kSpaceBeg = 0;
  static const u64 kSpaceSize = SANITIZER_MMAP_RANGE_SIZE;
  static const uptr kMetadataSize = 0;
  typedef __sanitizer::CompactSizeClassMap SizeClassMap;
  static const uptr kRegionSizeLog = 20;
  using AddressSpaceView = LocalAddressSpaceView;
  typedef __tsan::MapUnmapCallback MapUnmapCallback;
  static const uptr kFlags = 0;
};
typedef SizeClassAllocator32<AP32> PrimaryAllocator;
#else
struct AP64 {  // Allocator64 parameters. Deliberately using a short name.
#    if defined(__s390x__)
  typedef MappingS390x Mapping;
#    else
  typedef Mapping48AddressSpace Mapping;
#    endif
  static const uptr kSpaceBeg = Mapping::kHeapMemBeg;
  static const uptr kSpaceSize = Mapping::kHeapMemEnd - Mapping::kHeapMemBeg;
  static const uptr kMetadataSize = 0;
  typedef DefaultSizeClassMap SizeClassMap;
  typedef __tsan::MapUnmapCallback MapUnmapCallback;
  static const uptr kFlags = 0;
  using AddressSpaceView = LocalAddressSpaceView;
};
typedef SizeClassAllocator64<AP64> PrimaryAllocator;
#endif
typedef CombinedAllocator<PrimaryAllocator> Allocator;
typedef Allocator::AllocatorCache AllocatorCache;
Allocator *allocator();
#endif

struct ThreadSignalContext;

struct JmpBuf {
  uptr sp;
  int int_signal_send;
  bool in_blocking_func;
  uptr in_signal_handler;
  uptr *shadow_stack_pos;
};

// A Processor represents a physical thread, or a P for Go.
// It is used to store internal resources like allocate cache, and does not
// participate in race-detection logic (invisible to end user).
// In C++ it is tied to an OS thread just like ThreadState, however ideally
// it should be tied to a CPU (this way we will have fewer allocator caches).
// In Go it is tied to a P, so there are significantly fewer Processor's than
// ThreadState's (which are tied to Gs).
// A ThreadState must be wired with a Processor to handle events.
struct Processor {
  ThreadState *thr; // currently wired thread, or nullptr
#if !SANITIZER_GO
  AllocatorCache alloc_cache;
  InternalAllocatorCache internal_alloc_cache;
#endif
  DenseSlabAllocCache block_cache;
  DenseSlabAllocCache sync_cache;
  DenseSlabAllocCache clock_cache;
  DDPhysicalThread *dd_pt;
};

#if !SANITIZER_GO
// ScopedGlobalProcessor temporary setups a global processor for the current
// thread, if it does not have one. Intended for interceptors that can run
// at the very thread end, when we already destroyed the thread processor.
struct ScopedGlobalProcessor {
  ScopedGlobalProcessor();
  ~ScopedGlobalProcessor();
};
#endif

// This struct is stored in TLS.
struct ThreadState {
  FastState fast_state;
  // Synch epoch represents the threads's epoch before the last synchronization
  // action. It allows to reduce number of shadow state updates.
  // For example, fast_synch_epoch=100, last write to addr X was at epoch=150,
  // if we are processing write to X from the same thread at epoch=200,
  // we do nothing, because both writes happen in the same 'synch epoch'.
  // That is, if another memory access does not race with the former write,
  // it does not race with the latter as well.
  // QUESTION: can we can squeeze this into ThreadState::Fast?
  // E.g. ThreadState::Fast is a 44-bit, 32 are taken by synch_epoch and 12 are
  // taken by epoch between synchs.
  // This way we can save one load from tls.
  u64 fast_synch_epoch;
  // Technically `current` should be a separate THREADLOCAL variable;
  // but it is placed here in order to share cache line with previous fields.
  ThreadState* current;
  // This is a slow path flag. On fast path, fast_state.GetIgnoreBit() is read.
  // We do not distinguish beteween ignoring reads and writes
  // for better performance.
  int ignore_reads_and_writes;
  atomic_sint32_t pending_signals;
  int ignore_sync;
  int suppress_reports;
  // Go does not support ignores.
#if !SANITIZER_GO
  IgnoreSet mop_ignore_set;
  IgnoreSet sync_ignore_set;
#endif
  uptr *shadow_stack;
  uptr *shadow_stack_end;
  uptr *shadow_stack_pos;
  RawShadow *racy_shadow_addr;
  RawShadow racy_state[2];
  MutexSet mset;
  ThreadClock clock;
#if !SANITIZER_GO
  Vector<JmpBuf> jmp_bufs;
  int ignore_interceptors;
#endif
  const Tid tid;
  const int unique_id;
  bool in_symbolizer;
  bool in_ignored_lib;
  bool is_inited;
  bool is_dead;
  bool is_freeing;
  bool is_vptr_access;
  const uptr stk_addr;
  const uptr stk_size;
  const uptr tls_addr;
  const uptr tls_size;
  ThreadContext *tctx;

  DDLogicalThread *dd_lt;

  // Current wired Processor, or nullptr. Required to handle any events.
  Processor *proc1;
#if !SANITIZER_GO
  Processor *proc() { return proc1; }
#else
  Processor *proc();
#endif

  atomic_uintptr_t in_signal_handler;
  ThreadSignalContext *signal_ctx;

#if !SANITIZER_GO
  StackID last_sleep_stack_id;
  ThreadClock last_sleep_clock;
#endif

  // Set in regions of runtime that must be signal-safe and fork-safe.
  // If set, malloc must not be called.
  int nomalloc;

  const ReportDesc *current_report;

  // Current position in tctx->trace.Back()->events (Event*).
  atomic_uintptr_t trace_pos;
  // PC of the last memory access, used to compute PC deltas in the trace.
  uptr trace_prev_pc;
  Sid sid;
  Epoch epoch;

  explicit ThreadState(Context *ctx, Tid tid, int unique_id, u64 epoch,
                       unsigned reuse_count, uptr stk_addr, uptr stk_size,
                       uptr tls_addr, uptr tls_size);
} ALIGNED(SANITIZER_CACHE_LINE_SIZE);

#if !SANITIZER_GO
#if SANITIZER_APPLE || SANITIZER_ANDROID
ThreadState *cur_thread();
void set_cur_thread(ThreadState *thr);
void cur_thread_finalize();
inline ThreadState *cur_thread_init() { return cur_thread(); }
#  else
__attribute__((tls_model("initial-exec")))
extern THREADLOCAL char cur_thread_placeholder[];
inline ThreadState *cur_thread() {
  return reinterpret_cast<ThreadState *>(cur_thread_placeholder)->current;
}
inline ThreadState *cur_thread_init() {
  ThreadState *thr = reinterpret_cast<ThreadState *>(cur_thread_placeholder);
  if (UNLIKELY(!thr->current))
    thr->current = thr;
  return thr->current;
}
inline void set_cur_thread(ThreadState *thr) {
  reinterpret_cast<ThreadState *>(cur_thread_placeholder)->current = thr;
}
inline void cur_thread_finalize() { }
#  endif  // SANITIZER_APPLE || SANITIZER_ANDROID
#endif  // SANITIZER_GO

class ThreadContext final : public ThreadContextBase {
 public:
  explicit ThreadContext(Tid tid);
  ~ThreadContext();
  ThreadState *thr;
  StackID creation_stack_id;
  SyncClock sync;
  // Epoch at which the thread had started.
  // If we see an event from the thread stamped by an older epoch,
  // the event is from a dead thread that shared tid with this thread.
  u64 epoch0;
  u64 epoch1;

  v3::Trace trace;

  // Override superclass callbacks.
  void OnDead() override;
  void OnJoined(void *arg) override;
  void OnFinished() override;
  void OnStarted(void *arg) override;
  void OnCreated(void *arg) override;
  void OnReset() override;
  void OnDetached(void *arg) override;
};

struct RacyStacks {
  MD5Hash hash[2];
  bool operator==(const RacyStacks &other) const;
};

struct RacyAddress {
  uptr addr_min;
  uptr addr_max;
};

struct FiredSuppression {
  ReportType type;
  uptr pc_or_addr;
  Suppression *supp;
};

struct Context {
  Context();

  bool initialized;
#if !SANITIZER_GO
  bool after_multithreaded_fork;
#endif

  MetaMap metamap;

  Mutex report_mtx;
  int nreported;
  atomic_uint64_t last_symbolize_time_ns;

  void *background_thread;
  atomic_uint32_t stop_background_thread;

  ThreadRegistry thread_registry;

  Mutex racy_mtx;
  Vector<RacyStacks> racy_stacks;
  Vector<RacyAddress> racy_addresses;
  // Number of fired suppressions may be large enough.
  Mutex fired_suppressions_mtx;
  InternalMmapVector<FiredSuppression> fired_suppressions;
  DDetector *dd;

  ClockAlloc clock_alloc;

  Flags flags;
  fd_t memprof_fd;

  Mutex slot_mtx;
};

extern Context *ctx;  // The one and the only global runtime context.

ALWAYS_INLINE Flags *flags() {
  return &ctx->flags;
}

struct ScopedIgnoreInterceptors {
  ScopedIgnoreInterceptors() {
#if !SANITIZER_GO
    cur_thread()->ignore_interceptors++;
#endif
  }

  ~ScopedIgnoreInterceptors() {
#if !SANITIZER_GO
    cur_thread()->ignore_interceptors--;
#endif
  }
};

const char *GetObjectTypeFromTag(uptr tag);
const char *GetReportHeaderFromTag(uptr tag);
uptr TagFromShadowStackFrame(uptr pc);

class ScopedReportBase {
 public:
  void AddMemoryAccess(uptr addr, uptr external_tag, Shadow s, StackTrace stack,
                       const MutexSet *mset);
  void AddStack(StackTrace stack, bool suppressable = false);
  void AddThread(const ThreadContext *tctx, bool suppressable = false);
  void AddThread(Tid unique_tid, bool suppressable = false);
  void AddUniqueTid(Tid unique_tid);
  void AddMutex(const SyncVar *s);
  u64 AddMutex(u64 id);
  void AddLocation(uptr addr, uptr size);
  void AddSleep(StackID stack_id);
  void SetCount(int count);

  const ReportDesc *GetReport() const;

 protected:
  ScopedReportBase(ReportType typ, uptr tag);
  ~ScopedReportBase();

 private:
  ReportDesc *rep_;
  // Symbolizer makes lots of intercepted calls. If we try to process them,
  // at best it will cause deadlocks on internal mutexes.
  ScopedIgnoreInterceptors ignore_interceptors_;

  void AddDeadMutex(u64 id);

  ScopedReportBase(const ScopedReportBase &) = delete;
  void operator=(const ScopedReportBase &) = delete;
};

class ScopedReport : public ScopedReportBase {
 public:
  explicit ScopedReport(ReportType typ, uptr tag = kExternalTagNone);
  ~ScopedReport();

 private:
  ScopedErrorReportLock lock_;
};

bool ShouldReport(ThreadState *thr, ReportType typ);
ThreadContext *IsThreadStackOrTls(uptr addr, bool *is_stack);
void RestoreStack(Tid tid, const u64 epoch, VarSizeStackTrace *stk,
                  MutexSet *mset, uptr *tag = nullptr);

// The stack could look like:
//   <start> | <main> | <foo> | tag | <bar>
// This will extract the tag and keep:
//   <start> | <main> | <foo> | <bar>
template<typename StackTraceTy>
void ExtractTagFromStack(StackTraceTy *stack, uptr *tag = nullptr) {
  if (stack->size < 2) return;
  uptr possible_tag_pc = stack->trace[stack->size - 2];
  uptr possible_tag = TagFromShadowStackFrame(possible_tag_pc);
  if (possible_tag == kExternalTagNone) return;
  stack->trace_buffer[stack->size - 2] = stack->trace_buffer[stack->size - 1];
  stack->size -= 1;
  if (tag) *tag = possible_tag;
}

template<typename StackTraceTy>
void ObtainCurrentStack(ThreadState *thr, uptr toppc, StackTraceTy *stack,
                        uptr *tag = nullptr) {
  uptr size = thr->shadow_stack_pos - thr->shadow_stack;
  uptr start = 0;
  if (size + !!toppc > kStackTraceMax) {
    start = size + !!toppc - kStackTraceMax;
    size = kStackTraceMax - !!toppc;
  }
  stack->Init(&thr->shadow_stack[start], size, toppc);
  ExtractTagFromStack(stack, tag);
}

#define GET_STACK_TRACE_FATAL(thr, pc) \
  VarSizeStackTrace stack; \
  ObtainCurrentStack(thr, pc, &stack); \
  stack.ReverseOrder();

void MapShadow(uptr addr, uptr size);
void MapThreadTrace(uptr addr, uptr size, const char *name);
void DontNeedShadowFor(uptr addr, uptr size);
void UnmapShadow(ThreadState *thr, uptr addr, uptr size);
void InitializeShadowMemory();
void InitializeInterceptors();
void InitializeLibIgnore();
void InitializeDynamicAnnotations();

void ForkBefore(ThreadState *thr, uptr pc);
void ForkParentAfter(ThreadState *thr, uptr pc);
void ForkChildAfter(ThreadState *thr, uptr pc, bool start_thread);

void ReportRace(ThreadState *thr);
bool OutputReport(ThreadState *thr, const ScopedReport &srep);
bool IsFiredSuppression(Context *ctx, ReportType type, StackTrace trace);
bool IsExpectedReport(uptr addr, uptr size);

#if defined(TSAN_DEBUG_OUTPUT) && TSAN_DEBUG_OUTPUT >= 1
# define DPrintf Printf
#else
# define DPrintf(...)
#endif

#if defined(TSAN_DEBUG_OUTPUT) && TSAN_DEBUG_OUTPUT >= 2
# define DPrintf2 Printf
#else
# define DPrintf2(...)
#endif

StackID CurrentStackId(ThreadState *thr, uptr pc);
ReportStack *SymbolizeStackId(StackID stack_id);
void PrintCurrentStack(ThreadState *thr, uptr pc);
void PrintCurrentStackSlow(uptr pc);  // uses libunwind
MBlock *JavaHeapBlock(uptr addr, uptr *start);

void Initialize(ThreadState *thr);
void MaybeSpawnBackgroundThread();
int Finalize(ThreadState *thr);

void OnUserAlloc(ThreadState *thr, uptr pc, uptr p, uptr sz, bool write);
void OnUserFree(ThreadState *thr, uptr pc, uptr p, bool write);

void MemoryAccess(ThreadState *thr, uptr pc, uptr addr,
    int kAccessSizeLog, bool kAccessIsWrite, bool kIsAtomic);
void MemoryAccessImpl(ThreadState *thr, uptr addr,
    int kAccessSizeLog, bool kAccessIsWrite, bool kIsAtomic,
    u64 *shadow_mem, Shadow cur);
void MemoryAccessRange(ThreadState *thr, uptr pc, uptr addr,
    uptr size, bool is_write);
void UnalignedMemoryAccess(ThreadState *thr, uptr pc, uptr addr, uptr size,
                           AccessType typ);

const int kSizeLog1 = 0;
const int kSizeLog2 = 1;
const int kSizeLog4 = 2;
const int kSizeLog8 = 3;

ALWAYS_INLINE
void MemoryAccess(ThreadState *thr, uptr pc, uptr addr, uptr size,
                  AccessType typ) {
  int size_log;
  switch (size) {
    case 1:
      size_log = kSizeLog1;
      break;
    case 2:
      size_log = kSizeLog2;
      break;
    case 4:
      size_log = kSizeLog4;
      break;
    default:
      DCHECK_EQ(size, 8);
      size_log = kSizeLog8;
      break;
  }
  bool is_write = !(typ & kAccessRead);
  bool is_atomic = typ & kAccessAtomic;
  if (typ & kAccessVptr)
    thr->is_vptr_access = true;
  if (typ & kAccessFree)
    thr->is_freeing = true;
  MemoryAccess(thr, pc, addr, size_log, is_write, is_atomic);
  if (typ & kAccessVptr)
    thr->is_vptr_access = false;
  if (typ & kAccessFree)
    thr->is_freeing = false;
}

void MemoryResetRange(ThreadState *thr, uptr pc, uptr addr, uptr size);
void MemoryRangeFreed(ThreadState *thr, uptr pc, uptr addr, uptr size);
void MemoryRangeImitateWrite(ThreadState *thr, uptr pc, uptr addr, uptr size);
void MemoryRangeImitateWriteOrResetRange(ThreadState *thr, uptr pc, uptr addr,
                                         uptr size);

void ThreadIgnoreBegin(ThreadState *thr, uptr pc);
void ThreadIgnoreEnd(ThreadState *thr);
void ThreadIgnoreSyncBegin(ThreadState *thr, uptr pc);
void ThreadIgnoreSyncEnd(ThreadState *thr);

void FuncEntry(ThreadState *thr, uptr pc);
void FuncExit(ThreadState *thr);

Tid ThreadCreate(ThreadState *thr, uptr pc, uptr uid, bool detached);
void ThreadStart(ThreadState *thr, Tid tid, tid_t os_id,
                 ThreadType thread_type);
void ThreadFinish(ThreadState *thr);
Tid ThreadConsumeTid(ThreadState *thr, uptr pc, uptr uid);
void ThreadJoin(ThreadState *thr, uptr pc, Tid tid);
void ThreadDetach(ThreadState *thr, uptr pc, Tid tid);
void ThreadFinalize(ThreadState *thr);
void ThreadSetName(ThreadState *thr, const char *name);
int ThreadCount(ThreadState *thr);
void ProcessPendingSignalsImpl(ThreadState *thr);
void ThreadNotJoined(ThreadState *thr, uptr pc, Tid tid, uptr uid);

Processor *ProcCreate();
void ProcDestroy(Processor *proc);
void ProcWire(Processor *proc, ThreadState *thr);
void ProcUnwire(Processor *proc, ThreadState *thr);

// Note: the parameter is called flagz, because flags is already taken
// by the global function that returns flags.
void MutexCreate(ThreadState *thr, uptr pc, uptr addr, u32 flagz = 0);
void MutexDestroy(ThreadState *thr, uptr pc, uptr addr, u32 flagz = 0);
void MutexPreLock(ThreadState *thr, uptr pc, uptr addr, u32 flagz = 0);
void MutexPostLock(ThreadState *thr, uptr pc, uptr addr, u32 flagz = 0,
    int rec = 1);
int  MutexUnlock(ThreadState *thr, uptr pc, uptr addr, u32 flagz = 0);
void MutexPreReadLock(ThreadState *thr, uptr pc, uptr addr, u32 flagz = 0);
void MutexPostReadLock(ThreadState *thr, uptr pc, uptr addr, u32 flagz = 0);
void MutexReadUnlock(ThreadState *thr, uptr pc, uptr addr);
void MutexReadOrWriteUnlock(ThreadState *thr, uptr pc, uptr addr);
void MutexRepair(ThreadState *thr, uptr pc, uptr addr);  // call on EOWNERDEAD
void MutexInvalidAccess(ThreadState *thr, uptr pc, uptr addr);

void Acquire(ThreadState *thr, uptr pc, uptr addr);
// AcquireGlobal synchronizes the current thread with all other threads.
// In terms of happens-before relation, it draws a HB edge from all threads
// (where they happen to execute right now) to the current thread. We use it to
// handle Go finalizers. Namely, finalizer goroutine executes AcquireGlobal
// right before executing finalizers. This provides a coarse, but simple
// approximation of the actual required synchronization.
void AcquireGlobal(ThreadState *thr);
void Release(ThreadState *thr, uptr pc, uptr addr);
void ReleaseStoreAcquire(ThreadState *thr, uptr pc, uptr addr);
void ReleaseStore(ThreadState *thr, uptr pc, uptr addr);
void AfterSleep(ThreadState *thr, uptr pc);
void AcquireImpl(ThreadState *thr, uptr pc, SyncClock *c);
void ReleaseImpl(ThreadState *thr, uptr pc, SyncClock *c);
void ReleaseStoreAcquireImpl(ThreadState *thr, uptr pc, SyncClock *c);
void ReleaseStoreImpl(ThreadState *thr, uptr pc, SyncClock *c);
void AcquireReleaseImpl(ThreadState *thr, uptr pc, SyncClock *c);

// The hacky call uses custom calling convention and an assembly thunk.
// It is considerably faster that a normal call for the caller
// if it is not executed (it is intended for slow paths from hot functions).
// The trick is that the call preserves all registers and the compiler
// does not treat it as a call.
// If it does not work for you, use normal call.
#if !SANITIZER_DEBUG && defined(__x86_64__) && !SANITIZER_APPLE
// The caller may not create the stack frame for itself at all,
// so we create a reserve stack frame for it (1024b must be enough).
#define HACKY_CALL(f) \
  __asm__ __volatile__("sub $1024, %%rsp;" \
                       CFI_INL_ADJUST_CFA_OFFSET(1024) \
                       ".hidden " #f "_thunk;" \
                       "call " #f "_thunk;" \
                       "add $1024, %%rsp;" \
                       CFI_INL_ADJUST_CFA_OFFSET(-1024) \
                       ::: "memory", "cc");
#else
#define HACKY_CALL(f) f()
#endif

void TraceSwitch(ThreadState *thr);
uptr TraceTopPC(ThreadState *thr);
uptr TraceSize();
uptr TraceParts();
Trace *ThreadTrace(Tid tid);

extern "C" void __tsan_trace_switch();
void ALWAYS_INLINE TraceAddEvent(ThreadState *thr, FastState fs,
                                        EventType typ, u64 addr) {
  if (!kCollectHistory)
    return;
  // TraceSwitch accesses shadow_stack, but it's called infrequently,
  // so we check it here proactively.
  DCHECK(thr->shadow_stack);
  DCHECK_GE((int)typ, 0);
  DCHECK_LE((int)typ, 7);
  DCHECK_EQ(GetLsb(addr, kEventPCBits), addr);
  u64 pos = fs.GetTracePos();
  if (UNLIKELY((pos % kTracePartSize) == 0)) {
#if !SANITIZER_GO
    HACKY_CALL(__tsan_trace_switch);
#else
    TraceSwitch(thr);
#endif
  }
  Event *trace = (Event*)GetThreadTrace(fs.tid());
  Event *evp = &trace[pos];
  Event ev = (u64)addr | ((u64)typ << kEventPCBits);
  *evp = ev;
}

#if !SANITIZER_GO
uptr ALWAYS_INLINE HeapEnd() {
  return HeapMemEnd() + PrimaryAllocator::AdditionalSize();
}
#endif

ThreadState *FiberCreate(ThreadState *thr, uptr pc, unsigned flags);
void FiberDestroy(ThreadState *thr, uptr pc, ThreadState *fiber);
void FiberSwitch(ThreadState *thr, uptr pc, ThreadState *fiber, unsigned flags);

// These need to match __tsan_switch_to_fiber_* flags defined in
// tsan_interface.h. See documentation there as well.
enum FiberSwitchFlags {
  FiberSwitchFlagNoSync = 1 << 0, // __tsan_switch_to_fiber_no_sync
};

ALWAYS_INLINE void ProcessPendingSignals(ThreadState *thr) {
  if (UNLIKELY(atomic_load_relaxed(&thr->pending_signals)))
    ProcessPendingSignalsImpl(thr);
}

extern bool is_initialized;

ALWAYS_INLINE
void LazyInitialize(ThreadState *thr) {
  // If we can use .preinit_array, assume that __tsan_init
  // called from .preinit_array initializes runtime before
  // any instrumented code.
#if !SANITIZER_CAN_USE_PREINIT_ARRAY
  if (UNLIKELY(!is_initialized))
    Initialize(thr);
#endif
}

namespace v3 {

void TraceSwitchPart(ThreadState *thr);
bool RestoreStack(Tid tid, EventType type, Sid sid, Epoch epoch, uptr addr,
                  uptr size, AccessType typ, VarSizeStackTrace *pstk,
                  MutexSet *pmset, uptr *ptag);

template <typename EventT>
ALWAYS_INLINE WARN_UNUSED_RESULT bool TraceAcquire(ThreadState *thr,
                                                   EventT **ev) {
  Event *pos = reinterpret_cast<Event *>(atomic_load_relaxed(&thr->trace_pos));
#if SANITIZER_DEBUG
  // TraceSwitch acquires these mutexes,
  // so we lock them here to detect deadlocks more reliably.
  { Lock lock(&ctx->slot_mtx); }
  { Lock lock(&thr->tctx->trace.mtx); }
  TracePart *current = thr->tctx->trace.parts.Back();
  if (current) {
    DCHECK_GE(pos, &current->events[0]);
    DCHECK_LE(pos, &current->events[TracePart::kSize]);
  } else {
    DCHECK_EQ(pos, nullptr);
  }
#endif
  // TracePart is allocated with mmap and is at least 4K aligned.
  // So the following check is a faster way to check for part end.
  // It may have false positives in the middle of the trace,
  // they are filtered out in TraceSwitch.
  if (UNLIKELY(((uptr)(pos + 1) & TracePart::kAlignment) == 0))
    return false;
  *ev = reinterpret_cast<EventT *>(pos);
  return true;
}

template <typename EventT>
ALWAYS_INLINE void TraceRelease(ThreadState *thr, EventT *evp) {
  DCHECK_LE(evp + 1, &thr->tctx->trace.parts.Back()->events[TracePart::kSize]);
  atomic_store_relaxed(&thr->trace_pos, (uptr)(evp + 1));
}

template <typename EventT>
void TraceEvent(ThreadState *thr, EventT ev) {
  EventT *evp;
  if (!TraceAcquire(thr, &evp)) {
    TraceSwitchPart(thr);
    UNUSED bool res = TraceAcquire(thr, &evp);
    DCHECK(res);
  }
  *evp = ev;
  TraceRelease(thr, evp);
}

ALWAYS_INLINE WARN_UNUSED_RESULT bool TryTraceFunc(ThreadState *thr,
                                                   uptr pc = 0) {
  if (!kCollectHistory)
    return true;
  EventFunc *ev;
  if (UNLIKELY(!TraceAcquire(thr, &ev)))
    return false;
  ev->is_access = 0;
  ev->is_func = 1;
  ev->pc = pc;
  TraceRelease(thr, ev);
  return true;
}

WARN_UNUSED_RESULT
bool TryTraceMemoryAccess(ThreadState *thr, uptr pc, uptr addr, uptr size,
                          AccessType typ);
WARN_UNUSED_RESULT
bool TryTraceMemoryAccessRange(ThreadState *thr, uptr pc, uptr addr, uptr size,
                               AccessType typ);
void TraceMemoryAccessRange(ThreadState *thr, uptr pc, uptr addr, uptr size,
                            AccessType typ);
void TraceFunc(ThreadState *thr, uptr pc = 0);
void TraceMutexLock(ThreadState *thr, EventType type, uptr pc, uptr addr,
                    StackID stk);
void TraceMutexUnlock(ThreadState *thr, uptr addr);
void TraceTime(ThreadState *thr);

}  // namespace v3

void GrowShadowStack(ThreadState *thr);

ALWAYS_INLINE
void FuncEntry(ThreadState *thr, uptr pc) {
  DPrintf2("#%d: FuncEntry %p\n", (int)thr->fast_state.tid(), (void *)pc);
  if (kCollectHistory) {
    thr->fast_state.IncrementEpoch();
    TraceAddEvent(thr, thr->fast_state, EventTypeFuncEnter, pc);
  }

  // Shadow stack maintenance can be replaced with
  // stack unwinding during trace switch (which presumably must be faster).
  DCHECK_GE(thr->shadow_stack_pos, thr->shadow_stack);
#if !SANITIZER_GO
  DCHECK_LT(thr->shadow_stack_pos, thr->shadow_stack_end);
#else
  if (thr->shadow_stack_pos == thr->shadow_stack_end)
    GrowShadowStack(thr);
#endif
  thr->shadow_stack_pos[0] = pc;
  thr->shadow_stack_pos++;
}

ALWAYS_INLINE
void FuncExit(ThreadState *thr) {
  DPrintf2("#%d: FuncExit\n", (int)thr->fast_state.tid());
  if (kCollectHistory) {
    thr->fast_state.IncrementEpoch();
    TraceAddEvent(thr, thr->fast_state, EventTypeFuncExit, 0);
  }

  DCHECK_GT(thr->shadow_stack_pos, thr->shadow_stack);
#if !SANITIZER_GO
  DCHECK_LT(thr->shadow_stack_pos, thr->shadow_stack_end);
#endif
  thr->shadow_stack_pos--;
}

#if !SANITIZER_GO
extern void (*on_initialize)(void);
extern int (*on_finalize)(int);
#endif

}  // namespace __tsan

#endif  // TSAN_RTL_H
