//===-- tsan_rtl.h ----------------------------------------------*- C++ -*-===//
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
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_thread_registry.h"
#include "tsan_clock.h"
#include "tsan_defs.h"
#include "tsan_flags.h"
#include "tsan_sync.h"
#include "tsan_trace.h"
#include "tsan_vector.h"
#include "tsan_report.h"
#include "tsan_platform.h"
#include "tsan_mutexset.h"

#if SANITIZER_WORDSIZE != 64
# error "ThreadSanitizer is supported only on 64-bit platforms"
#endif

namespace __tsan {

// Descriptor of user's memory block.
struct MBlock {
  Mutex mtx;
  uptr size;
  u32 alloc_tid;
  u32 alloc_stack_id;
  SyncVar *head;

  MBlock()
    : mtx(MutexTypeMBlock, StatMtxMBlock) {
  }
};

#ifndef TSAN_GO
#if defined(TSAN_COMPAT_SHADOW) && TSAN_COMPAT_SHADOW
const uptr kAllocatorSpace = 0x7d0000000000ULL;
#else
const uptr kAllocatorSpace = 0x7d0000000000ULL;
#endif
const uptr kAllocatorSize  =  0x10000000000ULL;  // 1T.

struct MapUnmapCallback;
typedef SizeClassAllocator64<kAllocatorSpace, kAllocatorSize, sizeof(MBlock),
    DefaultSizeClassMap, MapUnmapCallback> PrimaryAllocator;
typedef SizeClassAllocatorLocalCache<PrimaryAllocator> AllocatorCache;
typedef LargeMmapAllocator<MapUnmapCallback> SecondaryAllocator;
typedef CombinedAllocator<PrimaryAllocator, AllocatorCache,
    SecondaryAllocator> Allocator;
Allocator *allocator();
#endif

void TsanCheckFailed(const char *file, int line, const char *cond,
                     u64 v1, u64 v2);

// FastState (from most significant bit):
//   ignore          : 1
//   tid             : kTidBits
//   epoch           : kClkBits
//   unused          : -
//   history_size    : 3
class FastState {
 public:
  FastState(u64 tid, u64 epoch) {
    x_ = tid << kTidShift;
    x_ |= epoch << kClkShift;
    DCHECK_EQ(tid, this->tid());
    DCHECK_EQ(epoch, this->epoch());
    DCHECK_EQ(GetIgnoreBit(), false);
  }

  explicit FastState(u64 x)
      : x_(x) {
  }

  u64 raw() const {
    return x_;
  }

  u64 tid() const {
    u64 res = (x_ & ~kIgnoreBit) >> kTidShift;
    return res;
  }

  u64 TidWithIgnore() const {
    u64 res = x_ >> kTidShift;
    return res;
  }

  u64 epoch() const {
    u64 res = (x_ << (kTidBits + 1)) >> (64 - kClkBits);
    return res;
  }

  void IncrementEpoch() {
    u64 old_epoch = epoch();
    x_ += 1 << kClkShift;
    DCHECK_EQ(old_epoch + 1, epoch());
    (void)old_epoch;
  }

  void SetIgnoreBit() { x_ |= kIgnoreBit; }
  void ClearIgnoreBit() { x_ &= ~kIgnoreBit; }
  bool GetIgnoreBit() const { return (s64)x_ < 0; }

  void SetHistorySize(int hs) {
    CHECK_GE(hs, 0);
    CHECK_LE(hs, 7);
    x_ = (x_ & ~7) | hs;
  }

  int GetHistorySize() const {
    return (int)(x_ & 7);
  }

  void ClearHistorySize() {
    x_ &= ~7;
  }

  u64 GetTracePos() const {
    const int hs = GetHistorySize();
    // When hs == 0, the trace consists of 2 parts.
    const u64 mask = (1ull << (kTracePartSizeBits + hs + 1)) - 1;
    return epoch() & mask;
  }

 private:
  friend class Shadow;
  static const int kTidShift = 64 - kTidBits - 1;
  static const int kClkShift = kTidShift - kClkBits;
  static const u64 kIgnoreBit = 1ull << 63;
  static const u64 kFreedBit = 1ull << 63;
  u64 x_;
};

// Shadow (from most significant bit):
//   freed           : 1
//   tid             : kTidBits
//   epoch           : kClkBits
//   is_atomic       : 1
//   is_read         : 1
//   size_log        : 2
//   addr0           : 3
class Shadow : public FastState {
 public:
  explicit Shadow(u64 x)
      : FastState(x) {
  }

  explicit Shadow(const FastState &s)
      : FastState(s.x_) {
    ClearHistorySize();
  }

  void SetAddr0AndSizeLog(u64 addr0, unsigned kAccessSizeLog) {
    DCHECK_EQ(x_ & 31, 0);
    DCHECK_LE(addr0, 7);
    DCHECK_LE(kAccessSizeLog, 3);
    x_ |= (kAccessSizeLog << 3) | addr0;
    DCHECK_EQ(kAccessSizeLog, size_log());
    DCHECK_EQ(addr0, this->addr0());
  }

  void SetWrite(unsigned kAccessIsWrite) {
    DCHECK_EQ(x_ & kReadBit, 0);
    if (!kAccessIsWrite)
      x_ |= kReadBit;
    DCHECK_EQ(kAccessIsWrite, IsWrite());
  }

  void SetAtomic(bool kIsAtomic) {
    DCHECK(!IsAtomic());
    if (kIsAtomic)
      x_ |= kAtomicBit;
    DCHECK_EQ(IsAtomic(), kIsAtomic);
  }

  bool IsAtomic() const {
    return x_ & kAtomicBit;
  }

  bool IsZero() const {
    return x_ == 0;
  }

  static inline bool TidsAreEqual(const Shadow s1, const Shadow s2) {
    u64 shifted_xor = (s1.x_ ^ s2.x_) >> kTidShift;
    DCHECK_EQ(shifted_xor == 0, s1.TidWithIgnore() == s2.TidWithIgnore());
    return shifted_xor == 0;
  }

  static inline bool Addr0AndSizeAreEqual(const Shadow s1, const Shadow s2) {
    u64 masked_xor = (s1.x_ ^ s2.x_) & 31;
    return masked_xor == 0;
  }

  static inline bool TwoRangesIntersect(Shadow s1, Shadow s2,
      unsigned kS2AccessSize) {
    bool res = false;
    u64 diff = s1.addr0() - s2.addr0();
    if ((s64)diff < 0) {  // s1.addr0 < s2.addr0  // NOLINT
      // if (s1.addr0() + size1) > s2.addr0()) return true;
      if (s1.size() > -diff)  res = true;
    } else {
      // if (s2.addr0() + kS2AccessSize > s1.addr0()) return true;
      if (kS2AccessSize > diff) res = true;
    }
    DCHECK_EQ(res, TwoRangesIntersectSLOW(s1, s2));
    DCHECK_EQ(res, TwoRangesIntersectSLOW(s2, s1));
    return res;
  }

  // The idea behind the offset is as follows.
  // Consider that we have 8 bool's contained within a single 8-byte block
  // (mapped to a single shadow "cell"). Now consider that we write to the bools
  // from a single thread (which we consider the common case).
  // W/o offsetting each access will have to scan 4 shadow values at average
  // to find the corresponding shadow value for the bool.
  // With offsetting we start scanning shadow with the offset so that
  // each access hits necessary shadow straight off (at least in an expected
  // optimistic case).
  // This logic works seamlessly for any layout of user data. For example,
  // if user data is {int, short, char, char}, then accesses to the int are
  // offsetted to 0, short - 4, 1st char - 6, 2nd char - 7. Hopefully, accesses
  // from a single thread won't need to scan all 8 shadow values.
  unsigned ComputeSearchOffset() {
    return x_ & 7;
  }
  u64 addr0() const { return x_ & 7; }
  u64 size() const { return 1ull << size_log(); }
  bool IsWrite() const { return !IsRead(); }
  bool IsRead() const { return x_ & kReadBit; }

  // The idea behind the freed bit is as follows.
  // When the memory is freed (or otherwise unaccessible) we write to the shadow
  // values with tid/epoch related to the free and the freed bit set.
  // During memory accesses processing the freed bit is considered
  // as msb of tid. So any access races with shadow with freed bit set
  // (it is as if write from a thread with which we never synchronized before).
  // This allows us to detect accesses to freed memory w/o additional
  // overheads in memory access processing and at the same time restore
  // tid/epoch of free.
  void MarkAsFreed() {
     x_ |= kFreedBit;
  }

  bool IsFreed() const {
    return x_ & kFreedBit;
  }

  bool GetFreedAndReset() {
    bool res = x_ & kFreedBit;
    x_ &= ~kFreedBit;
    return res;
  }

  bool IsBothReadsOrAtomic(bool kIsWrite, bool kIsAtomic) const {
    // analyzes 5-th bit (is_read) and 6-th bit (is_atomic)
    bool v = x_ & u64(((kIsWrite ^ 1) << kReadShift)
        | (kIsAtomic << kAtomicShift));
    DCHECK_EQ(v, (!IsWrite() && !kIsWrite) || (IsAtomic() && kIsAtomic));
    return v;
  }

  bool IsRWNotWeaker(bool kIsWrite, bool kIsAtomic) const {
    bool v = ((x_ >> kReadShift) & 3)
        <= u64((kIsWrite ^ 1) | (kIsAtomic << 1));
    DCHECK_EQ(v, (IsAtomic() < kIsAtomic) ||
        (IsAtomic() == kIsAtomic && !IsWrite() <= !kIsWrite));
    return v;
  }

  bool IsRWWeakerOrEqual(bool kIsWrite, bool kIsAtomic) const {
    bool v = ((x_ >> kReadShift) & 3)
        >= u64((kIsWrite ^ 1) | (kIsAtomic << 1));
    DCHECK_EQ(v, (IsAtomic() > kIsAtomic) ||
        (IsAtomic() == kIsAtomic && !IsWrite() >= !kIsWrite));
    return v;
  }

 private:
  static const u64 kReadShift   = 5;
  static const u64 kReadBit     = 1ull << kReadShift;
  static const u64 kAtomicShift = 6;
  static const u64 kAtomicBit   = 1ull << kAtomicShift;

  u64 size_log() const { return (x_ >> 3) & 3; }

  static bool TwoRangesIntersectSLOW(const Shadow s1, const Shadow s2) {
    if (s1.addr0() == s2.addr0()) return true;
    if (s1.addr0() < s2.addr0() && s1.addr0() + s1.size() > s2.addr0())
      return true;
    if (s2.addr0() < s1.addr0() && s2.addr0() + s2.size() > s1.addr0())
      return true;
    return false;
  }
};

struct SignalContext;

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
  // This is a slow path flag. On fast path, fast_state.GetIgnoreBit() is read.
  // We do not distinguish beteween ignoring reads and writes
  // for better performance.
  int ignore_reads_and_writes;
  uptr *shadow_stack_pos;
  u64 *racy_shadow_addr;
  u64 racy_state[2];
  Trace trace;
#ifndef TSAN_GO
  // C/C++ uses embed shadow stack of fixed size.
  uptr shadow_stack[kShadowStackSize];
#else
  // Go uses satellite shadow stack with dynamic size.
  uptr *shadow_stack;
  uptr *shadow_stack_end;
#endif
  MutexSet mset;
  ThreadClock clock;
#ifndef TSAN_GO
  AllocatorCache alloc_cache;
#endif
  u64 stat[StatCnt];
  const int tid;
  const int unique_id;
  int in_rtl;
  bool in_symbolizer;
  bool is_alive;
  bool is_freeing;
  const uptr stk_addr;
  const uptr stk_size;
  const uptr tls_addr;
  const uptr tls_size;

  DeadlockDetector deadlock_detector;

  bool in_signal_handler;
  SignalContext *signal_ctx;

#ifndef TSAN_GO
  u32 last_sleep_stack_id;
  ThreadClock last_sleep_clock;
#endif

  // Set in regions of runtime that must be signal-safe and fork-safe.
  // If set, malloc must not be called.
  int nomalloc;

  explicit ThreadState(Context *ctx, int tid, int unique_id, u64 epoch,
                       uptr stk_addr, uptr stk_size,
                       uptr tls_addr, uptr tls_size);
};

Context *CTX();

#ifndef TSAN_GO
extern THREADLOCAL char cur_thread_placeholder[];
INLINE ThreadState *cur_thread() {
  return reinterpret_cast<ThreadState *>(&cur_thread_placeholder);
}
#endif

// An info about a thread that is hold for some time after its termination.
struct ThreadDeadInfo {
  Trace trace;
};

class ThreadContext : public ThreadContextBase {
 public:
  explicit ThreadContext(int tid);
  ~ThreadContext();
  ThreadState *thr;
#ifdef TSAN_GO
  StackTrace creation_stack;
#else
  u32 creation_stack_id;
#endif
  SyncClock sync;
  // Epoch at which the thread had started.
  // If we see an event from the thread stamped by an older epoch,
  // the event is from a dead thread that shared tid with this thread.
  u64 epoch0;
  u64 epoch1;
  ThreadDeadInfo *dead_info;

  // Override superclass callbacks.
  void OnDead();
  void OnJoined(void *arg);
  void OnFinished();
  void OnStarted(void *arg);
  void OnCreated(void *arg);
  void OnReset(void *arg);
};

struct RacyStacks {
  MD5Hash hash[2];
  bool operator==(const RacyStacks &other) const {
    if (hash[0] == other.hash[0] && hash[1] == other.hash[1])
      return true;
    if (hash[0] == other.hash[1] && hash[1] == other.hash[0])
      return true;
    return false;
  }
};

struct RacyAddress {
  uptr addr_min;
  uptr addr_max;
};

struct FiredSuppression {
  ReportType type;
  uptr pc;
};

struct Context {
  Context();

  bool initialized;

  SyncTab synctab;

  Mutex report_mtx;
  int nreported;
  int nmissed_expected;

  ThreadRegistry *thread_registry;

  Vector<RacyStacks> racy_stacks;
  Vector<RacyAddress> racy_addresses;
  Vector<FiredSuppression> fired_suppressions;

  Flags flags;

  u64 stat[StatCnt];
  u64 int_alloc_cnt[MBlockTypeCount];
  u64 int_alloc_siz[MBlockTypeCount];
};

class ScopedInRtl {
 public:
  ScopedInRtl();
  ~ScopedInRtl();
 private:
  ThreadState*thr_;
  int in_rtl_;
  int errno_;
};

class ScopedReport {
 public:
  explicit ScopedReport(ReportType typ);
  ~ScopedReport();

  void AddStack(const StackTrace *stack);
  void AddMemoryAccess(uptr addr, Shadow s, const StackTrace *stack,
                       const MutexSet *mset);
  void AddThread(const ThreadContext *tctx);
  void AddMutex(const SyncVar *s);
  void AddLocation(uptr addr, uptr size);
  void AddSleep(u32 stack_id);

  const ReportDesc *GetReport() const;

 private:
  Context *ctx_;
  ReportDesc *rep_;

  void AddMutex(u64 id);

  ScopedReport(const ScopedReport&);
  void operator = (const ScopedReport&);
};

void RestoreStack(int tid, const u64 epoch, StackTrace *stk, MutexSet *mset);

void StatAggregate(u64 *dst, u64 *src);
void StatOutput(u64 *stat);
void ALWAYS_INLINE INLINE StatInc(ThreadState *thr, StatType typ, u64 n = 1) {
  if (kCollectStats)
    thr->stat[typ] += n;
}
void ALWAYS_INLINE INLINE StatSet(ThreadState *thr, StatType typ, u64 n) {
  if (kCollectStats)
    thr->stat[typ] = n;
}

void MapShadow(uptr addr, uptr size);
void MapThreadTrace(uptr addr, uptr size);
void DontNeedShadowFor(uptr addr, uptr size);
void InitializeShadowMemory();
void InitializeInterceptors();
void InitializeDynamicAnnotations();

void ReportRace(ThreadState *thr);
bool OutputReport(Context *ctx,
                  const ScopedReport &srep,
                  const ReportStack *suppress_stack1 = 0,
                  const ReportStack *suppress_stack2 = 0);
bool IsFiredSuppression(Context *ctx,
                        const ScopedReport &srep,
                        const StackTrace &trace);
bool IsExpectedReport(uptr addr, uptr size);
bool FrameIsInternal(const ReportStack *frame);
ReportStack *SkipTsanInternalFrames(ReportStack *ent);

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

u32 CurrentStackId(ThreadState *thr, uptr pc);
void PrintCurrentStack(ThreadState *thr, uptr pc);
void PrintCurrentStackSlow();  // uses libunwind

void Initialize(ThreadState *thr);
int Finalize(ThreadState *thr);

SyncVar* GetJavaSync(ThreadState *thr, uptr pc, uptr addr,
                     bool write_lock, bool create);
SyncVar* GetAndRemoveJavaSync(ThreadState *thr, uptr pc, uptr addr);

void MemoryAccess(ThreadState *thr, uptr pc, uptr addr,
    int kAccessSizeLog, bool kAccessIsWrite, bool kIsAtomic);
void MemoryAccessImpl(ThreadState *thr, uptr addr,
    int kAccessSizeLog, bool kAccessIsWrite, bool kIsAtomic,
    u64 *shadow_mem, Shadow cur);
void MemoryAccessRange(ThreadState *thr, uptr pc, uptr addr,
    uptr size, bool is_write);
void MemoryAccessRangeStep(ThreadState *thr, uptr pc, uptr addr,
    uptr size, uptr step, bool is_write);

const int kSizeLog1 = 0;
const int kSizeLog2 = 1;
const int kSizeLog4 = 2;
const int kSizeLog8 = 3;

void ALWAYS_INLINE INLINE MemoryRead(ThreadState *thr, uptr pc,
                                     uptr addr, int kAccessSizeLog) {
  MemoryAccess(thr, pc, addr, kAccessSizeLog, false, false);
}

void ALWAYS_INLINE INLINE MemoryWrite(ThreadState *thr, uptr pc,
                                      uptr addr, int kAccessSizeLog) {
  MemoryAccess(thr, pc, addr, kAccessSizeLog, true, false);
}

void ALWAYS_INLINE INLINE MemoryReadAtomic(ThreadState *thr, uptr pc,
                                           uptr addr, int kAccessSizeLog) {
  MemoryAccess(thr, pc, addr, kAccessSizeLog, false, true);
}

void ALWAYS_INLINE INLINE MemoryWriteAtomic(ThreadState *thr, uptr pc,
                                            uptr addr, int kAccessSizeLog) {
  MemoryAccess(thr, pc, addr, kAccessSizeLog, true, true);
}

void MemoryResetRange(ThreadState *thr, uptr pc, uptr addr, uptr size);
void MemoryRangeFreed(ThreadState *thr, uptr pc, uptr addr, uptr size);
void MemoryRangeImitateWrite(ThreadState *thr, uptr pc, uptr addr, uptr size);
void IgnoreCtl(ThreadState *thr, bool write, bool begin);

void FuncEntry(ThreadState *thr, uptr pc);
void FuncExit(ThreadState *thr);

int ThreadCreate(ThreadState *thr, uptr pc, uptr uid, bool detached);
void ThreadStart(ThreadState *thr, int tid, uptr os_id);
void ThreadFinish(ThreadState *thr);
int ThreadTid(ThreadState *thr, uptr pc, uptr uid);
void ThreadJoin(ThreadState *thr, uptr pc, int tid);
void ThreadDetach(ThreadState *thr, uptr pc, int tid);
void ThreadFinalize(ThreadState *thr);
void ThreadSetName(ThreadState *thr, const char *name);
int ThreadCount(ThreadState *thr);
void ProcessPendingSignals(ThreadState *thr);

void MutexCreate(ThreadState *thr, uptr pc, uptr addr,
                 bool rw, bool recursive, bool linker_init);
void MutexDestroy(ThreadState *thr, uptr pc, uptr addr);
void MutexLock(ThreadState *thr, uptr pc, uptr addr);
void MutexUnlock(ThreadState *thr, uptr pc, uptr addr);
void MutexReadLock(ThreadState *thr, uptr pc, uptr addr);
void MutexReadUnlock(ThreadState *thr, uptr pc, uptr addr);
void MutexReadOrWriteUnlock(ThreadState *thr, uptr pc, uptr addr);

void Acquire(ThreadState *thr, uptr pc, uptr addr);
void AcquireGlobal(ThreadState *thr, uptr pc);
void Release(ThreadState *thr, uptr pc, uptr addr);
void ReleaseStore(ThreadState *thr, uptr pc, uptr addr);
void AfterSleep(ThreadState *thr, uptr pc);

// The hacky call uses custom calling convention and an assembly thunk.
// It is considerably faster that a normal call for the caller
// if it is not executed (it is intended for slow paths from hot functions).
// The trick is that the call preserves all registers and the compiler
// does not treat it as a call.
// If it does not work for you, use normal call.
#if TSAN_DEBUG == 0
// The caller may not create the stack frame for itself at all,
// so we create a reserve stack frame for it (1024b must be enough).
#define HACKY_CALL(f) \
  __asm__ __volatile__("sub $1024, %%rsp;" \
                       "/*.cfi_adjust_cfa_offset 1024;*/" \
                       ".hidden " #f "_thunk;" \
                       "call " #f "_thunk;" \
                       "add $1024, %%rsp;" \
                       "/*.cfi_adjust_cfa_offset -1024;*/" \
                       ::: "memory", "cc");
#else
#define HACKY_CALL(f) f()
#endif

void TraceSwitch(ThreadState *thr);
uptr TraceTopPC(ThreadState *thr);
uptr TraceSize();
uptr TraceParts();

extern "C" void __tsan_trace_switch();
void ALWAYS_INLINE INLINE TraceAddEvent(ThreadState *thr, FastState fs,
                                        EventType typ, u64 addr) {
  DCHECK_GE((int)typ, 0);
  DCHECK_LE((int)typ, 7);
  DCHECK_EQ(GetLsb(addr, 61), addr);
  StatInc(thr, StatEvents);
  u64 pos = fs.GetTracePos();
  if (UNLIKELY((pos % kTracePartSize) == 0)) {
#ifndef TSAN_GO
    HACKY_CALL(__tsan_trace_switch);
#else
    TraceSwitch(thr);
#endif
  }
  Event *trace = (Event*)GetThreadTrace(fs.tid());
  Event *evp = &trace[pos];
  Event ev = (u64)addr | ((u64)typ << 61);
  *evp = ev;
}

}  // namespace __tsan

#endif  // TSAN_RTL_H
