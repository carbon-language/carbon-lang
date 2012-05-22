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

#include "tsan_clock.h"
#include "tsan_defs.h"
#include "tsan_flags.h"
#include "tsan_sync.h"
#include "tsan_trace.h"
#include "tsan_vector.h"
#include "tsan_report.h"

namespace __tsan {

void Printf(const char *format, ...) FORMAT(1, 2);
uptr Snprintf(char *buffer, uptr length, const char *format, ...)  FORMAT(3, 4);

inline void NOINLINE breakhere() {
  volatile int x = 42;
  (void)x;
}

// FastState (from most significant bit):
//   unused          : 1
//   tid             : kTidBits
//   epoch           : kClkBits
//   unused          : -
//   ignore_bit      : 1
class FastState {
 public:
  FastState(u64 tid, u64 epoch) {
    x_ = tid << kTidShift;
    x_ |= epoch << kClkShift;
    DCHECK(tid == this->tid());
    DCHECK(epoch == this->epoch());
  }

  explicit FastState(u64 x)
      : x_(x) {
  }

  u64 tid() const {
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
  bool GetIgnoreBit() const { return x_ & kIgnoreBit; }

 private:
  friend class Shadow;
  static const int kTidShift = 64 - kTidBits - 1;
  static const int kClkShift = kTidShift - kClkBits;
  static const u64 kIgnoreBit = 1ull;
  static const u64 kFreedBit = 1ull << 63;
  u64 x_;
};

// Shadow (from most significant bit):
//   freed           : 1
//   tid             : kTidBits
//   epoch           : kClkBits
//   is_write        : 1
//   size_log        : 2
//   addr0           : 3
class Shadow: public FastState {
 public:
  explicit Shadow(u64 x) : FastState(x) { }

  explicit Shadow(const FastState &s) : FastState(s.x_) { }

  void SetAddr0AndSizeLog(u64 addr0, unsigned kAccessSizeLog) {
    DCHECK_EQ(x_ & 31, 0);
    DCHECK_LE(addr0, 7);
    DCHECK_LE(kAccessSizeLog, 3);
    x_ |= (kAccessSizeLog << 3) | addr0;
    DCHECK_EQ(kAccessSizeLog, size_log());
    DCHECK_EQ(addr0, this->addr0());
  }

  void SetWrite(unsigned kAccessIsWrite) {
    DCHECK_EQ(x_ & 32, 0);
    if (kAccessIsWrite)
      x_ |= 32;
    DCHECK_EQ(kAccessIsWrite, is_write());
  }

  bool IsZero() const { return x_ == 0; }
  u64 raw() const { return x_; }

  static inline bool TidsAreEqual(const Shadow s1, const Shadow s2) {
    u64 shifted_xor = (s1.x_ ^ s2.x_) >> kTidShift;
    DCHECK_EQ(shifted_xor == 0, s1.tid() == s2.tid());
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
  bool is_write() const { return x_ & 32; }

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

  bool GetFreedAndReset() {
    bool res = x_ & kFreedBit;
    x_ &= ~kFreedBit;
    return res;
  }

 private:
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

// Freed memory.
// As if 8-byte write by thread 0xff..f at epoch 0xff..f, races with everything.
const u64 kShadowFreed = 0xfffffffffffffff8ull;

const int kSigCount = 128;
const int kShadowStackSize = 1024;

struct my_siginfo_t {
  int opaque[128];
};

struct SignalDesc {
  bool armed;
  bool sigaction;
  my_siginfo_t siginfo;
};

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
  uptr shadow_stack[kShadowStackSize];
  ThreadClock clock;
  u64 stat[StatCnt];
  const int tid;
  int in_rtl;
  const uptr stk_addr;
  const uptr stk_size;
  const uptr tls_addr;
  const uptr tls_size;

  DeadlockDetector deadlock_detector;

  bool in_signal_handler;
  int int_signal_send;
  int pending_signal_count;
  SignalDesc pending_signals[kSigCount];

  explicit ThreadState(Context *ctx, int tid, u64 epoch,
                       uptr stk_addr, uptr stk_size,
                       uptr tls_addr, uptr tls_size);
};

Context *CTX();
extern THREADLOCAL char cur_thread_placeholder[];

INLINE ThreadState *cur_thread() {
  return reinterpret_cast<ThreadState *>(&cur_thread_placeholder);
}

enum ThreadStatus {
  ThreadStatusInvalid,   // Non-existent thread, data is invalid.
  ThreadStatusCreated,   // Created but not yet running.
  ThreadStatusRunning,   // The thread is currently running.
  ThreadStatusFinished,  // Joinable thread is finished but not yet joined.
  ThreadStatusDead,      // Joined, but some info (trace) is still alive.
};

// An info about a thread that is hold for some time after its termination.
struct ThreadDeadInfo {
  Trace trace;
};

struct ThreadContext {
  const int tid;
  int unique_id;  // Non-rolling thread id.
  uptr user_id;  // Some opaque user thread id (e.g. pthread_t).
  ThreadState *thr;
  ThreadStatus status;
  bool detached;
  int reuse_count;
  SyncClock sync;
  // Epoch at which the thread had started.
  // If we see an event from the thread stamped by an older epoch,
  // the event is from a dead thread that shared tid with this thread.
  u64 epoch0;
  u64 epoch1;
  StackTrace creation_stack;
  ThreadDeadInfo *dead_info;
  ThreadContext *dead_next;  // In dead thread list.

  explicit ThreadContext(int tid);
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

struct Context {
  Context();

  bool initialized;

  SyncTab synctab;

  Mutex report_mtx;
  int nreported;
  int nmissed_expected;

  Mutex thread_mtx;
  unsigned thread_seq;
  unsigned unique_thread_seq;
  int alive_threads;
  int max_alive_threads;
  ThreadContext *threads[kMaxTid];
  int dead_list_size;
  ThreadContext* dead_list_head;
  ThreadContext* dead_list_tail;

  Vector<RacyStacks> racy_stacks;
  Vector<RacyAddress> racy_addresses;

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
  void AddMemoryAccess(uptr addr, Shadow s, const StackTrace *stack);
  void AddThread(const ThreadContext *tctx);
  void AddMutex(const SyncVar *s);
  void AddLocation(uptr addr, uptr size);

  const ReportDesc *GetReport() const;

 private:
  Context *ctx_;
  ReportDesc *rep_;

  ScopedReport(const ScopedReport&);
  void operator = (const ScopedReport&);
};

void StatAggregate(u64 *dst, u64 *src);
void StatOutput(u64 *stat);
void ALWAYS_INLINE INLINE StatInc(ThreadState *thr, StatType typ, u64 n = 1) {
  if (kCollectStats)
    thr->stat[typ] += n;
}

void InitializeShadowMemory();
void InitializeInterceptors();
void InitializeDynamicAnnotations();
void Die() NORETURN;

void ReportRace(ThreadState *thr);
bool OutputReport(const ScopedReport &srep,
                  const ReportStack *suppress_stack = 0);
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

void Initialize(ThreadState *thr);
int Finalize(ThreadState *thr);

void MemoryAccess(ThreadState *thr, uptr pc, uptr addr,
    int kAccessSizeLog, bool kAccessIsWrite);
void MemoryAccessImpl(ThreadState *thr, uptr addr,
    int kAccessSizeLog, bool kAccessIsWrite, FastState fast_state,
    u64 *shadow_mem, Shadow cur);
void MemoryRead1Byte(ThreadState *thr, uptr pc, uptr addr);
void MemoryWrite1Byte(ThreadState *thr, uptr pc, uptr addr);
void MemoryRead8Byte(ThreadState *thr, uptr pc, uptr addr);
void MemoryWrite8Byte(ThreadState *thr, uptr pc, uptr addr);
void MemoryAccessRange(ThreadState *thr, uptr pc, uptr addr,
                       uptr size, bool is_write);
void MemoryResetRange(ThreadState *thr, uptr pc, uptr addr, uptr size);
void MemoryRangeFreed(ThreadState *thr, uptr pc, uptr addr, uptr size);
void IgnoreCtl(ThreadState *thr, bool write, bool begin);

void FuncEntry(ThreadState *thr, uptr pc);
void FuncExit(ThreadState *thr);

int ThreadCreate(ThreadState *thr, uptr pc, uptr uid, bool detached);
void ThreadStart(ThreadState *thr, int tid);
void ThreadFinish(ThreadState *thr);
int ThreadTid(ThreadState *thr, uptr pc, uptr uid);
void ThreadJoin(ThreadState *thr, uptr pc, int tid);
void ThreadDetach(ThreadState *thr, uptr pc, int tid);
void ThreadFinalize(ThreadState *thr);

void MutexCreate(ThreadState *thr, uptr pc, uptr addr, bool rw, bool recursive);
void MutexDestroy(ThreadState *thr, uptr pc, uptr addr);
void MutexLock(ThreadState *thr, uptr pc, uptr addr);
void MutexUnlock(ThreadState *thr, uptr pc, uptr addr);
void MutexReadLock(ThreadState *thr, uptr pc, uptr addr);
void MutexReadUnlock(ThreadState *thr, uptr pc, uptr addr);
void MutexReadOrWriteUnlock(ThreadState *thr, uptr pc, uptr addr);

void Acquire(ThreadState *thr, uptr pc, uptr addr);
void Release(ThreadState *thr, uptr pc, uptr addr);

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
  __asm__ __volatile__("sub $0x400, %%rsp;" \
                       "call " #f "_thunk;" \
                       "add $0x400, %%rsp;" ::: "memory");
#else
#define HACKY_CALL(f) f()
#endif

extern "C" void __tsan_trace_switch();
void ALWAYS_INLINE INLINE TraceAddEvent(ThreadState *thr, u64 epoch,
                                        EventType typ, uptr addr) {
  StatInc(thr, StatEvents);
  if (UNLIKELY((epoch % kTracePartSize) == 0))
    HACKY_CALL(__tsan_trace_switch);
  Event *evp = &thr->trace.events[epoch % kTraceSize];
  Event ev = (u64)addr | ((u64)typ << 61);
  *evp = ev;
}

}  // namespace __tsan

#endif  // TSAN_RTL_H
