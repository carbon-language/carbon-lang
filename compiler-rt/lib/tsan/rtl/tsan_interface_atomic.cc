//===-- tsan_interface_atomic.cc ------------------------------------------===//
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
//===----------------------------------------------------------------------===//

// ThreadSanitizer atomic operations are based on C++11/C1x standards.
// For background see C++11 standard.  A slightly older, publically
// available draft of the standard (not entirely up-to-date, but close enough
// for casual browsing) is available here:
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2011/n3242.pdf
// The following page contains more background information:
// http://www.hpl.hp.com/personal/Hans_Boehm/c++mm/

#include "sanitizer_common/sanitizer_placement_new.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "tsan_interface_atomic.h"
#include "tsan_flags.h"
#include "tsan_rtl.h"

using namespace __tsan;  // NOLINT

#define SCOPED_ATOMIC(func, ...) \
    const uptr callpc = (uptr)__builtin_return_address(0); \
    uptr pc = __sanitizer::StackTrace::GetCurrentPc(); \
    pc = __sanitizer::StackTrace::GetPreviousInstructionPc(pc); \
    mo = ConvertOrder(mo); \
    mo = flags()->force_seq_cst_atomics ? (morder)mo_seq_cst : mo; \
    ThreadState *const thr = cur_thread(); \
    AtomicStatInc(thr, sizeof(*a), mo, StatAtomic##func); \
    ScopedAtomic sa(thr, callpc, __FUNCTION__); \
    return Atomic##func(thr, pc, __VA_ARGS__); \
/**/

class ScopedAtomic {
 public:
  ScopedAtomic(ThreadState *thr, uptr pc, const char *func)
      : thr_(thr) {
    CHECK_EQ(thr_->in_rtl, 0);
    ProcessPendingSignals(thr);
    FuncEntry(thr_, pc);
    DPrintf("#%d: %s\n", thr_->tid, func);
    thr_->in_rtl++;
  }
  ~ScopedAtomic() {
    thr_->in_rtl--;
    CHECK_EQ(thr_->in_rtl, 0);
    FuncExit(thr_);
  }
 private:
  ThreadState *thr_;
};

// Some shortcuts.
typedef __tsan_memory_order morder;
typedef __tsan_atomic8 a8;
typedef __tsan_atomic16 a16;
typedef __tsan_atomic32 a32;
typedef __tsan_atomic64 a64;
typedef __tsan_atomic128 a128;
const morder mo_relaxed = __tsan_memory_order_relaxed;
const morder mo_consume = __tsan_memory_order_consume;
const morder mo_acquire = __tsan_memory_order_acquire;
const morder mo_release = __tsan_memory_order_release;
const morder mo_acq_rel = __tsan_memory_order_acq_rel;
const morder mo_seq_cst = __tsan_memory_order_seq_cst;

static void AtomicStatInc(ThreadState *thr, uptr size, morder mo, StatType t) {
  StatInc(thr, StatAtomic);
  StatInc(thr, t);
  StatInc(thr, size == 1 ? StatAtomic1
             : size == 2 ? StatAtomic2
             : size == 4 ? StatAtomic4
             : size == 8 ? StatAtomic8
             :             StatAtomic16);
  StatInc(thr, mo == mo_relaxed ? StatAtomicRelaxed
             : mo == mo_consume ? StatAtomicConsume
             : mo == mo_acquire ? StatAtomicAcquire
             : mo == mo_release ? StatAtomicRelease
             : mo == mo_acq_rel ? StatAtomicAcq_Rel
             :                    StatAtomicSeq_Cst);
}

static bool IsLoadOrder(morder mo) {
  return mo == mo_relaxed || mo == mo_consume
      || mo == mo_acquire || mo == mo_seq_cst;
}

static bool IsStoreOrder(morder mo) {
  return mo == mo_relaxed || mo == mo_release || mo == mo_seq_cst;
}

static bool IsReleaseOrder(morder mo) {
  return mo == mo_release || mo == mo_acq_rel || mo == mo_seq_cst;
}

static bool IsAcquireOrder(morder mo) {
  return mo == mo_consume || mo == mo_acquire
      || mo == mo_acq_rel || mo == mo_seq_cst;
}

static bool IsAcqRelOrder(morder mo) {
  return mo == mo_acq_rel || mo == mo_seq_cst;
}

static morder ConvertOrder(morder mo) {
  if (mo > (morder)100500) {
    mo = morder(mo - 100500);
    if (mo ==  morder(1 << 0))
      mo = mo_relaxed;
    else if (mo == morder(1 << 1))
      mo = mo_consume;
    else if (mo == morder(1 << 2))
      mo = mo_acquire;
    else if (mo == morder(1 << 3))
      mo = mo_release;
    else if (mo == morder(1 << 4))
      mo = mo_acq_rel;
    else if (mo == morder(1 << 5))
      mo = mo_seq_cst;
  }
  CHECK_GE(mo, mo_relaxed);
  CHECK_LE(mo, mo_seq_cst);
  return mo;
}

template<typename T> T func_xchg(volatile T *v, T op) {
  T res = __sync_lock_test_and_set(v, op);
  // __sync_lock_test_and_set does not contain full barrier.
  __sync_synchronize();
  return res;
}

template<typename T> T func_add(volatile T *v, T op) {
  return __sync_fetch_and_add(v, op);
}

template<typename T> T func_sub(volatile T *v, T op) {
  return __sync_fetch_and_sub(v, op);
}

template<typename T> T func_and(volatile T *v, T op) {
  return __sync_fetch_and_and(v, op);
}

template<typename T> T func_or(volatile T *v, T op) {
  return __sync_fetch_and_or(v, op);
}

template<typename T> T func_xor(volatile T *v, T op) {
  return __sync_fetch_and_xor(v, op);
}

template<typename T> T func_nand(volatile T *v, T op) {
  // clang does not support __sync_fetch_and_nand.
  T cmp = *v;
  for (;;) {
    T newv = ~(cmp & op);
    T cur = __sync_val_compare_and_swap(v, cmp, newv);
    if (cmp == cur)
      return cmp;
    cmp = cur;
  }
}

template<typename T> T func_cas(volatile T *v, T cmp, T xch) {
  return __sync_val_compare_and_swap(v, cmp, xch);
}

// clang does not support 128-bit atomic ops.
// Atomic ops are executed under tsan internal mutex,
// here we assume that the atomic variables are not accessed
// from non-instrumented code.
#ifndef __GCC_HAVE_SYNC_COMPARE_AND_SWAP_16
a128 func_xchg(volatile a128 *v, a128 op) {
  a128 cmp = *v;
  *v = op;
  return cmp;
}

a128 func_add(volatile a128 *v, a128 op) {
  a128 cmp = *v;
  *v = cmp + op;
  return cmp;
}

a128 func_sub(volatile a128 *v, a128 op) {
  a128 cmp = *v;
  *v = cmp - op;
  return cmp;
}

a128 func_and(volatile a128 *v, a128 op) {
  a128 cmp = *v;
  *v = cmp & op;
  return cmp;
}

a128 func_or(volatile a128 *v, a128 op) {
  a128 cmp = *v;
  *v = cmp | op;
  return cmp;
}

a128 func_xor(volatile a128 *v, a128 op) {
  a128 cmp = *v;
  *v = cmp ^ op;
  return cmp;
}

a128 func_nand(volatile a128 *v, a128 op) {
  a128 cmp = *v;
  *v = ~(cmp & op);
  return cmp;
}

a128 func_cas(volatile a128 *v, a128 cmp, a128 xch) {
  a128 cur = *v;
  if (cur == cmp)
    *v = xch;
  return cur;
}
#endif

template<typename T>
static int SizeLog() {
  if (sizeof(T) <= 1)
    return kSizeLog1;
  else if (sizeof(T) <= 2)
    return kSizeLog2;
  else if (sizeof(T) <= 4)
    return kSizeLog4;
  else
    return kSizeLog8;
  // For 16-byte atomics we also use 8-byte memory access,
  // this leads to false negatives only in very obscure cases.
}

template<typename T>
static T AtomicLoad(ThreadState *thr, uptr pc, const volatile T *a,
    morder mo) {
  CHECK(IsLoadOrder(mo));
  // This fast-path is critical for performance.
  // Assume the access is atomic.
  if (!IsAcquireOrder(mo) && sizeof(T) <= sizeof(a)) {
    MemoryReadAtomic(thr, pc, (uptr)a, SizeLog<T>());
    return *a;
  }
  SyncVar *s = CTX()->synctab.GetOrCreateAndLock(thr, pc, (uptr)a, false);
  thr->clock.set(thr->tid, thr->fast_state.epoch());
  thr->clock.acquire(&s->clock);
  T v = *a;
  s->mtx.ReadUnlock();
  __sync_synchronize();
  MemoryReadAtomic(thr, pc, (uptr)a, SizeLog<T>());
  return v;
}

template<typename T>
static void AtomicStore(ThreadState *thr, uptr pc, volatile T *a, T v,
    morder mo) {
  CHECK(IsStoreOrder(mo));
  MemoryWriteAtomic(thr, pc, (uptr)a, SizeLog<T>());
  // This fast-path is critical for performance.
  // Assume the access is atomic.
  // Strictly saying even relaxed store cuts off release sequence,
  // so must reset the clock.
  if (!IsReleaseOrder(mo) && sizeof(T) <= sizeof(a)) {
    *a = v;
    return;
  }
  __sync_synchronize();
  SyncVar *s = CTX()->synctab.GetOrCreateAndLock(thr, pc, (uptr)a, true);
  thr->clock.set(thr->tid, thr->fast_state.epoch());
  thr->clock.ReleaseStore(&s->clock);
  *a = v;
  s->mtx.Unlock();
  // Trainling memory barrier to provide sequential consistency
  // for Dekker-like store-load synchronization.
  __sync_synchronize();
}

template<typename T, T (*F)(volatile T *v, T op)>
static T AtomicRMW(ThreadState *thr, uptr pc, volatile T *a, T v, morder mo) {
  MemoryWriteAtomic(thr, pc, (uptr)a, SizeLog<T>());
  SyncVar *s = CTX()->synctab.GetOrCreateAndLock(thr, pc, (uptr)a, true);
  thr->clock.set(thr->tid, thr->fast_state.epoch());
  if (IsAcqRelOrder(mo))
    thr->clock.acq_rel(&s->clock);
  else if (IsReleaseOrder(mo))
    thr->clock.release(&s->clock);
  else if (IsAcquireOrder(mo))
    thr->clock.acquire(&s->clock);
  v = F(a, v);
  s->mtx.Unlock();
  return v;
}

template<typename T>
static T AtomicExchange(ThreadState *thr, uptr pc, volatile T *a, T v,
    morder mo) {
  return AtomicRMW<T, func_xchg>(thr, pc, a, v, mo);
}

template<typename T>
static T AtomicFetchAdd(ThreadState *thr, uptr pc, volatile T *a, T v,
    morder mo) {
  return AtomicRMW<T, func_add>(thr, pc, a, v, mo);
}

template<typename T>
static T AtomicFetchSub(ThreadState *thr, uptr pc, volatile T *a, T v,
    morder mo) {
  return AtomicRMW<T, func_sub>(thr, pc, a, v, mo);
}

template<typename T>
static T AtomicFetchAnd(ThreadState *thr, uptr pc, volatile T *a, T v,
    morder mo) {
  return AtomicRMW<T, func_and>(thr, pc, a, v, mo);
}

template<typename T>
static T AtomicFetchOr(ThreadState *thr, uptr pc, volatile T *a, T v,
    morder mo) {
  return AtomicRMW<T, func_or>(thr, pc, a, v, mo);
}

template<typename T>
static T AtomicFetchXor(ThreadState *thr, uptr pc, volatile T *a, T v,
    morder mo) {
  return AtomicRMW<T, func_xor>(thr, pc, a, v, mo);
}

template<typename T>
static T AtomicFetchNand(ThreadState *thr, uptr pc, volatile T *a, T v,
    morder mo) {
  return AtomicRMW<T, func_nand>(thr, pc, a, v, mo);
}

template<typename T>
static bool AtomicCAS(ThreadState *thr, uptr pc,
    volatile T *a, T *c, T v, morder mo, morder fmo) {
  (void)fmo;  // Unused because llvm does not pass it yet.
  MemoryWriteAtomic(thr, pc, (uptr)a, SizeLog<T>());
  SyncVar *s = CTX()->synctab.GetOrCreateAndLock(thr, pc, (uptr)a, true);
  thr->clock.set(thr->tid, thr->fast_state.epoch());
  if (IsAcqRelOrder(mo))
    thr->clock.acq_rel(&s->clock);
  else if (IsReleaseOrder(mo))
    thr->clock.release(&s->clock);
  else if (IsAcquireOrder(mo))
    thr->clock.acquire(&s->clock);
  T cc = *c;
  T pr = func_cas(a, cc, v);
  s->mtx.Unlock();
  if (pr == cc)
    return true;
  *c = pr;
  return false;
}

template<typename T>
static T AtomicCAS(ThreadState *thr, uptr pc,
    volatile T *a, T c, T v, morder mo, morder fmo) {
  AtomicCAS(thr, pc, a, &c, v, mo, fmo);
  return c;
}

static void AtomicFence(ThreadState *thr, uptr pc, morder mo) {
  // FIXME(dvyukov): not implemented.
  __sync_synchronize();
}

a8 __tsan_atomic8_load(const volatile a8 *a, morder mo) {
  SCOPED_ATOMIC(Load, a, mo);
}

a16 __tsan_atomic16_load(const volatile a16 *a, morder mo) {
  SCOPED_ATOMIC(Load, a, mo);
}

a32 __tsan_atomic32_load(const volatile a32 *a, morder mo) {
  SCOPED_ATOMIC(Load, a, mo);
}

a64 __tsan_atomic64_load(const volatile a64 *a, morder mo) {
  SCOPED_ATOMIC(Load, a, mo);
}

#if __TSAN_HAS_INT128
a128 __tsan_atomic128_load(const volatile a128 *a, morder mo) {
  SCOPED_ATOMIC(Load, a, mo);
}
#endif

void __tsan_atomic8_store(volatile a8 *a, a8 v, morder mo) {
  SCOPED_ATOMIC(Store, a, v, mo);
}

void __tsan_atomic16_store(volatile a16 *a, a16 v, morder mo) {
  SCOPED_ATOMIC(Store, a, v, mo);
}

void __tsan_atomic32_store(volatile a32 *a, a32 v, morder mo) {
  SCOPED_ATOMIC(Store, a, v, mo);
}

void __tsan_atomic64_store(volatile a64 *a, a64 v, morder mo) {
  SCOPED_ATOMIC(Store, a, v, mo);
}

#if __TSAN_HAS_INT128
void __tsan_atomic128_store(volatile a128 *a, a128 v, morder mo) {
  SCOPED_ATOMIC(Store, a, v, mo);
}
#endif

a8 __tsan_atomic8_exchange(volatile a8 *a, a8 v, morder mo) {
  SCOPED_ATOMIC(Exchange, a, v, mo);
}

a16 __tsan_atomic16_exchange(volatile a16 *a, a16 v, morder mo) {
  SCOPED_ATOMIC(Exchange, a, v, mo);
}

a32 __tsan_atomic32_exchange(volatile a32 *a, a32 v, morder mo) {
  SCOPED_ATOMIC(Exchange, a, v, mo);
}

a64 __tsan_atomic64_exchange(volatile a64 *a, a64 v, morder mo) {
  SCOPED_ATOMIC(Exchange, a, v, mo);
}

#if __TSAN_HAS_INT128
a128 __tsan_atomic128_exchange(volatile a128 *a, a128 v, morder mo) {
  SCOPED_ATOMIC(Exchange, a, v, mo);
}
#endif

a8 __tsan_atomic8_fetch_add(volatile a8 *a, a8 v, morder mo) {
  SCOPED_ATOMIC(FetchAdd, a, v, mo);
}

a16 __tsan_atomic16_fetch_add(volatile a16 *a, a16 v, morder mo) {
  SCOPED_ATOMIC(FetchAdd, a, v, mo);
}

a32 __tsan_atomic32_fetch_add(volatile a32 *a, a32 v, morder mo) {
  SCOPED_ATOMIC(FetchAdd, a, v, mo);
}

a64 __tsan_atomic64_fetch_add(volatile a64 *a, a64 v, morder mo) {
  SCOPED_ATOMIC(FetchAdd, a, v, mo);
}

#if __TSAN_HAS_INT128
a128 __tsan_atomic128_fetch_add(volatile a128 *a, a128 v, morder mo) {
  SCOPED_ATOMIC(FetchAdd, a, v, mo);
}
#endif

a8 __tsan_atomic8_fetch_sub(volatile a8 *a, a8 v, morder mo) {
  SCOPED_ATOMIC(FetchSub, a, v, mo);
}

a16 __tsan_atomic16_fetch_sub(volatile a16 *a, a16 v, morder mo) {
  SCOPED_ATOMIC(FetchSub, a, v, mo);
}

a32 __tsan_atomic32_fetch_sub(volatile a32 *a, a32 v, morder mo) {
  SCOPED_ATOMIC(FetchSub, a, v, mo);
}

a64 __tsan_atomic64_fetch_sub(volatile a64 *a, a64 v, morder mo) {
  SCOPED_ATOMIC(FetchSub, a, v, mo);
}

#if __TSAN_HAS_INT128
a128 __tsan_atomic128_fetch_sub(volatile a128 *a, a128 v, morder mo) {
  SCOPED_ATOMIC(FetchSub, a, v, mo);
}
#endif

a8 __tsan_atomic8_fetch_and(volatile a8 *a, a8 v, morder mo) {
  SCOPED_ATOMIC(FetchAnd, a, v, mo);
}

a16 __tsan_atomic16_fetch_and(volatile a16 *a, a16 v, morder mo) {
  SCOPED_ATOMIC(FetchAnd, a, v, mo);
}

a32 __tsan_atomic32_fetch_and(volatile a32 *a, a32 v, morder mo) {
  SCOPED_ATOMIC(FetchAnd, a, v, mo);
}

a64 __tsan_atomic64_fetch_and(volatile a64 *a, a64 v, morder mo) {
  SCOPED_ATOMIC(FetchAnd, a, v, mo);
}

#if __TSAN_HAS_INT128
a128 __tsan_atomic128_fetch_and(volatile a128 *a, a128 v, morder mo) {
  SCOPED_ATOMIC(FetchAnd, a, v, mo);
}
#endif

a8 __tsan_atomic8_fetch_or(volatile a8 *a, a8 v, morder mo) {
  SCOPED_ATOMIC(FetchOr, a, v, mo);
}

a16 __tsan_atomic16_fetch_or(volatile a16 *a, a16 v, morder mo) {
  SCOPED_ATOMIC(FetchOr, a, v, mo);
}

a32 __tsan_atomic32_fetch_or(volatile a32 *a, a32 v, morder mo) {
  SCOPED_ATOMIC(FetchOr, a, v, mo);
}

a64 __tsan_atomic64_fetch_or(volatile a64 *a, a64 v, morder mo) {
  SCOPED_ATOMIC(FetchOr, a, v, mo);
}

#if __TSAN_HAS_INT128
a128 __tsan_atomic128_fetch_or(volatile a128 *a, a128 v, morder mo) {
  SCOPED_ATOMIC(FetchOr, a, v, mo);
}
#endif

a8 __tsan_atomic8_fetch_xor(volatile a8 *a, a8 v, morder mo) {
  SCOPED_ATOMIC(FetchXor, a, v, mo);
}

a16 __tsan_atomic16_fetch_xor(volatile a16 *a, a16 v, morder mo) {
  SCOPED_ATOMIC(FetchXor, a, v, mo);
}

a32 __tsan_atomic32_fetch_xor(volatile a32 *a, a32 v, morder mo) {
  SCOPED_ATOMIC(FetchXor, a, v, mo);
}

a64 __tsan_atomic64_fetch_xor(volatile a64 *a, a64 v, morder mo) {
  SCOPED_ATOMIC(FetchXor, a, v, mo);
}

#if __TSAN_HAS_INT128
a128 __tsan_atomic128_fetch_xor(volatile a128 *a, a128 v, morder mo) {
  SCOPED_ATOMIC(FetchXor, a, v, mo);
}
#endif

a8 __tsan_atomic8_fetch_nand(volatile a8 *a, a8 v, morder mo) {
  SCOPED_ATOMIC(FetchNand, a, v, mo);
}

a16 __tsan_atomic16_fetch_nand(volatile a16 *a, a16 v, morder mo) {
  SCOPED_ATOMIC(FetchNand, a, v, mo);
}

a32 __tsan_atomic32_fetch_nand(volatile a32 *a, a32 v, morder mo) {
  SCOPED_ATOMIC(FetchNand, a, v, mo);
}

a64 __tsan_atomic64_fetch_nand(volatile a64 *a, a64 v, morder mo) {
  SCOPED_ATOMIC(FetchNand, a, v, mo);
}

#if __TSAN_HAS_INT128
a128 __tsan_atomic128_fetch_nand(volatile a128 *a, a128 v, morder mo) {
  SCOPED_ATOMIC(FetchNand, a, v, mo);
}
#endif

int __tsan_atomic8_compare_exchange_strong(volatile a8 *a, a8 *c, a8 v,
    morder mo, morder fmo) {
  SCOPED_ATOMIC(CAS, a, c, v, mo, fmo);
}

int __tsan_atomic16_compare_exchange_strong(volatile a16 *a, a16 *c, a16 v,
    morder mo, morder fmo) {
  SCOPED_ATOMIC(CAS, a, c, v, mo, fmo);
}

int __tsan_atomic32_compare_exchange_strong(volatile a32 *a, a32 *c, a32 v,
    morder mo, morder fmo) {
  SCOPED_ATOMIC(CAS, a, c, v, mo, fmo);
}

int __tsan_atomic64_compare_exchange_strong(volatile a64 *a, a64 *c, a64 v,
    morder mo, morder fmo) {
  SCOPED_ATOMIC(CAS, a, c, v, mo, fmo);
}

#if __TSAN_HAS_INT128
int __tsan_atomic128_compare_exchange_strong(volatile a128 *a, a128 *c, a128 v,
    morder mo, morder fmo) {
  SCOPED_ATOMIC(CAS, a, c, v, mo, fmo);
}
#endif

int __tsan_atomic8_compare_exchange_weak(volatile a8 *a, a8 *c, a8 v,
    morder mo, morder fmo) {
  SCOPED_ATOMIC(CAS, a, c, v, mo, fmo);
}

int __tsan_atomic16_compare_exchange_weak(volatile a16 *a, a16 *c, a16 v,
    morder mo, morder fmo) {
  SCOPED_ATOMIC(CAS, a, c, v, mo, fmo);
}

int __tsan_atomic32_compare_exchange_weak(volatile a32 *a, a32 *c, a32 v,
    morder mo, morder fmo) {
  SCOPED_ATOMIC(CAS, a, c, v, mo, fmo);
}

int __tsan_atomic64_compare_exchange_weak(volatile a64 *a, a64 *c, a64 v,
    morder mo, morder fmo) {
  SCOPED_ATOMIC(CAS, a, c, v, mo, fmo);
}

#if __TSAN_HAS_INT128
int __tsan_atomic128_compare_exchange_weak(volatile a128 *a, a128 *c, a128 v,
    morder mo, morder fmo) {
  SCOPED_ATOMIC(CAS, a, c, v, mo, fmo);
}
#endif

a8 __tsan_atomic8_compare_exchange_val(volatile a8 *a, a8 c, a8 v,
    morder mo, morder fmo) {
  SCOPED_ATOMIC(CAS, a, c, v, mo, fmo);
}
a16 __tsan_atomic16_compare_exchange_val(volatile a16 *a, a16 c, a16 v,
    morder mo, morder fmo) {
  SCOPED_ATOMIC(CAS, a, c, v, mo, fmo);
}

a32 __tsan_atomic32_compare_exchange_val(volatile a32 *a, a32 c, a32 v,
    morder mo, morder fmo) {
  SCOPED_ATOMIC(CAS, a, c, v, mo, fmo);
}

a64 __tsan_atomic64_compare_exchange_val(volatile a64 *a, a64 c, a64 v,
    morder mo, morder fmo) {
  SCOPED_ATOMIC(CAS, a, c, v, mo, fmo);
}

#if __TSAN_HAS_INT128
a128 __tsan_atomic64_compare_exchange_val(volatile a128 *a, a128 c, a128 v,
    morder mo, morder fmo) {
  SCOPED_ATOMIC(CAS, a, c, v, mo, fmo);
}
#endif

void __tsan_atomic_thread_fence(morder mo) {
  char* a;
  SCOPED_ATOMIC(Fence, mo);
}

void __tsan_atomic_signal_fence(morder mo) {
}
