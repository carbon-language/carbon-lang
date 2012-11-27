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
#include "tsan_interface_atomic.h"
#include "tsan_flags.h"
#include "tsan_rtl.h"

using namespace __tsan;  // NOLINT

class ScopedAtomic {
 public:
  ScopedAtomic(ThreadState *thr, uptr pc, const char *func)
      : thr_(thr) {
    CHECK_EQ(thr_->in_rtl, 1);  // 1 due to our own ScopedInRtl member.
    DPrintf("#%d: %s\n", thr_->tid, func);
  }
  ~ScopedAtomic() {
    CHECK_EQ(thr_->in_rtl, 1);
  }
 private:
  ThreadState *thr_;
  ScopedInRtl in_rtl_;
};

// Some shortcuts.
typedef __tsan_memory_order morder;
typedef __tsan_atomic8 a8;
typedef __tsan_atomic16 a16;
typedef __tsan_atomic32 a32;
typedef __tsan_atomic64 a64;
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
             :             StatAtomic8);
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

template<typename T> T func_xchg(T v, T op) {
  return op;
}

template<typename T> T func_add(T v, T op) {
  return v + op;
}

template<typename T> T func_sub(T v, T op) {
  return v - op;
}

template<typename T> T func_and(T v, T op) {
  return v & op;
}

template<typename T> T func_or(T v, T op) {
  return v | op;
}

template<typename T> T func_xor(T v, T op) {
  return v ^ op;
}

template<typename T> T func_nand(T v, T op) {
  return ~v & op;
}

#define SCOPED_ATOMIC(func, ...) \
    mo = ConvertOrder(mo); \
    mo = flags()->force_seq_cst_atomics ? (morder)mo_seq_cst : mo; \
    ThreadState *const thr = cur_thread(); \
    ProcessPendingSignals(thr); \
    const uptr pc = (uptr)__builtin_return_address(0); \
    AtomicStatInc(thr, sizeof(*a), mo, StatAtomic##func); \
    ScopedAtomic sa(thr, pc, __FUNCTION__); \
    return Atomic##func(thr, pc, __VA_ARGS__); \
/**/

template<typename T>
static T AtomicLoad(ThreadState *thr, uptr pc, const volatile T *a,
    morder mo) {
  CHECK(IsLoadOrder(mo));
  // This fast-path is critical for performance.
  // Assume the access is atomic.
  if (!IsAcquireOrder(mo) && sizeof(T) <= sizeof(a))
    return *a;
  SyncVar *s = CTX()->synctab.GetAndLock(thr, pc, (uptr)a, false);
  thr->clock.set(thr->tid, thr->fast_state.epoch());
  thr->clock.acquire(&s->clock);
  T v = *a;
  s->mtx.ReadUnlock();
  return v;
}

template<typename T>
static void AtomicStore(ThreadState *thr, uptr pc, volatile T *a, T v,
    morder mo) {
  CHECK(IsStoreOrder(mo));
  // This fast-path is critical for performance.
  // Assume the access is atomic.
  // Strictly saying even relaxed store cuts off release sequence,
  // so must reset the clock.
  if (!IsReleaseOrder(mo) && sizeof(T) <= sizeof(a)) {
    *a = v;
    return;
  }
  SyncVar *s = CTX()->synctab.GetAndLock(thr, pc, (uptr)a, true);
  thr->clock.set(thr->tid, thr->fast_state.epoch());
  thr->clock.ReleaseStore(&s->clock);
  *a = v;
  s->mtx.Unlock();
}

template<typename T, T (*F)(T v, T op)>
static T AtomicRMW(ThreadState *thr, uptr pc, volatile T *a, T v, morder mo) {
  SyncVar *s = CTX()->synctab.GetAndLock(thr, pc, (uptr)a, true);
  thr->clock.set(thr->tid, thr->fast_state.epoch());
  if (IsAcqRelOrder(mo))
    thr->clock.acq_rel(&s->clock);
  else if (IsReleaseOrder(mo))
    thr->clock.release(&s->clock);
  else if (IsAcquireOrder(mo))
    thr->clock.acquire(&s->clock);
  T c = *a;
  *a = F(c, v);
  s->mtx.Unlock();
  return c;
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
  SyncVar *s = CTX()->synctab.GetAndLock(thr, pc, (uptr)a, true);
  thr->clock.set(thr->tid, thr->fast_state.epoch());
  if (IsAcqRelOrder(mo))
    thr->clock.acq_rel(&s->clock);
  else if (IsReleaseOrder(mo))
    thr->clock.release(&s->clock);
  else if (IsAcquireOrder(mo))
    thr->clock.acquire(&s->clock);
  T cur = *a;
  bool res = false;
  if (cur == *c) {
    *a = v;
    res = true;
  } else {
    *c = cur;
  }
  s->mtx.Unlock();
  return res;
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

void __tsan_atomic_thread_fence(morder mo) {
  char* a;
  SCOPED_ATOMIC(Fence, mo);
}

void __tsan_atomic_signal_fence(morder mo) {
}
