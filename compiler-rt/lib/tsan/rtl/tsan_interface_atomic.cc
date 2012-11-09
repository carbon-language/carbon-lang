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

#define SCOPED_ATOMIC(func, ...) \
    mo = ConvertOrder(mo); \
    mo = flags()->force_seq_cst_atomics ? (morder)mo_seq_cst : mo; \
    ThreadState *const thr = cur_thread(); \
    const uptr pc = (uptr)__builtin_return_address(0); \
    AtomicStatInc(thr, sizeof(*a), mo, StatAtomic##func); \
    ScopedAtomic sa(thr, pc, __FUNCTION__); \
    return Atomic##func(thr, pc, __VA_ARGS__); \
/**/

template<typename T>
static T AtomicLoad(ThreadState *thr, uptr pc, const volatile T *a,
    morder mo) {
  CHECK(IsLoadOrder(mo));
  T v = *a;
  if (IsAcquireOrder(mo))
    Acquire(thr, pc, (uptr)a);
  return v;
}

template<typename T>
static void AtomicStore(ThreadState *thr, uptr pc, volatile T *a, T v,
    morder mo) {
  CHECK(IsStoreOrder(mo));
  if (IsReleaseOrder(mo))
    ReleaseStore(thr, pc, (uptr)a);
  *a = v;
}

template<typename T>
static T AtomicExchange(ThreadState *thr, uptr pc, volatile T *a, T v,
    morder mo) {
  if (IsReleaseOrder(mo))
    Release(thr, pc, (uptr)a);
  v = __sync_lock_test_and_set(a, v);
  if (IsAcquireOrder(mo))
    Acquire(thr, pc, (uptr)a);
  return v;
}

template<typename T>
static T AtomicFetchAdd(ThreadState *thr, uptr pc, volatile T *a, T v,
    morder mo) {
  if (IsReleaseOrder(mo))
    Release(thr, pc, (uptr)a);
  v = __sync_fetch_and_add(a, v);
  if (IsAcquireOrder(mo))
    Acquire(thr, pc, (uptr)a);
  return v;
}

template<typename T>
static T AtomicFetchSub(ThreadState *thr, uptr pc, volatile T *a, T v,
    morder mo) {
  if (IsReleaseOrder(mo))
    Release(thr, pc, (uptr)a);
  v = __sync_fetch_and_sub(a, v);
  if (IsAcquireOrder(mo))
    Acquire(thr, pc, (uptr)a);
  return v;
}

template<typename T>
static T AtomicFetchAnd(ThreadState *thr, uptr pc, volatile T *a, T v,
    morder mo) {
  if (IsReleaseOrder(mo))
    Release(thr, pc, (uptr)a);
  v = __sync_fetch_and_and(a, v);
  if (IsAcquireOrder(mo))
    Acquire(thr, pc, (uptr)a);
  return v;
}

template<typename T>
static T AtomicFetchOr(ThreadState *thr, uptr pc, volatile T *a, T v,
    morder mo) {
  if (IsReleaseOrder(mo))
    Release(thr, pc, (uptr)a);
  v = __sync_fetch_and_or(a, v);
  if (IsAcquireOrder(mo))
    Acquire(thr, pc, (uptr)a);
  return v;
}

template<typename T>
static T AtomicFetchXor(ThreadState *thr, uptr pc, volatile T *a, T v,
    morder mo) {
  if (IsReleaseOrder(mo))
    Release(thr, pc, (uptr)a);
  v = __sync_fetch_and_xor(a, v);
  if (IsAcquireOrder(mo))
    Acquire(thr, pc, (uptr)a);
  return v;
}

template<typename T>
static bool AtomicCAS(ThreadState *thr, uptr pc,
    volatile T *a, T *c, T v, morder mo) {
  if (IsReleaseOrder(mo))
    Release(thr, pc, (uptr)a);
  T cc = *c;
  T pr = __sync_val_compare_and_swap(a, cc, v);
  if (IsAcquireOrder(mo))
    Acquire(thr, pc, (uptr)a);
  if (pr == cc)
    return true;
  *c = pr;
  return false;
}

template<typename T>
static T AtomicCAS(ThreadState *thr, uptr pc,
    volatile T *a, T c, T v, morder mo) {
  AtomicCAS(thr, pc, a, &c, v, mo);
  return c;
}

static void AtomicFence(ThreadState *thr, uptr pc, morder mo) {
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

int __tsan_atomic8_compare_exchange_strong(volatile a8 *a, a8 *c, a8 v,
    morder mo) {
  SCOPED_ATOMIC(CAS, a, c, v, mo);
}

int __tsan_atomic16_compare_exchange_strong(volatile a16 *a, a16 *c, a16 v,
    morder mo) {
  SCOPED_ATOMIC(CAS, a, c, v, mo);
}

int __tsan_atomic32_compare_exchange_strong(volatile a32 *a, a32 *c, a32 v,
    morder mo) {
  SCOPED_ATOMIC(CAS, a, c, v, mo);
}

int __tsan_atomic64_compare_exchange_strong(volatile a64 *a, a64 *c, a64 v,
    morder mo) {
  SCOPED_ATOMIC(CAS, a, c, v, mo);
}

int __tsan_atomic8_compare_exchange_weak(volatile a8 *a, a8 *c, a8 v,
    morder mo) {
  SCOPED_ATOMIC(CAS, a, c, v, mo);
}

int __tsan_atomic16_compare_exchange_weak(volatile a16 *a, a16 *c, a16 v,
    morder mo) {
  SCOPED_ATOMIC(CAS, a, c, v, mo);
}

int __tsan_atomic32_compare_exchange_weak(volatile a32 *a, a32 *c, a32 v,
    morder mo) {
  SCOPED_ATOMIC(CAS, a, c, v, mo);
}

int __tsan_atomic64_compare_exchange_weak(volatile a64 *a, a64 *c, a64 v,
    morder mo) {
  SCOPED_ATOMIC(CAS, a, c, v, mo);
}

a8 __tsan_atomic8_compare_exchange_val(volatile a8 *a, a8 c, a8 v,
    morder mo) {
  SCOPED_ATOMIC(CAS, a, c, v, mo);
}
a16 __tsan_atomic16_compare_exchange_val(volatile a16 *a, a16 c, a16 v,
    morder mo) {
  SCOPED_ATOMIC(CAS, a, c, v, mo);
}

a32 __tsan_atomic32_compare_exchange_val(volatile a32 *a, a32 c, a32 v,
    morder mo) {
  SCOPED_ATOMIC(CAS, a, c, v, mo);
}

a64 __tsan_atomic64_compare_exchange_val(volatile a64 *a, a64 c, a64 v,
    morder mo) {
  SCOPED_ATOMIC(CAS, a, c, v, mo);
}

void __tsan_atomic_thread_fence(morder mo) {
  char* a;
  SCOPED_ATOMIC(Fence, mo);
}

void __tsan_atomic_signal_fence(morder mo) {
}
