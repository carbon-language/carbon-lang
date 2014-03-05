//===-- sanitizer_deadlock_detector1.cc -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Deadlock detector implementation based on NxN adjacency bit matrix.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_deadlock_detector_interface.h"
#include "sanitizer_deadlock_detector.h"
#include "sanitizer_allocator_internal.h"
#include "sanitizer_placement_new.h"
#include "sanitizer_mutex.h"

#if SANITIZER_DEADLOCK_DETECTOR_VERSION == 1

namespace __sanitizer {

typedef TwoLevelBitVector<> DDBV;  // DeadlockDetector's bit vector.

struct DDPhysicalThread {
};

struct DDLogicalThread {
  u64 ctx;
  DeadlockDetectorTLS<DDBV> dd;
  DDReport rep;
  bool report_pending;
};

struct DD : public DDetector {
  SpinMutex mtx;
  DeadlockDetector<DDBV> dd;

  DD();

  DDPhysicalThread* CreatePhysicalThread();
  void DestroyPhysicalThread(DDPhysicalThread *pt);

  DDLogicalThread* CreateLogicalThread(u64 ctx);
  void DestroyLogicalThread(DDLogicalThread *lt);

  void MutexInit(DDCallback *cb, DDMutex *m);
  void MutexBeforeLock(DDCallback *cb, DDMutex *m, bool wlock);
  void MutexAfterLock(DDCallback *cb, DDMutex *m, bool wlock, bool trylock);
  void MutexBeforeUnlock(DDCallback *cb, DDMutex *m, bool wlock);
  void MutexDestroy(DDCallback *cb, DDMutex *m);

  DDReport *GetReport(DDCallback *cb);

  void MutexEnsureID(DDLogicalThread *lt, DDMutex *m);
};

DDetector *DDetector::Create() {
  void *mem = MmapOrDie(sizeof(DD), "deadlock detector");
  return new(mem) DD();
}

DD::DD() {
  dd.clear();
}

DDPhysicalThread* DD::CreatePhysicalThread() {
  return 0;
}

void DD::DestroyPhysicalThread(DDPhysicalThread *pt) {
}

DDLogicalThread* DD::CreateLogicalThread(u64 ctx) {
  DDLogicalThread *lt = (DDLogicalThread*)InternalAlloc(sizeof(*lt));
  lt->ctx = ctx;
  lt->dd.clear();
  lt->report_pending = false;
  return lt;
}

void DD::DestroyLogicalThread(DDLogicalThread *lt) {
  lt->~DDLogicalThread();
  InternalFree(lt);
}

void DD::MutexInit(DDCallback *cb, DDMutex *m) {
  m->id = 0;
  m->stk = cb->Unwind();
}

void DD::MutexEnsureID(DDLogicalThread *lt, DDMutex *m) {
  if (!dd.nodeBelongsToCurrentEpoch(m->id))
    m->id = dd.newNode(reinterpret_cast<uptr>(m));
  dd.ensureCurrentEpoch(&lt->dd);
}

void DD::MutexBeforeLock(DDCallback *cb,
    DDMutex *m, bool wlock) {
}

void DD::MutexAfterLock(DDCallback *cb, DDMutex *m, bool wlock, bool trylock) {
  DDLogicalThread *lt = cb->lt;
  if (dd.onFirstLock(&lt->dd, m->id))
    return;
  SpinMutexLock lk(&mtx);
  MutexEnsureID(lt, m);
  CHECK(!dd.isHeld(&lt->dd, m->id));
  // Printf("T%d MutexLock:   %zx\n", thr->tid, s->deadlock_detector_id);
  bool has_deadlock = trylock
      ? dd.onTryLock(&lt->dd, m->id)
       : dd.onLock(&lt->dd, m->id);
  if (has_deadlock) {
    uptr path[10];
    uptr len = dd.findPathToHeldLock(&lt->dd, m->id,
                                          path, ARRAY_SIZE(path));
    CHECK_GT(len, 0U);  // Hm.. cycle of 10 locks? I'd like to see that.
    lt->report_pending = true;
    DDReport *rep = &lt->rep;
    rep->n = len;
    for (uptr i = 0; i < len; i++) {
      DDMutex *m0 = (DDMutex*)dd.getData(path[i]);
      DDMutex *m1 = (DDMutex*)dd.getData(path[i < len - 1 ? i + 1 : 0]);
      rep->loop[i].thr_ctx = 0;  // don't know
      rep->loop[i].mtx_ctx0 = m0->ctx;
      rep->loop[i].mtx_ctx1 = m1->ctx;
      rep->loop[i].stk = m0->stk;
    }
  }
}

void DD::MutexBeforeUnlock(DDCallback *cb, DDMutex *m, bool wlock) {
  // Printf("T%d MutexUnlock: %zx; recursion %d\n", thr->tid,
  //        s->deadlock_detector_id, s->recursion);
  dd.onUnlock(&cb->lt->dd, m->id);
}

void DD::MutexDestroy(DDCallback *cb,
    DDMutex *m) {
  if (!m->id) return;
  SpinMutexLock lk(&mtx);
  if (dd.nodeBelongsToCurrentEpoch(m->id))
    dd.removeNode(m->id);
  m->id = 0;
}

DDReport *DD::GetReport(DDCallback *cb) {
  if (!cb->lt->report_pending)
    return 0;
  cb->lt->report_pending = false;
  return &cb->lt->rep;
}

}  // namespace __sanitizer
#endif  // #if SANITIZER_DEADLOCK_DETECTOR_VERSION == 1
