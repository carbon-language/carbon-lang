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

namespace __sanitizer {

typedef TwoLevelBitVector<> DDBV;  // DeadlockDetector's bit vector.

struct DDPhysicalThread {
};

struct DDLogicalThread {
  u64 ctx;
  DeadlockDetectorTLS<DDBV> dd;
  DDReport rep;
};

struct DDetectorImpl : public DDetector {
  SpinMutex mtx;
  DeadlockDetector<DDBV> dd;

  DDetectorImpl();

  virtual DDPhysicalThread* CreatePhysicalThread();
  virtual void DestroyPhysicalThread(DDPhysicalThread *pt);

  virtual DDLogicalThread* CreateLogicalThread(u64 ctx);
  virtual void DestroyLogicalThread(DDLogicalThread *lt);

  virtual void MutexInit(DDMutex *m, u32 stk, u64 ctx);
  virtual DDReport *MutexLock(DDPhysicalThread *pt, DDLogicalThread *lt,
      DDMutex *m, bool writelock, bool trylock);
  virtual DDReport *MutexUnlock(DDPhysicalThread *pt, DDLogicalThread *lt,
      DDMutex *m, bool writelock);
  virtual void MutexDestroy(DDPhysicalThread *pt, DDLogicalThread *lt,
      DDMutex *m);

  void MutexEnsureID(DDLogicalThread *lt, DDMutex *m);
};

DDetector *DDetector::Create() {
  void *mem = MmapOrDie(sizeof(DDetectorImpl), "deadlock detector");
  return new(mem) DDetectorImpl();
}

DDetectorImpl::DDetectorImpl() {
  dd.clear();
}

DDPhysicalThread* DDetectorImpl::CreatePhysicalThread() {
  return 0;
}

void DDetectorImpl::DestroyPhysicalThread(DDPhysicalThread *pt) {
}

DDLogicalThread* DDetectorImpl::CreateLogicalThread(u64 ctx) {
  DDLogicalThread *lt = (DDLogicalThread*)InternalAlloc(sizeof(*lt));
  lt->ctx = ctx;
  lt->dd.clear();
  return lt;
}

void DDetectorImpl::DestroyLogicalThread(DDLogicalThread *lt) {
  lt->~DDLogicalThread();
  InternalFree(lt);
}

void DDetectorImpl::MutexInit(DDMutex *m, u32 stk, u64 ctx) {
  m->id = 0;
  m->stk = stk;
  m->ctx = ctx;
}

void DDetectorImpl::MutexEnsureID(DDLogicalThread *lt, DDMutex *m) {
  if (!dd.nodeBelongsToCurrentEpoch(m->id))
    m->id = dd.newNode(reinterpret_cast<uptr>(m));
  dd.ensureCurrentEpoch(&lt->dd);
}

DDReport *DDetectorImpl::MutexLock(DDPhysicalThread *pt, DDLogicalThread *lt,
    DDMutex *m, bool writelock, bool trylock) {
  if (dd.onFirstLock(&lt->dd, m->id))
    return 0;
  SpinMutexLock lk(&mtx);
  MutexEnsureID(lt, m);
  CHECK(!dd.isHeld(&lt->dd, m->id));
  // Printf("T%d MutexLock:   %zx\n", thr->tid, s->deadlock_detector_id);
  bool has_deadlock = trylock
      ? dd.onTryLock(&lt->dd, m->id)
       : dd.onLock(&lt->dd, m->id);
  DDReport *rep = 0;
  if (has_deadlock) {
    uptr path[10];
    uptr len = dd.findPathToHeldLock(&lt->dd, m->id,
                                          path, ARRAY_SIZE(path));
    CHECK_GT(len, 0U);  // Hm.. cycle of 10 locks? I'd like to see that.
    rep = &lt->rep;
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
  return rep;
}

DDReport *DDetectorImpl::MutexUnlock(DDPhysicalThread *pt, DDLogicalThread *lt,
    DDMutex *m, bool writelock) {
  // Printf("T%d MutexUnlock: %zx; recursion %d\n", thr->tid,
  //        s->deadlock_detector_id, s->recursion);
  dd.onUnlock(&lt->dd, m->id);
  return 0;
}

void DDetectorImpl::MutexDestroy(DDPhysicalThread *pt, DDLogicalThread *lt,
    DDMutex *m) {
  if (!m->id) return;
  SpinMutexLock lk(&mtx);
  if (dd.nodeBelongsToCurrentEpoch(m->id))
    dd.removeNode(m->id);
  m->id = 0;
}

}  // namespace __sanitizer
