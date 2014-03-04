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
#include "sanitizer_allocator_internal.h"
#include "sanitizer_placement_new.h"
//#include "sanitizer_mutex.h"

#if SANITIZER_DEADLOCK_DETECTOR_VERSION == 2

namespace __sanitizer {

struct DDPhysicalThread {
  DDReport rep;
};

struct DDLogicalThread {
  u64 ctx;
};

struct DDetectorImpl : public DDetector {
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
};

DDetector *DDetector::Create() {
  void *mem = MmapOrDie(sizeof(DDetectorImpl), "deadlock detector");
  return new(mem) DDetectorImpl();
}

DDetectorImpl::DDetectorImpl() {
}

DDPhysicalThread* DDetectorImpl::CreatePhysicalThread() {
  void *mem = InternalAlloc(sizeof(DDPhysicalThread));
  DDPhysicalThread *pt = new(mem) DDPhysicalThread();
  return pt;
}

void DDetectorImpl::DestroyPhysicalThread(DDPhysicalThread *pt) {
  pt->~DDPhysicalThread();
  InternalFree(pt);
}

DDLogicalThread* DDetectorImpl::CreateLogicalThread(u64 ctx) {
  void *mem = InternalAlloc(sizeof(
  DDLogicalThread));
  DDLogicalThread *lt = new(mem) DDLogicalThread();
  lt->ctx = ctx;
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

DDReport *DDetectorImpl::MutexLock(DDPhysicalThread *pt, DDLogicalThread *lt,
    DDMutex *m, bool writelock, bool trylock) {
    /*
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
  */
  return 0;
}

DDReport *DDetectorImpl::MutexUnlock(DDPhysicalThread *pt, DDLogicalThread *lt,
    DDMutex *m, bool writelock) {
  //dd.onUnlock(&lt->dd, m->id);
  return 0;
}

void DDetectorImpl::MutexDestroy(DDPhysicalThread *pt, DDLogicalThread *lt,
    DDMutex *m) {
    /*
  if (!m->id) return;
  SpinMutexLock lk(&mtx);
  if (dd.nodeBelongsToCurrentEpoch(m->id))
    dd.removeNode(m->id);
  m->id = 0;
  */
}

}  // namespace __sanitizer
#endif  // #if SANITIZER_DEADLOCK_DETECTOR_VERSION == 2
