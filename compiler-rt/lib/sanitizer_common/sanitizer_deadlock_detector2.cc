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
#include "sanitizer_mutex.h"

#if SANITIZER_DEADLOCK_DETECTOR_VERSION == 2

namespace __sanitizer {

const int kMaxMutex = 1024;
const int kMaxNesting = 64;
const u32 kNoId = -1;
const u32 kEndId = -2;
const int kMaxLink = 8;

struct Id {
  u32 id;
  u32 seq;

  explicit Id(u32 id = 0, u32 seq = 0)
      : id(id)
      , seq(seq) {
  }
};

struct Link {
  u32 id;
  u32 seq;
  u32 tid;
  u32 stk;

  explicit Link(u32 id = 0, u32 seq = 0, u32 tid = 0, u32 stk = 0)
      : id(id)
      , seq(seq)
      , tid(tid)
      , stk(stk) {
  }
};

struct DDPhysicalThread {
  DDReport rep;
  bool report_pending;
  bool visited[kMaxMutex];
  Link pending[kMaxMutex];
  Link path[kMaxMutex];
};

struct DDLogicalThread {
  u64 ctx;
  u32 locked[kMaxNesting];
  int nlocked;
};

struct Mutex {
  StaticSpinMutex mtx;
  u32 seq;
  int nlink;
  Link link[kMaxLink];
};

struct DD : public DDetector {
  explicit DD();

  DDPhysicalThread* CreatePhysicalThread();
  void DestroyPhysicalThread(DDPhysicalThread *pt);

  DDLogicalThread* CreateLogicalThread(u64 ctx);
  void DestroyLogicalThread(DDLogicalThread *lt);

  void MutexInit(DDCallback *cb, DDMutex *m);
  void MutexBeforeLock(DDCallback *cb, DDMutex *m, bool wlock);
  void MutexAfterLock(DDCallback *cb, DDMutex *m, bool wlock,
      bool trylock);
  void MutexBeforeUnlock(DDCallback *cb, DDMutex *m, bool wlock);
  void MutexDestroy(DDCallback *cb, DDMutex *m);

  DDReport *GetReport(DDCallback *cb);

  void CycleCheck(DDPhysicalThread *pt, DDLogicalThread *lt, DDMutex *mtx);
  void Report(DDPhysicalThread *pt, DDLogicalThread *lt, int npath);

  SpinMutex mtx;
  u32 free_id[kMaxMutex];
  int nfree_id;
  int id_gen;

  Mutex mutex[kMaxMutex];
};

DDetector *DDetector::Create() {
  void *mem = MmapOrDie(sizeof(DD), "deadlock detector");
  return new(mem) DD();
}

DD::DD() {
  nfree_id = 0;
  id_gen = 0;
}

DDPhysicalThread* DD::CreatePhysicalThread() {
  DDPhysicalThread *pt = (DDPhysicalThread*)MmapOrDie(sizeof(DDPhysicalThread),
      "deadlock detector (physical thread)");
  return pt;
}

void DD::DestroyPhysicalThread(DDPhysicalThread *pt) {
  pt->~DDPhysicalThread();
  InternalFree(pt);
}

DDLogicalThread* DD::CreateLogicalThread(u64 ctx) {
  DDLogicalThread *lt = (DDLogicalThread*)InternalAlloc(
      sizeof(DDLogicalThread));
  lt->ctx = ctx;
  lt->nlocked = 0;
  return lt;
}

void DD::DestroyLogicalThread(DDLogicalThread *lt) {
  lt->~DDLogicalThread();
  InternalFree(lt);
}

void DD::MutexInit(DDCallback *cb, DDMutex *m) {
  VPrintf(2, "#%llu: DD::MutexInit(%p)\n", cb->lt->ctx, m);
  m->id = kNoId;
  m->recursion = 0;
  atomic_store(&m->owner, 0, memory_order_relaxed);
}

void DD::MutexBeforeLock(DDCallback *cb, DDMutex *m, bool wlock) {
  VPrintf(2, "#%llu: DD::MutexBeforeLock(%p, wlock=%d) nlocked=%d\n",
      cb->lt->ctx, m, wlock, cb->lt->nlocked);
  DDPhysicalThread *pt = cb->pt;
  DDLogicalThread *lt = cb->lt;

  uptr owner = atomic_load(&m->owner, memory_order_relaxed);
  if (owner == (uptr)cb->lt) {
    VPrintf(3, "#%llu: DD::MutexBeforeLock recursive\n",
        cb->lt->ctx);
    return;
  }

  CHECK_LE(lt->nlocked, kMaxNesting);

  if (m->id == kNoId) {
    // FIXME(dvyukov): don't allocate id if lt->nlocked == 0
    SpinMutexLock l(&mtx);
    if (nfree_id > 0)
      m->id = free_id[--nfree_id];
    else
      m->id = id_gen++;
    CHECK_LE(id_gen, kMaxMutex);
    VPrintf(3, "#%llu: DD::MutexBeforeLock assign id %d\n",
        cb->lt->ctx, m->id);
  }

  lt->locked[lt->nlocked++] = m->id;
  if (lt->nlocked == 1) {
    VPrintf(3, "#%llu: DD::MutexBeforeLock first mutex\n",
        cb->lt->ctx);
    return;
  }

  bool added = false;
  Mutex *mtx = &mutex[m->id];
  for (int i = 0; i < lt->nlocked - 1; i++) {
    u32 id1 = lt->locked[i];
    Mutex *mtx1 = &mutex[id1];
    SpinMutexLock l(&mtx1->mtx);
    if (mtx1->nlink == kMaxLink) {
      // FIXME(dvyukov): check stale links
      continue;
    }
    int li = 0;
    for (; li < mtx1->nlink; li++) {
      Link *link = &mtx1->link[li];
      if (link->id == m->id) {
        if (link->seq != mtx->seq) {
          link->seq = mtx->seq;
          link->tid = lt->ctx;
          link->stk = cb->Unwind();
          added = true;
          VPrintf(3, "#%llu: DD::MutexBeforeLock added %d->%d link\n",
              cb->lt->ctx, mtx1 - mutex, m->id);
        }
        break;
      }
    }
    if (li == mtx1->nlink) {
      // FIXME(dvyukov): check stale links
      Link *link = &mtx1->link[mtx1->nlink++];
      link->id = m->id;
      link->seq = mtx->seq;
      link->tid = lt->ctx;
      link->stk = cb->Unwind();
      added = true;
      VPrintf(3, "#%llu: DD::MutexBeforeLock added %d->%d link\n",
          cb->lt->ctx, mtx1 - mutex, m->id);
    }
  }

  if (!added || mtx->nlink == 0) {
    VPrintf(3, "#%llu: DD::MutexBeforeLock don't check\n",
        cb->lt->ctx);
    return;
  }

  CycleCheck(pt, lt, m);
}

void DD::MutexAfterLock(DDCallback *cb, DDMutex *m, bool wlock,
    bool trylock) {
  VPrintf(2, "#%llu: DD::MutexAfterLock(%p, wlock=%d, try=%d) nlocked=%d\n",
      cb->lt->ctx, m, wlock, trylock, cb->lt->nlocked);
  DDLogicalThread *lt = cb->lt;

  uptr owner = atomic_load(&m->owner, memory_order_relaxed);
  if (owner == (uptr)cb->lt) {
    VPrintf(3, "#%llu: DD::MutexAfterLock recursive\n", cb->lt->ctx);
    CHECK(wlock);
    m->recursion++;
    return;
  }
  CHECK_EQ(owner, 0);
  if (wlock) {
    VPrintf(3, "#%llu: DD::MutexAfterLock set owner\n", cb->lt->ctx);
    CHECK_EQ(m->recursion, 0);
    m->recursion = 1;
    atomic_store(&m->owner, (uptr)cb->lt, memory_order_relaxed);
  }

  if (!trylock)
    return;

  CHECK_LE(lt->nlocked, kMaxNesting);
  if (m->id == kNoId) {
    // FIXME(dvyukov): don't allocate id if lt->nlocked == 0
    SpinMutexLock l(&mtx);
    if (nfree_id > 0)
      m->id = free_id[--nfree_id];
    else
      m->id = id_gen++;
    CHECK_LE(id_gen, kMaxMutex);
    VPrintf(3, "#%llu: DD::MutexAfterLock assign id %d\n", cb->lt->ctx, m->id);
  }

  lt->locked[lt->nlocked++] = m->id;
}

void DD::MutexBeforeUnlock(DDCallback *cb, DDMutex *m, bool wlock) {
  VPrintf(2, "#%llu: DD::MutexBeforeUnlock(%p, wlock=%d) nlocked=%d\n",
      cb->lt->ctx, m, wlock, cb->lt->nlocked);
  DDLogicalThread *lt = cb->lt;

  uptr owner = atomic_load(&m->owner, memory_order_relaxed);
  if (owner == (uptr)cb->lt) {
    VPrintf(3, "#%llu: DD::MutexBeforeUnlock recursive\n", cb->lt->ctx);
    if (--m->recursion > 0)
      return;
    VPrintf(3, "#%llu: DD::MutexBeforeUnlock reset owner\n", cb->lt->ctx);
    atomic_store(&m->owner, 0, memory_order_relaxed);
  }
  CHECK_NE(m->id, kNoId);
  int last = lt->nlocked - 1;
  for (int i = last; i >= 0; i--) {
    if (cb->lt->locked[i] == m->id) {
      lt->locked[i] = lt->locked[last];
      lt->nlocked--;
      break;
    }
  }
}

void DD::MutexDestroy(DDCallback *cb, DDMutex *m) {
  VPrintf(2, "#%llu: DD::MutexDestroy(%p)\n",
      cb->lt->ctx, m);
  if (m->id == kNoId)
    return;
  {
    Mutex *mtx = &mutex[m->id];
    SpinMutexLock l(&mtx->mtx);
    mtx->seq++;
    mtx->nlink = 0;
  }
  {
    SpinMutexLock l(&mtx);
    free_id[nfree_id++] = m->id;
    m->id = kNoId;
  }
}

void DD::CycleCheck(DDPhysicalThread *pt, DDLogicalThread *lt,
    DDMutex *m) {
  // Mutex *mtx = &mutex[m->id];
  internal_memset(pt->visited, 0, sizeof(pt->visited));
  int npath = 0;
  int npending = 0;
  {
    Mutex *mtx = &mutex[m->id];
    SpinMutexLock l(&mtx->mtx);
    for (int li = 0; li < mtx->nlink; li++)
      pt->pending[npending++] = mtx->link[li];
  }
  while (npending > 0) {
    Link link = pt->pending[--npending];
    if (link.id == kEndId) {
      npath--;
      continue;
    }
    if (pt->visited[link.id])
      continue;
    Mutex *mtx1 = &mutex[link.id];
    SpinMutexLock l(&mtx1->mtx);
    if (mtx1->seq != link.seq)
      continue;
    pt->visited[link.id] = true;
    if (mtx1->nlink == 0)
      continue;
    pt->path[npath++] = link;
    pt->pending[npending++] = Link(kEndId);
    if (link.id == m->id)
      return Report(pt, lt, npath);  // Bingo!
    for (int li = 0; li < mtx1->nlink; li++) {
      Link *link1 = &mtx1->link[li];
      // Mutex *mtx2 = &mutex[link->id];
      // FIXME(dvyukov): fast seq check
      // FIXME(dvyukov): fast nlink != 0 check
      // FIXME(dvyukov): fast pending check?
      // FIXME(dvyukov): npending can be larger than kMaxMutex
      pt->pending[npending++] = *link1;
    }
  }
}

void DD::Report(DDPhysicalThread *pt, DDLogicalThread *lt, int npath) {
  DDReport *rep = &pt->rep;
  rep->n = npath;
  for (int i = 0; i < npath; i++) {
    Link *link = &pt->path[i];
    Link *link0 = &pt->path[i ? i - 1 : npath - 1];
    rep->loop[i].thr_ctx = link->tid;
    rep->loop[i].mtx_ctx0 = link0->id;
    rep->loop[i].mtx_ctx1 = link->id;
    rep->loop[i].stk = link->stk;
  }
  pt->report_pending = true;
}

DDReport *DD::GetReport(DDCallback *cb) {
  if (!cb->pt->report_pending)
    return 0;
  cb->pt->report_pending = false;
  return &cb->pt->rep;
}

}  // namespace __sanitizer
#endif  // #if SANITIZER_DEADLOCK_DETECTOR_VERSION == 2
