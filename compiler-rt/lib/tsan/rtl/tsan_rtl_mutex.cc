//===-- tsan_rtl_mutex.cc -------------------------------------------------===//
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

#include <sanitizer_common/sanitizer_deadlock_detector.h>

#include "tsan_rtl.h"
#include "tsan_flags.h"
#include "tsan_sync.h"
#include "tsan_report.h"
#include "tsan_symbolize.h"
#include "tsan_platform.h"

namespace __tsan {


static void EnsureDeadlockDetectorID(Context *ctx, SyncVar *s) {
  if (!ctx->dd.nodeBelongsToCurrentEpoch(s->deadlock_detector_id))
    s->deadlock_detector_id = ctx->dd.newNode(reinterpret_cast<uptr>(s));
}

void MutexCreate(ThreadState *thr, uptr pc, uptr addr,
                 bool rw, bool recursive, bool linker_init) {
  Context *ctx = CTX();
  DPrintf("#%d: MutexCreate %zx\n", thr->tid, addr);
  StatInc(thr, StatMutexCreate);
  if (!linker_init && IsAppMem(addr)) {
    CHECK(!thr->is_freeing);
    thr->is_freeing = true;
    MemoryWrite(thr, pc, addr, kSizeLog1);
    thr->is_freeing = false;
  }
  SyncVar *s = ctx->synctab.GetOrCreateAndLock(thr, pc, addr, true);
  s->is_rw = rw;
  s->is_recursive = recursive;
  s->is_linker_init = linker_init;
  s->mtx.Unlock();
}

void MutexDestroy(ThreadState *thr, uptr pc, uptr addr) {
  Context *ctx = CTX();
  DPrintf("#%d: MutexDestroy %zx\n", thr->tid, addr);
  StatInc(thr, StatMutexDestroy);
#ifndef TSAN_GO
  // Global mutexes not marked as LINKER_INITIALIZED
  // cause tons of not interesting reports, so just ignore it.
  if (IsGlobalVar(addr))
    return;
#endif
  SyncVar *s = ctx->synctab.GetAndRemove(thr, pc, addr);
  if (s == 0)
    return;
  if (common_flags()->detect_deadlocks) {
    Lock lk(&ctx->dd_mtx);
    if (ctx->dd.nodeBelongsToCurrentEpoch(s->deadlock_detector_id))
      ctx->dd.removeNode(s->deadlock_detector_id);
    s->deadlock_detector_id = 0;
  }
  if (IsAppMem(addr)) {
    CHECK(!thr->is_freeing);
    thr->is_freeing = true;
    MemoryWrite(thr, pc, addr, kSizeLog1);
    thr->is_freeing = false;
  }
  if (flags()->report_destroy_locked
      && s->owner_tid != SyncVar::kInvalidTid
      && !s->is_broken) {
    s->is_broken = true;
    ThreadRegistryLock l(ctx->thread_registry);
    ScopedReport rep(ReportTypeMutexDestroyLocked);
    rep.AddMutex(s);
    StackTrace trace;
    trace.ObtainCurrent(thr, pc);
    rep.AddStack(&trace);
    FastState last(s->last_lock);
    RestoreStack(last.tid(), last.epoch(), &trace, 0);
    rep.AddStack(&trace);
    rep.AddLocation(s->addr, 1);
    OutputReport(ctx, rep);
  }
  thr->mset.Remove(s->GetId());
  DestroyAndFree(s);
}

void MutexLock(ThreadState *thr, uptr pc, uptr addr, int rec) {
  Context *ctx = CTX();
  DPrintf("#%d: MutexLock %zx rec=%d\n", thr->tid, addr, rec);
  CHECK_GT(rec, 0);
  if (IsAppMem(addr))
    MemoryReadAtomic(thr, pc, addr, kSizeLog1);
  SyncVar *s = ctx->synctab.GetOrCreateAndLock(thr, pc, addr, true);
  thr->fast_state.IncrementEpoch();
  TraceAddEvent(thr, thr->fast_state, EventTypeLock, s->GetId());
  if (s->owner_tid == SyncVar::kInvalidTid) {
    CHECK_EQ(s->recursion, 0);
    s->owner_tid = thr->tid;
    s->last_lock = thr->fast_state.raw();
  } else if (s->owner_tid == thr->tid) {
    CHECK_GT(s->recursion, 0);
  } else {
    Printf("ThreadSanitizer WARNING: double lock of mutex %p\n", addr);
    PrintCurrentStack(thr, pc);
  }
  if (s->recursion == 0) {
    StatInc(thr, StatMutexLock);
    AcquireImpl(thr, pc, &s->clock);
    AcquireImpl(thr, pc, &s->read_clock);
  } else if (!s->is_recursive) {
    StatInc(thr, StatMutexRecLock);
  }
  s->recursion += rec;
  thr->mset.Add(s->GetId(), true, thr->fast_state.epoch());
  if (common_flags()->detect_deadlocks) {
    Lock lk(&ctx->dd_mtx);
    EnsureDeadlockDetectorID(ctx, s);
    if (ctx->dd.isHeld(&thr->deadlock_detector_tls, s->deadlock_detector_id)) {
      // FIXME: add tests, handle the real recursive locks.
      Printf("ThreadSanitizer: reursive-lock\n");
    }
    // Printf("MutexLock: %zx\n", s->deadlock_detector_id);
    bool has_deadlock =
        ctx->dd.onLock(&thr->deadlock_detector_tls, s->deadlock_detector_id);
    if (has_deadlock) {
      uptr path[10];
      uptr len = ctx->dd.findPathToHeldLock(&thr->deadlock_detector_tls,
                                         s->deadlock_detector_id, path,
                                         ARRAY_SIZE(path));
      CHECK_GT(len, 0U);  // Hm.. cycle of 10 locks? I'd like to see that.
      ThreadRegistryLock l(CTX()->thread_registry);
      ScopedReport rep(ReportTypeDeadlock);
      for (uptr i = 0; i < len; i++)
        rep.AddMutex(reinterpret_cast<SyncVar*>(ctx->dd.getData(path[i])));
      StackTrace trace;
      trace.ObtainCurrent(thr, pc);
      rep.AddStack(&trace);
      OutputReport(CTX(), rep);
    }
  }
  s->mtx.Unlock();
}

int MutexUnlock(ThreadState *thr, uptr pc, uptr addr, bool all) {
  Context *ctx = CTX();
  DPrintf("#%d: MutexUnlock %zx all=%d\n", thr->tid, addr, all);
  if (IsAppMem(addr))
    MemoryReadAtomic(thr, pc, addr, kSizeLog1);
  SyncVar *s = ctx->synctab.GetOrCreateAndLock(thr, pc, addr, true);
  thr->fast_state.IncrementEpoch();
  TraceAddEvent(thr, thr->fast_state, EventTypeUnlock, s->GetId());
  int rec = 0;
  if (s->recursion == 0) {
    if (!s->is_broken) {
      s->is_broken = true;
      Printf("ThreadSanitizer WARNING: unlock of unlocked mutex %p\n", addr);
      PrintCurrentStack(thr, pc);
    }
  } else if (s->owner_tid != thr->tid) {
    if (!s->is_broken) {
      s->is_broken = true;
      Printf("ThreadSanitizer WARNING: mutex %p is unlocked by wrong thread\n",
             addr);
      PrintCurrentStack(thr, pc);
    }
  } else {
    rec = all ? s->recursion : 1;
    s->recursion -= rec;
    if (s->recursion == 0) {
      StatInc(thr, StatMutexUnlock);
      s->owner_tid = SyncVar::kInvalidTid;
      ReleaseStoreImpl(thr, pc, &s->clock);
    } else {
      StatInc(thr, StatMutexRecUnlock);
    }
  }
  thr->mset.Del(s->GetId(), true);
  if (common_flags()->detect_deadlocks) {
    Lock lk(&ctx->dd_mtx);
    EnsureDeadlockDetectorID(ctx, s);
    // Printf("MutexUnlock: %zx\n", s->deadlock_detector_id);
    ctx->dd.onUnlock(&thr->deadlock_detector_tls,
                                 s->deadlock_detector_id);
  }
  s->mtx.Unlock();
  return rec;
}

void MutexReadLock(ThreadState *thr, uptr pc, uptr addr) {
  DPrintf("#%d: MutexReadLock %zx\n", thr->tid, addr);
  StatInc(thr, StatMutexReadLock);
  if (IsAppMem(addr))
    MemoryReadAtomic(thr, pc, addr, kSizeLog1);
  SyncVar *s = CTX()->synctab.GetOrCreateAndLock(thr, pc, addr, false);
  thr->fast_state.IncrementEpoch();
  TraceAddEvent(thr, thr->fast_state, EventTypeRLock, s->GetId());
  if (s->owner_tid != SyncVar::kInvalidTid) {
    Printf("ThreadSanitizer WARNING: read lock of a write locked mutex %p\n",
           addr);
    PrintCurrentStack(thr, pc);
  }
  AcquireImpl(thr, pc, &s->clock);
  s->last_lock = thr->fast_state.raw();
  thr->mset.Add(s->GetId(), false, thr->fast_state.epoch());
  s->mtx.ReadUnlock();
}

void MutexReadUnlock(ThreadState *thr, uptr pc, uptr addr) {
  DPrintf("#%d: MutexReadUnlock %zx\n", thr->tid, addr);
  StatInc(thr, StatMutexReadUnlock);
  if (IsAppMem(addr))
    MemoryReadAtomic(thr, pc, addr, kSizeLog1);
  SyncVar *s = CTX()->synctab.GetOrCreateAndLock(thr, pc, addr, true);
  thr->fast_state.IncrementEpoch();
  TraceAddEvent(thr, thr->fast_state, EventTypeRUnlock, s->GetId());
  if (s->owner_tid != SyncVar::kInvalidTid) {
    Printf("ThreadSanitizer WARNING: read unlock of a write locked mutex %p\n",
           addr);
    PrintCurrentStack(thr, pc);
  }
  ReleaseImpl(thr, pc, &s->read_clock);
  s->mtx.Unlock();
  thr->mset.Del(s->GetId(), false);
}

void MutexReadOrWriteUnlock(ThreadState *thr, uptr pc, uptr addr) {
  DPrintf("#%d: MutexReadOrWriteUnlock %zx\n", thr->tid, addr);
  if (IsAppMem(addr))
    MemoryReadAtomic(thr, pc, addr, kSizeLog1);
  SyncVar *s = CTX()->synctab.GetOrCreateAndLock(thr, pc, addr, true);
  bool write = true;
  if (s->owner_tid == SyncVar::kInvalidTid) {
    // Seems to be read unlock.
    write = false;
    StatInc(thr, StatMutexReadUnlock);
    thr->fast_state.IncrementEpoch();
    TraceAddEvent(thr, thr->fast_state, EventTypeRUnlock, s->GetId());
    ReleaseImpl(thr, pc, &s->read_clock);
  } else if (s->owner_tid == thr->tid) {
    // Seems to be write unlock.
    thr->fast_state.IncrementEpoch();
    TraceAddEvent(thr, thr->fast_state, EventTypeUnlock, s->GetId());
    CHECK_GT(s->recursion, 0);
    s->recursion--;
    if (s->recursion == 0) {
      StatInc(thr, StatMutexUnlock);
      s->owner_tid = SyncVar::kInvalidTid;
      ReleaseImpl(thr, pc, &s->clock);
    } else {
      StatInc(thr, StatMutexRecUnlock);
    }
  } else if (!s->is_broken) {
    s->is_broken = true;
    Printf("ThreadSanitizer WARNING: mutex %p is unlock by wrong thread\n",
           addr);
    PrintCurrentStack(thr, pc);
  }
  thr->mset.Del(s->GetId(), write);
  s->mtx.Unlock();
}

void MutexRepair(ThreadState *thr, uptr pc, uptr addr) {
  Context *ctx = CTX();
  DPrintf("#%d: MutexRepair %zx\n", thr->tid, addr);
  SyncVar *s = ctx->synctab.GetOrCreateAndLock(thr, pc, addr, true);
  s->owner_tid = SyncVar::kInvalidTid;
  s->recursion = 0;
  s->mtx.Unlock();
}

void Acquire(ThreadState *thr, uptr pc, uptr addr) {
  DPrintf("#%d: Acquire %zx\n", thr->tid, addr);
  if (thr->ignore_sync)
    return;
  SyncVar *s = CTX()->synctab.GetOrCreateAndLock(thr, pc, addr, false);
  AcquireImpl(thr, pc, &s->clock);
  s->mtx.ReadUnlock();
}

static void UpdateClockCallback(ThreadContextBase *tctx_base, void *arg) {
  ThreadState *thr = reinterpret_cast<ThreadState*>(arg);
  ThreadContext *tctx = static_cast<ThreadContext*>(tctx_base);
  if (tctx->status == ThreadStatusRunning)
    thr->clock.set(tctx->tid, tctx->thr->fast_state.epoch());
  else
    thr->clock.set(tctx->tid, tctx->epoch1);
}

void AcquireGlobal(ThreadState *thr, uptr pc) {
  DPrintf("#%d: AcquireGlobal\n", thr->tid);
  if (thr->ignore_sync)
    return;
  ThreadRegistryLock l(CTX()->thread_registry);
  CTX()->thread_registry->RunCallbackForEachThreadLocked(
      UpdateClockCallback, thr);
}

void Release(ThreadState *thr, uptr pc, uptr addr) {
  DPrintf("#%d: Release %zx\n", thr->tid, addr);
  if (thr->ignore_sync)
    return;
  SyncVar *s = CTX()->synctab.GetOrCreateAndLock(thr, pc, addr, true);
  thr->fast_state.IncrementEpoch();
  // Can't increment epoch w/o writing to the trace as well.
  TraceAddEvent(thr, thr->fast_state, EventTypeMop, 0);
  ReleaseImpl(thr, pc, &s->clock);
  s->mtx.Unlock();
}

void ReleaseStore(ThreadState *thr, uptr pc, uptr addr) {
  DPrintf("#%d: ReleaseStore %zx\n", thr->tid, addr);
  if (thr->ignore_sync)
    return;
  SyncVar *s = CTX()->synctab.GetOrCreateAndLock(thr, pc, addr, true);
  thr->fast_state.IncrementEpoch();
  // Can't increment epoch w/o writing to the trace as well.
  TraceAddEvent(thr, thr->fast_state, EventTypeMop, 0);
  ReleaseStoreImpl(thr, pc, &s->clock);
  s->mtx.Unlock();
}

#ifndef TSAN_GO
static void UpdateSleepClockCallback(ThreadContextBase *tctx_base, void *arg) {
  ThreadState *thr = reinterpret_cast<ThreadState*>(arg);
  ThreadContext *tctx = static_cast<ThreadContext*>(tctx_base);
  if (tctx->status == ThreadStatusRunning)
    thr->last_sleep_clock.set(tctx->tid, tctx->thr->fast_state.epoch());
  else
    thr->last_sleep_clock.set(tctx->tid, tctx->epoch1);
}

void AfterSleep(ThreadState *thr, uptr pc) {
  DPrintf("#%d: AfterSleep %zx\n", thr->tid);
  if (thr->ignore_sync)
    return;
  thr->last_sleep_stack_id = CurrentStackId(thr, pc);
  ThreadRegistryLock l(CTX()->thread_registry);
  CTX()->thread_registry->RunCallbackForEachThreadLocked(
      UpdateSleepClockCallback, thr);
}
#endif

void AcquireImpl(ThreadState *thr, uptr pc, SyncClock *c) {
  if (thr->ignore_sync)
    return;
  thr->clock.set(thr->tid, thr->fast_state.epoch());
  thr->clock.acquire(c);
  StatInc(thr, StatSyncAcquire);
}

void ReleaseImpl(ThreadState *thr, uptr pc, SyncClock *c) {
  if (thr->ignore_sync)
    return;
  thr->clock.set(thr->tid, thr->fast_state.epoch());
  thr->fast_synch_epoch = thr->fast_state.epoch();
  thr->clock.release(c);
  StatInc(thr, StatSyncRelease);
}

void ReleaseStoreImpl(ThreadState *thr, uptr pc, SyncClock *c) {
  if (thr->ignore_sync)
    return;
  thr->clock.set(thr->tid, thr->fast_state.epoch());
  thr->fast_synch_epoch = thr->fast_state.epoch();
  thr->clock.ReleaseStore(c);
  StatInc(thr, StatSyncRelease);
}

void AcquireReleaseImpl(ThreadState *thr, uptr pc, SyncClock *c) {
  if (thr->ignore_sync)
    return;
  thr->clock.set(thr->tid, thr->fast_state.epoch());
  thr->fast_synch_epoch = thr->fast_state.epoch();
  thr->clock.acq_rel(c);
  StatInc(thr, StatSyncAcquire);
  StatInc(thr, StatSyncRelease);
}

}  // namespace __tsan
