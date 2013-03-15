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

#include "tsan_rtl.h"
#include "tsan_flags.h"
#include "tsan_sync.h"
#include "tsan_report.h"
#include "tsan_symbolize.h"
#include "tsan_platform.h"

namespace __tsan {

void MutexCreate(ThreadState *thr, uptr pc, uptr addr,
                 bool rw, bool recursive, bool linker_init) {
  Context *ctx = CTX();
  CHECK_GT(thr->in_rtl, 0);
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
  CHECK_GT(thr->in_rtl, 0);
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

void MutexLock(ThreadState *thr, uptr pc, uptr addr) {
  CHECK_GT(thr->in_rtl, 0);
  DPrintf("#%d: MutexLock %zx\n", thr->tid, addr);
  if (IsAppMem(addr))
    MemoryReadAtomic(thr, pc, addr, kSizeLog1);
  SyncVar *s = CTX()->synctab.GetOrCreateAndLock(thr, pc, addr, true);
  thr->fast_state.IncrementEpoch();
  TraceAddEvent(thr, thr->fast_state, EventTypeLock, s->GetId());
  if (s->owner_tid == SyncVar::kInvalidTid) {
    CHECK_EQ(s->recursion, 0);
    s->owner_tid = thr->tid;
    s->last_lock = thr->fast_state.raw();
  } else if (s->owner_tid == thr->tid) {
    CHECK_GT(s->recursion, 0);
  } else {
    Printf("ThreadSanitizer WARNING: double lock\n");
    PrintCurrentStack(thr, pc);
  }
  if (s->recursion == 0) {
    StatInc(thr, StatMutexLock);
    thr->clock.set(thr->tid, thr->fast_state.epoch());
    thr->clock.acquire(&s->clock);
    StatInc(thr, StatSyncAcquire);
    thr->clock.acquire(&s->read_clock);
    StatInc(thr, StatSyncAcquire);
  } else if (!s->is_recursive) {
    StatInc(thr, StatMutexRecLock);
  }
  s->recursion++;
  thr->mset.Add(s->GetId(), true, thr->fast_state.epoch());
  s->mtx.Unlock();
}

void MutexUnlock(ThreadState *thr, uptr pc, uptr addr) {
  CHECK_GT(thr->in_rtl, 0);
  DPrintf("#%d: MutexUnlock %zx\n", thr->tid, addr);
  if (IsAppMem(addr))
    MemoryReadAtomic(thr, pc, addr, kSizeLog1);
  SyncVar *s = CTX()->synctab.GetOrCreateAndLock(thr, pc, addr, true);
  thr->fast_state.IncrementEpoch();
  TraceAddEvent(thr, thr->fast_state, EventTypeUnlock, s->GetId());
  if (s->recursion == 0) {
    if (!s->is_broken) {
      s->is_broken = true;
      Printf("ThreadSanitizer WARNING: unlock of unlocked mutex\n");
      PrintCurrentStack(thr, pc);
    }
  } else if (s->owner_tid != thr->tid) {
    if (!s->is_broken) {
      s->is_broken = true;
      Printf("ThreadSanitizer WARNING: mutex unlock by another thread\n");
      PrintCurrentStack(thr, pc);
    }
  } else {
    s->recursion--;
    if (s->recursion == 0) {
      StatInc(thr, StatMutexUnlock);
      s->owner_tid = SyncVar::kInvalidTid;
      thr->clock.set(thr->tid, thr->fast_state.epoch());
      thr->fast_synch_epoch = thr->fast_state.epoch();
      thr->clock.ReleaseStore(&s->clock);
      StatInc(thr, StatSyncRelease);
    } else {
      StatInc(thr, StatMutexRecUnlock);
    }
  }
  thr->mset.Del(s->GetId(), true);
  s->mtx.Unlock();
}

void MutexReadLock(ThreadState *thr, uptr pc, uptr addr) {
  CHECK_GT(thr->in_rtl, 0);
  DPrintf("#%d: MutexReadLock %zx\n", thr->tid, addr);
  StatInc(thr, StatMutexReadLock);
  if (IsAppMem(addr))
    MemoryReadAtomic(thr, pc, addr, kSizeLog1);
  SyncVar *s = CTX()->synctab.GetOrCreateAndLock(thr, pc, addr, false);
  thr->fast_state.IncrementEpoch();
  TraceAddEvent(thr, thr->fast_state, EventTypeRLock, s->GetId());
  if (s->owner_tid != SyncVar::kInvalidTid) {
    Printf("ThreadSanitizer WARNING: read lock of a write locked mutex\n");
    PrintCurrentStack(thr, pc);
  }
  thr->clock.set(thr->tid, thr->fast_state.epoch());
  thr->clock.acquire(&s->clock);
  s->last_lock = thr->fast_state.raw();
  StatInc(thr, StatSyncAcquire);
  thr->mset.Add(s->GetId(), false, thr->fast_state.epoch());
  s->mtx.ReadUnlock();
}

void MutexReadUnlock(ThreadState *thr, uptr pc, uptr addr) {
  CHECK_GT(thr->in_rtl, 0);
  DPrintf("#%d: MutexReadUnlock %zx\n", thr->tid, addr);
  StatInc(thr, StatMutexReadUnlock);
  if (IsAppMem(addr))
    MemoryReadAtomic(thr, pc, addr, kSizeLog1);
  SyncVar *s = CTX()->synctab.GetOrCreateAndLock(thr, pc, addr, true);
  thr->fast_state.IncrementEpoch();
  TraceAddEvent(thr, thr->fast_state, EventTypeRUnlock, s->GetId());
  if (s->owner_tid != SyncVar::kInvalidTid) {
    Printf("ThreadSanitizer WARNING: read unlock of a write "
               "locked mutex\n");
    PrintCurrentStack(thr, pc);
  }
  thr->clock.set(thr->tid, thr->fast_state.epoch());
  thr->fast_synch_epoch = thr->fast_state.epoch();
  thr->clock.release(&s->read_clock);
  StatInc(thr, StatSyncRelease);
  s->mtx.Unlock();
  thr->mset.Del(s->GetId(), false);
}

void MutexReadOrWriteUnlock(ThreadState *thr, uptr pc, uptr addr) {
  CHECK_GT(thr->in_rtl, 0);
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
    thr->clock.set(thr->tid, thr->fast_state.epoch());
    thr->fast_synch_epoch = thr->fast_state.epoch();
    thr->clock.release(&s->read_clock);
    StatInc(thr, StatSyncRelease);
  } else if (s->owner_tid == thr->tid) {
    // Seems to be write unlock.
    thr->fast_state.IncrementEpoch();
    TraceAddEvent(thr, thr->fast_state, EventTypeUnlock, s->GetId());
    CHECK_GT(s->recursion, 0);
    s->recursion--;
    if (s->recursion == 0) {
      StatInc(thr, StatMutexUnlock);
      s->owner_tid = SyncVar::kInvalidTid;
      // FIXME: Refactor me, plz.
      // The sequence of events is quite tricky and doubled in several places.
      // First, it's a bug to increment the epoch w/o writing to the trace.
      // Then, the acquire/release logic can be factored out as well.
      thr->clock.set(thr->tid, thr->fast_state.epoch());
      thr->fast_synch_epoch = thr->fast_state.epoch();
      thr->clock.ReleaseStore(&s->clock);
      StatInc(thr, StatSyncRelease);
    } else {
      StatInc(thr, StatMutexRecUnlock);
    }
  } else if (!s->is_broken) {
    s->is_broken = true;
    Printf("ThreadSanitizer WARNING: mutex unlock by another thread\n");
    PrintCurrentStack(thr, pc);
  }
  thr->mset.Del(s->GetId(), write);
  s->mtx.Unlock();
}

void Acquire(ThreadState *thr, uptr pc, uptr addr) {
  CHECK_GT(thr->in_rtl, 0);
  DPrintf("#%d: Acquire %zx\n", thr->tid, addr);
  SyncVar *s = CTX()->synctab.GetOrCreateAndLock(thr, pc, addr, false);
  thr->clock.set(thr->tid, thr->fast_state.epoch());
  thr->clock.acquire(&s->clock);
  StatInc(thr, StatSyncAcquire);
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
  ThreadRegistryLock l(CTX()->thread_registry);
  CTX()->thread_registry->RunCallbackForEachThreadLocked(
      UpdateClockCallback, thr);
}

void Release(ThreadState *thr, uptr pc, uptr addr) {
  CHECK_GT(thr->in_rtl, 0);
  DPrintf("#%d: Release %zx\n", thr->tid, addr);
  SyncVar *s = CTX()->synctab.GetOrCreateAndLock(thr, pc, addr, true);
  thr->clock.set(thr->tid, thr->fast_state.epoch());
  thr->clock.release(&s->clock);
  StatInc(thr, StatSyncRelease);
  s->mtx.Unlock();
}

void ReleaseStore(ThreadState *thr, uptr pc, uptr addr) {
  CHECK_GT(thr->in_rtl, 0);
  DPrintf("#%d: ReleaseStore %zx\n", thr->tid, addr);
  SyncVar *s = CTX()->synctab.GetOrCreateAndLock(thr, pc, addr, true);
  thr->clock.set(thr->tid, thr->fast_state.epoch());
  thr->clock.ReleaseStore(&s->clock);
  StatInc(thr, StatSyncRelease);
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
  thr->last_sleep_stack_id = CurrentStackId(thr, pc);
  ThreadRegistryLock l(CTX()->thread_registry);
  CTX()->thread_registry->RunCallbackForEachThreadLocked(
      UpdateSleepClockCallback, thr);
}
#endif

}  // namespace __tsan
