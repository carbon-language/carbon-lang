//===-- tsan_rtl_thread.cc ------------------------------------------------===//
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
#include "tsan_rtl.h"
#include "tsan_mman.h"
#include "tsan_platform.h"
#include "tsan_report.h"
#include "tsan_sync.h"

namespace __tsan {

// ThreadContext implementation.

ThreadContext::ThreadContext(int tid)
  : ThreadContextBase(tid)
  , thr()
  , sync()
  , epoch0()
  , epoch1() {
}

#ifndef TSAN_GO
ThreadContext::~ThreadContext() {
}
#endif

void ThreadContext::OnDead() {
  sync.Reset();
}

void ThreadContext::OnJoined(void *arg) {
  ThreadState *caller_thr = static_cast<ThreadState *>(arg);
  caller_thr->clock.acquire(&sync);
  StatInc(caller_thr, StatSyncAcquire);
  sync.Reset();
}

struct OnCreatedArgs {
  ThreadState *thr;
  uptr pc;
};

void ThreadContext::OnCreated(void *arg) {
  thr = 0;
  if (tid == 0)
    return;
  OnCreatedArgs *args = static_cast<OnCreatedArgs *>(arg);
  args->thr->fast_state.IncrementEpoch();
  // Can't increment epoch w/o writing to the trace as well.
  TraceAddEvent(args->thr, args->thr->fast_state, EventTypeMop, 0);
  args->thr->clock.set(args->thr->tid, args->thr->fast_state.epoch());
  args->thr->fast_synch_epoch = args->thr->fast_state.epoch();
  args->thr->clock.release(&sync);
  StatInc(args->thr, StatSyncRelease);
#ifdef TSAN_GO
  creation_stack.ObtainCurrent(args->thr, args->pc);
#else
  creation_stack_id = CurrentStackId(args->thr, args->pc);
#endif
  if (reuse_count == 0)
    StatInc(args->thr, StatThreadMaxTid);
}

void ThreadContext::OnReset() {
  sync.Reset();
  FlushUnneededShadowMemory(GetThreadTrace(tid), TraceSize() * sizeof(Event));
  //!!! FlushUnneededShadowMemory(GetThreadTraceHeader(tid), sizeof(Trace));
}

struct OnStartedArgs {
  ThreadState *thr;
  uptr stk_addr;
  uptr stk_size;
  uptr tls_addr;
  uptr tls_size;
};

void ThreadContext::OnStarted(void *arg) {
  OnStartedArgs *args = static_cast<OnStartedArgs*>(arg);
  thr = args->thr;
  // RoundUp so that one trace part does not contain events
  // from different threads.
  epoch0 = RoundUp(epoch1 + 1, kTracePartSize);
  epoch1 = (u64)-1;
  new(thr) ThreadState(CTX(), tid, unique_id,
      epoch0, args->stk_addr, args->stk_size, args->tls_addr, args->tls_size);
#ifdef TSAN_GO
  // Setup dynamic shadow stack.
  const int kInitStackSize = 8;
  args->thr->shadow_stack = (uptr*)internal_alloc(MBlockShadowStack,
      kInitStackSize * sizeof(uptr));
  args->thr->shadow_stack_pos = thr->shadow_stack;
  args->thr->shadow_stack_end = thr->shadow_stack + kInitStackSize;
#endif
#ifndef TSAN_GO
  AllocatorThreadStart(args->thr);
#endif
  thr = args->thr;
  thr->fast_synch_epoch = epoch0;
  thr->clock.set(tid, epoch0);
  thr->clock.acquire(&sync);
  thr->fast_state.SetHistorySize(flags()->history_size);
  const uptr trace = (epoch0 / kTracePartSize) % TraceParts();
  Trace *thr_trace = ThreadTrace(thr->tid);
  thr_trace->headers[trace].epoch0 = epoch0;
  StatInc(thr, StatSyncAcquire);
  sync.Reset();
  DPrintf("#%d: ThreadStart epoch=%zu stk_addr=%zx stk_size=%zx "
          "tls_addr=%zx tls_size=%zx\n",
          tid, (uptr)epoch0, args->stk_addr, args->stk_size,
          args->tls_addr, args->tls_size);
  thr->is_alive = true;
}

void ThreadContext::OnFinished() {
  if (!detached) {
    thr->fast_state.IncrementEpoch();
    // Can't increment epoch w/o writing to the trace as well.
    TraceAddEvent(thr, thr->fast_state, EventTypeMop, 0);
    thr->clock.set(thr->tid, thr->fast_state.epoch());
    thr->fast_synch_epoch = thr->fast_state.epoch();
    thr->clock.release(&sync);
    StatInc(thr, StatSyncRelease);
  }
  epoch1 = thr->fast_state.epoch();

#ifndef TSAN_GO
  AllocatorThreadFinish(thr);
#endif
  thr->~ThreadState();
  StatAggregate(CTX()->stat, thr->stat);
  thr = 0;
}

static void MaybeReportThreadLeak(ThreadContextBase *tctx_base, void *unused) {
  ThreadContext *tctx = static_cast<ThreadContext*>(tctx_base);
  if (tctx->detached)
    return;
  if (tctx->status != ThreadStatusCreated
      && tctx->status != ThreadStatusRunning
      && tctx->status != ThreadStatusFinished)
    return;
  ScopedReport rep(ReportTypeThreadLeak);
  rep.AddThread(tctx);
  OutputReport(CTX(), rep);
}

void ThreadFinalize(ThreadState *thr) {
  CHECK_GT(thr->in_rtl, 0);
  if (!flags()->report_thread_leaks)
    return;
  ThreadRegistryLock l(CTX()->thread_registry);
  CTX()->thread_registry->RunCallbackForEachThreadLocked(
      MaybeReportThreadLeak, 0);
}

int ThreadCount(ThreadState *thr) {
  CHECK_GT(thr->in_rtl, 0);
  Context *ctx = CTX();
  uptr result;
  ctx->thread_registry->GetNumberOfThreads(0, 0, &result);
  return (int)result;
}

int ThreadCreate(ThreadState *thr, uptr pc, uptr uid, bool detached) {
  CHECK_GT(thr->in_rtl, 0);
  StatInc(thr, StatThreadCreate);
  Context *ctx = CTX();
  OnCreatedArgs args = { thr, pc };
  int tid = ctx->thread_registry->CreateThread(uid, detached, thr->tid, &args);
  DPrintf("#%d: ThreadCreate tid=%d uid=%zu\n", thr->tid, tid, uid);
  StatSet(thr, StatThreadMaxAlive, ctx->thread_registry->GetMaxAliveThreads());
  return tid;
}

void ThreadStart(ThreadState *thr, int tid, uptr os_id) {
  CHECK_GT(thr->in_rtl, 0);
  uptr stk_addr = 0;
  uptr stk_size = 0;
  uptr tls_addr = 0;
  uptr tls_size = 0;
  GetThreadStackAndTls(tid == 0, &stk_addr, &stk_size, &tls_addr, &tls_size);

  if (tid) {
    if (stk_addr && stk_size)
      MemoryRangeImitateWrite(thr, /*pc=*/ 1, stk_addr, stk_size);

    if (tls_addr && tls_size) {
      // Check that the thr object is in tls;
      const uptr thr_beg = (uptr)thr;
      const uptr thr_end = (uptr)thr + sizeof(*thr);
      CHECK_GE(thr_beg, tls_addr);
      CHECK_LE(thr_beg, tls_addr + tls_size);
      CHECK_GE(thr_end, tls_addr);
      CHECK_LE(thr_end, tls_addr + tls_size);
      // Since the thr object is huge, skip it.
      MemoryRangeImitateWrite(thr, /*pc=*/ 2, tls_addr, thr_beg - tls_addr);
      MemoryRangeImitateWrite(thr, /*pc=*/ 2,
          thr_end, tls_addr + tls_size - thr_end);
    }
  }

  OnStartedArgs args = { thr, stk_addr, stk_size, tls_addr, tls_size };
  CTX()->thread_registry->StartThread(tid, os_id, &args);
}

void ThreadFinish(ThreadState *thr) {
  CHECK_GT(thr->in_rtl, 0);
  StatInc(thr, StatThreadFinish);
  if (thr->stk_addr && thr->stk_size)
    DontNeedShadowFor(thr->stk_addr, thr->stk_size);
  if (thr->tls_addr && thr->tls_size)
    DontNeedShadowFor(thr->tls_addr, thr->tls_size);
  thr->is_alive = false;
  Context *ctx = CTX();
  ctx->thread_registry->FinishThread(thr->tid);
}

static bool FindThreadByUid(ThreadContextBase *tctx, void *arg) {
  uptr uid = (uptr)arg;
  if (tctx->user_id == uid && tctx->status != ThreadStatusInvalid) {
    tctx->user_id = 0;
    return true;
  }
  return false;
}

int ThreadTid(ThreadState *thr, uptr pc, uptr uid) {
  CHECK_GT(thr->in_rtl, 0);
  Context *ctx = CTX();
  int res = ctx->thread_registry->FindThread(FindThreadByUid, (void*)uid);
  DPrintf("#%d: ThreadTid uid=%zu tid=%d\n", thr->tid, uid, res);
  return res;
}

void ThreadJoin(ThreadState *thr, uptr pc, int tid) {
  CHECK_GT(thr->in_rtl, 0);
  CHECK_GT(tid, 0);
  CHECK_LT(tid, kMaxTid);
  DPrintf("#%d: ThreadJoin tid=%d\n", thr->tid, tid);
  Context *ctx = CTX();
  ctx->thread_registry->JoinThread(tid, thr);
}

void ThreadDetach(ThreadState *thr, uptr pc, int tid) {
  CHECK_GT(thr->in_rtl, 0);
  CHECK_GT(tid, 0);
  CHECK_LT(tid, kMaxTid);
  Context *ctx = CTX();
  ctx->thread_registry->DetachThread(tid);
}

void ThreadSetName(ThreadState *thr, const char *name) {
  CHECK_GT(thr->in_rtl, 0);
  CTX()->thread_registry->SetThreadName(thr->tid, name);
}

void MemoryAccessRange(ThreadState *thr, uptr pc, uptr addr,
                       uptr size, bool is_write) {
  if (size == 0)
    return;

  u64 *shadow_mem = (u64*)MemToShadow(addr);
  DPrintf2("#%d: MemoryAccessRange: @%p %p size=%d is_write=%d\n",
      thr->tid, (void*)pc, (void*)addr,
      (int)size, is_write);

#if TSAN_DEBUG
  if (!IsAppMem(addr)) {
    Printf("Access to non app mem %zx\n", addr);
    DCHECK(IsAppMem(addr));
  }
  if (!IsAppMem(addr + size - 1)) {
    Printf("Access to non app mem %zx\n", addr + size - 1);
    DCHECK(IsAppMem(addr + size - 1));
  }
  if (!IsShadowMem((uptr)shadow_mem)) {
    Printf("Bad shadow addr %p (%zx)\n", shadow_mem, addr);
    DCHECK(IsShadowMem((uptr)shadow_mem));
  }
  if (!IsShadowMem((uptr)(shadow_mem + size * kShadowCnt / 8 - 1))) {
    Printf("Bad shadow addr %p (%zx)\n",
               shadow_mem + size * kShadowCnt / 8 - 1, addr + size - 1);
    DCHECK(IsShadowMem((uptr)(shadow_mem + size * kShadowCnt / 8 - 1)));
  }
#endif

  StatInc(thr, StatMopRange);

  FastState fast_state = thr->fast_state;
  if (fast_state.GetIgnoreBit())
    return;

  fast_state.IncrementEpoch();
  thr->fast_state = fast_state;
  TraceAddEvent(thr, fast_state, EventTypeMop, pc);

  bool unaligned = (addr % kShadowCell) != 0;

  // Handle unaligned beginning, if any.
  for (; addr % kShadowCell && size; addr++, size--) {
    int const kAccessSizeLog = 0;
    Shadow cur(fast_state);
    cur.SetWrite(is_write);
    cur.SetAddr0AndSizeLog(addr & (kShadowCell - 1), kAccessSizeLog);
    MemoryAccessImpl(thr, addr, kAccessSizeLog, is_write, false,
        shadow_mem, cur);
  }
  if (unaligned)
    shadow_mem += kShadowCnt;
  // Handle middle part, if any.
  for (; size >= kShadowCell; addr += kShadowCell, size -= kShadowCell) {
    int const kAccessSizeLog = 3;
    Shadow cur(fast_state);
    cur.SetWrite(is_write);
    cur.SetAddr0AndSizeLog(0, kAccessSizeLog);
    MemoryAccessImpl(thr, addr, kAccessSizeLog, is_write, false,
        shadow_mem, cur);
    shadow_mem += kShadowCnt;
  }
  // Handle ending, if any.
  for (; size; addr++, size--) {
    int const kAccessSizeLog = 0;
    Shadow cur(fast_state);
    cur.SetWrite(is_write);
    cur.SetAddr0AndSizeLog(addr & (kShadowCell - 1), kAccessSizeLog);
    MemoryAccessImpl(thr, addr, kAccessSizeLog, is_write, false,
        shadow_mem, cur);
  }
}

void MemoryAccessRangeStep(ThreadState *thr, uptr pc, uptr addr,
    uptr size, uptr step, bool is_write) {
  if (size == 0)
    return;
  FastState fast_state = thr->fast_state;
  if (fast_state.GetIgnoreBit())
    return;
  StatInc(thr, StatMopRange);
  fast_state.IncrementEpoch();
  thr->fast_state = fast_state;
  TraceAddEvent(thr, fast_state, EventTypeMop, pc);

  for (uptr addr_end = addr + size; addr < addr_end; addr += step) {
    u64 *shadow_mem = (u64*)MemToShadow(addr);
    Shadow cur(fast_state);
    cur.SetWrite(is_write);
    cur.SetAddr0AndSizeLog(addr & (kShadowCell - 1), kSizeLog1);
    MemoryAccessImpl(thr, addr, kSizeLog1, is_write, false,
        shadow_mem, cur);
  }
}
}  // namespace __tsan
