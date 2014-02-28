//===-- dd_rtl.cc ---------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "dd_rtl.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "sanitizer_common/sanitizer_stackdepot.h"

namespace __dsan {

static Context ctx0;
static Context * const ctx = &ctx0;

void Initialize() {
  InitializeInterceptors();
  //common_flags()->allow_addr2line = true;
  common_flags()->symbolize = true;
  ctx->dd = DDetector::Create();
}

void ThreadInit(Thread *thr) {
  thr->dd_pt = ctx->dd->CreatePhysicalThread();
  thr->dd_lt = ctx->dd->CreateLogicalThread(0);
}

void ThreadDestroy(Thread *thr) {
  ctx->dd->DestroyPhysicalThread(thr->dd_pt);
  ctx->dd->DestroyLogicalThread(thr->dd_lt);
}

static u32 CurrentStackTrace(Thread *thr) {
  StackTrace trace;
  thr->in_symbolizer = true;
  trace.Unwind(1000, 0, 0, 0, 0, 0, false);
  thr->in_symbolizer = false;
  const uptr skip = 4;
  if (trace.size <= skip)
    return 0;
  return StackDepotPut(trace.trace + skip, trace.size - skip);
}

static void PrintStackTrace(Thread *thr, u32 stk) {
  uptr size = 0;
  const uptr *trace = StackDepotGet(stk, &size);
  thr->in_symbolizer = true;
  StackTrace::PrintStack(trace, size);
  thr->in_symbolizer = false;
}

static Mutex *FindMutex(Thread *thr, uptr m) {
  SpinMutexLock l(&ctx->mutex_mtx);
  for (Mutex *mtx = ctx->mutex_list; mtx; mtx = mtx->link) {
    if (mtx->addr == m)
      return mtx;
  }
  Mutex *mtx = (Mutex*)InternalAlloc(sizeof(*mtx));
  internal_memset(mtx, 0, sizeof(*mtx));
  mtx->addr = m;
  ctx->dd->MutexInit(&mtx->dd, CurrentStackTrace(thr), ctx->mutex_seq++);
  mtx->link = ctx->mutex_list;
  ctx->mutex_list = mtx;
  return mtx;
}

static Mutex *FindMutexAndRemove(uptr m) {
  SpinMutexLock l(&ctx->mutex_mtx);
  Mutex **prev = &ctx->mutex_list;
  for (;;) {
    Mutex *mtx = *prev;
    if (mtx == 0)
      return 0;
    if (mtx->addr == m) {
      *prev = mtx->link;
      return mtx;
    }
    prev = &mtx->link;
  }
}

static void ReportDeadlock(Thread *thr, DDReport *rep) {
  Printf("==============================\n");
  Printf("DEADLOCK\n");
  PrintStackTrace(thr, CurrentStackTrace(thr));
  for (int i = 0; i < rep->n; i++) {
    Printf("Mutex %llu created at:\n", rep->loop[i].mtx_ctx0);
    PrintStackTrace(thr, rep->loop[i].stk);
  }
  Printf("==============================\n");
}

void MutexLock(Thread *thr, uptr m, bool writelock, bool trylock) {
  if (thr->in_symbolizer)
    return;
  Mutex *mtx = FindMutex(thr, m);
  DDReport *rep = ctx->dd->MutexLock(thr->dd_pt, thr->dd_lt, &mtx->dd,
      writelock, trylock);
  if (rep)
    ReportDeadlock(thr, rep);
}

void MutexUnlock(Thread *thr, uptr m, bool writelock) {
  if (thr->in_symbolizer)
    return;
  Mutex *mtx = FindMutex(thr, m);
  ctx->dd->MutexUnlock(thr->dd_pt, thr->dd_lt, &mtx->dd, writelock);
}

void MutexDestroy(Thread *thr, uptr m) {
  if (thr->in_symbolizer)
    return;
  Mutex *mtx = FindMutexAndRemove(m);
  if (mtx == 0)
    return;
  ctx->dd->MutexDestroy(thr->dd_pt, thr->dd_lt, &mtx->dd);
  InternalFree(mtx);
}

}  // namespace __dsan

__attribute__((section(".preinit_array"), used))
void (*__local_dsan_preinit)(void) = __dsan::Initialize;
