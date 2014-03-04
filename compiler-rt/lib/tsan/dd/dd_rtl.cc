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
#include "sanitizer_common/sanitizer_placement_new.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "sanitizer_common/sanitizer_stackdepot.h"

namespace __dsan {

static Context *ctx;

void Initialize() {
  static u64 ctx_mem[sizeof(Context) / sizeof(u64) + 1];
  ctx = new(ctx_mem) Context();

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
  thr->ignore_interceptors = true;
  trace.Unwind(1000, 0, 0, 0, 0, 0, false);
  thr->ignore_interceptors = false;
  const uptr skip = 4;
  if (trace.size <= skip)
    return 0;
  return StackDepotPut(trace.trace + skip, trace.size - skip);
}

static void PrintStackTrace(Thread *thr, u32 stk) {
  uptr size = 0;
  const uptr *trace = StackDepotGet(stk, &size);
  thr->ignore_interceptors = true;
  StackTrace::PrintStack(trace, size);
  thr->ignore_interceptors = false;
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
  if (thr->ignore_interceptors)
    return;
  DDReport *rep = 0;
  {
    MutexHashMap::Handle h(&ctx->mutex_map, m);
    if (h.created())
      ctx->dd->MutexInit(&h->dd, CurrentStackTrace(thr), m);
    rep = ctx->dd->MutexLock(thr->dd_pt, thr->dd_lt, &h->dd,
                             writelock, trylock);
  }
  if (rep)
    ReportDeadlock(thr, rep);
}

void MutexUnlock(Thread *thr, uptr m, bool writelock) {
  if (thr->ignore_interceptors)
    return;
  MutexHashMap::Handle h(&ctx->mutex_map, m);
  ctx->dd->MutexUnlock(thr->dd_pt, thr->dd_lt, &h->dd, writelock);
}

void MutexDestroy(Thread *thr, uptr m) {
  if (thr->ignore_interceptors)
    return;
  MutexHashMap::Handle h(&ctx->mutex_map, m, true);
  if (!h.exists())
    return;
  ctx->dd->MutexDestroy(thr->dd_pt, thr->dd_lt, &h->dd);
}

}  // namespace __dsan
