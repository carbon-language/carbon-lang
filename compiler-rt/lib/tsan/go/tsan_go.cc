//===-- tsan_go.cc --------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// ThreadSanitizer runtime for Go language.
//
//===----------------------------------------------------------------------===//

#include "tsan_rtl.h"
#include "tsan_symbolize.h"
#include "sanitizer_common/sanitizer_common.h"
#include <stdlib.h>

namespace __tsan {

void InitializeInterceptors() {
}

void InitializeDynamicAnnotations() {
}

bool IsExpectedReport(uptr addr, uptr size) {
  return false;
}

void *internal_start_thread(void(*func)(void*), void *arg) {
  return 0;
}

void internal_join_thread(void *th) {
}

ReportLocation *SymbolizeData(uptr addr) {
  return 0;
}

ReportStack *NewReportStackEntry(uptr addr) {
  ReportStack *ent = (ReportStack*)internal_alloc(MBlockReportStack,
                                                  sizeof(ReportStack));
  internal_memset(ent, 0, sizeof(*ent));
  ent->pc = addr;
  return ent;
}

void *internal_alloc(MBlockType typ, uptr sz) {
  return InternalAlloc(sz);
}

void internal_free(void *p) {
  InternalFree(p);
}

struct SymbolizeContext {
  uptr pc;
  char *func;
  char *file;
  uptr line;
  uptr off;
  uptr res;
};

// Callback into Go.
static void (*symbolize_cb)(SymbolizeContext *ctx);

ReportStack *SymbolizeCode(uptr addr) {
  ReportStack *s = (ReportStack*)internal_alloc(MBlockReportStack,
                                                sizeof(ReportStack));
  internal_memset(s, 0, sizeof(*s));
  s->pc = addr;
  SymbolizeContext ctx;
  internal_memset(&ctx, 0, sizeof(ctx));
  ctx.pc = addr;
  symbolize_cb(&ctx);
  if (ctx.res) {
    s->offset = ctx.off;
    s->func = internal_strdup(ctx.func ? ctx.func : "??");
    s->file = internal_strdup(ctx.file ? ctx.file : "-");
    s->line = ctx.line;
    s->col = 0;
  }
  return s;
}

extern "C" {

static ThreadState *main_thr;
static bool inited;

static ThreadState *AllocGoroutine() {
  ThreadState *thr = (ThreadState*)internal_alloc(MBlockThreadContex,
      sizeof(ThreadState));
  internal_memset(thr, 0, sizeof(*thr));
  return thr;
}

void __tsan_init(ThreadState **thrp, void (*cb)(SymbolizeContext *cb)) {
  symbolize_cb = cb;
  ThreadState *thr = AllocGoroutine();
  main_thr = *thrp = thr;
  Initialize(thr);
  inited = true;
}

void __tsan_fini() {
  // FIXME: Not necessary thread 0.
  ThreadState *thr = main_thr;
  int res = Finalize(thr);
  exit(res);
}

void __tsan_map_shadow(uptr addr, uptr size) {
  MapShadow(addr, size);
}

void __tsan_read(ThreadState *thr, void *addr, void *pc) {
  MemoryRead(thr, (uptr)pc, (uptr)addr, kSizeLog1);
}

void __tsan_read_pc(ThreadState *thr, void *addr, uptr callpc, uptr pc) {
  if (callpc != 0)
    FuncEntry(thr, callpc);
  MemoryRead(thr, (uptr)pc, (uptr)addr, kSizeLog1);
  if (callpc != 0)
    FuncExit(thr);
}

void __tsan_write(ThreadState *thr, void *addr, void *pc) {
  MemoryWrite(thr, (uptr)pc, (uptr)addr, kSizeLog1);
}

void __tsan_write_pc(ThreadState *thr, void *addr, uptr callpc, uptr pc) {
  if (callpc != 0)
    FuncEntry(thr, callpc);
  MemoryWrite(thr, (uptr)pc, (uptr)addr, kSizeLog1);
  if (callpc != 0)
    FuncExit(thr);
}

void __tsan_read_range(ThreadState *thr, void *addr, uptr size, uptr pc) {
  MemoryAccessRange(thr, (uptr)pc, (uptr)addr, size, false);
}

void __tsan_write_range(ThreadState *thr, void *addr, uptr size, uptr pc) {
  MemoryAccessRange(thr, (uptr)pc, (uptr)addr, size, true);
}

void __tsan_func_enter(ThreadState *thr, void *pc) {
  FuncEntry(thr, (uptr)pc);
}

void __tsan_func_exit(ThreadState *thr) {
  FuncExit(thr);
}

void __tsan_malloc(void *p, uptr sz) {
  if (!inited)
    return;
  MemoryResetRange(0, 0, (uptr)p, sz);
}

void __tsan_go_start(ThreadState *parent, ThreadState **pthr, void *pc) {
  ThreadState *thr = AllocGoroutine();
  *pthr = thr;
  int goid = ThreadCreate(parent, (uptr)pc, 0, true);
  ThreadStart(thr, goid, 0);
}

void __tsan_go_end(ThreadState *thr) {
  ThreadFinish(thr);
  internal_free(thr);
}

void __tsan_acquire(ThreadState *thr, void *addr) {
  Acquire(thr, 0, (uptr)addr);
}

void __tsan_release(ThreadState *thr, void *addr) {
  ReleaseStore(thr, 0, (uptr)addr);
}

void __tsan_release_merge(ThreadState *thr, void *addr) {
  Release(thr, 0, (uptr)addr);
}

void __tsan_finalizer_goroutine(ThreadState *thr) {
  AcquireGlobal(thr, 0);
}

void __tsan_mutex_before_lock(ThreadState *thr, uptr addr, uptr write) {
}

void __tsan_mutex_after_lock(ThreadState *thr, uptr addr, uptr write) {
  if (write)
    MutexLock(thr, 0, addr);
  else
    MutexReadLock(thr, 0, addr);
}

void __tsan_mutex_before_unlock(ThreadState *thr, uptr addr, uptr write) {
  if (write)
    MutexUnlock(thr, 0, addr);
  else
    MutexReadUnlock(thr, 0, addr);
}

}  // extern "C"
}  // namespace __tsan

namespace __sanitizer {

void SymbolizerPrepareForSandboxing() {
  // Nothing to do here for Go.
}

}  // namespace __sanitizer
