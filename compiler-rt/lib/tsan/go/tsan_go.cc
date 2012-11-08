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

const int kMaxGoroutinesEver = 128*1024;

static ThreadState *goroutines[kMaxGoroutinesEver];

void InitializeInterceptors() {
}

void InitializeDynamicAnnotations() {
}

bool IsExpectedReport(uptr addr, uptr size) {
  return false;
}

void internal_start_thread(void(*func)(void*), void *arg) {
}

ReportStack *SymbolizeData(uptr addr) {
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

// Callback into Go.
extern "C" int __tsan_symbolize(uptr pc, char **func, char **file,
    int *line, int *off);

ReportStack *SymbolizeCode(uptr addr) {
  ReportStack *s = (ReportStack*)internal_alloc(MBlockReportStack,
                                                sizeof(ReportStack));
  internal_memset(s, 0, sizeof(*s));
  s->pc = addr;
  char *func = 0, *file = 0;
  int line = 0, off = 0;
  if (__tsan_symbolize(addr, &func, &file, &line, &off)) {
    s->offset = off;
    s->func = internal_strdup(func ? func : "??");
    s->file = internal_strdup(file ? file : "-");
    s->line = line;
    s->col = 0;
    free(func);
    free(file);
  }
  return s;
}

extern "C" {

static void AllocGoroutine(int tid) {
  if (tid >= kMaxGoroutinesEver) {
    Printf("FATAL: Reached goroutine limit\n");
    Die();
  }
  ThreadState *thr = (ThreadState*)internal_alloc(MBlockThreadContex,
      sizeof(ThreadState));
  internal_memset(thr, 0, sizeof(*thr));
  goroutines[tid] = thr;
}

void __tsan_init() {
  AllocGoroutine(0);
  ThreadState *thr = goroutines[0];
  thr->in_rtl++;
  Initialize(thr);
  thr->in_rtl--;
}

void __tsan_fini() {
  // FIXME: Not necessary thread 0.
  ThreadState *thr = goroutines[0];
  thr->in_rtl++;
  int res = Finalize(thr);
  thr->in_rtl--;
  exit(res);
}

void __tsan_map_shadow(uptr addr, uptr size) {
  MapShadow(addr, size);
}

void __tsan_read(int goid, void *addr, void *pc) {
  ThreadState *thr = goroutines[goid];
  MemoryAccess(thr, (uptr)pc, (uptr)addr, 0, false);
}

void __tsan_write(int goid, void *addr, void *pc) {
  ThreadState *thr = goroutines[goid];
  MemoryAccess(thr, (uptr)pc, (uptr)addr, 0, true);
}

void __tsan_read_range(int goid, void *addr, uptr size, uptr step, void *pc) {
  ThreadState *thr = goroutines[goid];
  for (uptr i = 0; i < size; i += step)
	  MemoryAccess(thr, (uptr)pc, (uptr)addr + i, 0, false);
}

void __tsan_write_range(int goid, void *addr, uptr size, uptr step, void *pc) {
  ThreadState *thr = goroutines[goid];
  for (uptr i = 0; i < size; i += step)
	  MemoryAccess(thr, (uptr)pc, (uptr)addr + i, 0, true);
}

void __tsan_func_enter(int goid, void *pc) {
  ThreadState *thr = goroutines[goid];
  FuncEntry(thr, (uptr)pc);
}

void __tsan_func_exit(int goid) {
  ThreadState *thr = goroutines[goid];
  FuncExit(thr);
}

void __tsan_malloc(int goid, void *p, uptr sz, void *pc) {
  ThreadState *thr = goroutines[goid];
  if (thr == 0)  // probably before __tsan_init()
    return;
  thr->in_rtl++;
  MemoryResetRange(thr, (uptr)pc, (uptr)p, sz);
  MemoryAccessRange(thr, (uptr)pc, (uptr)p, sz, true);
  thr->in_rtl--;
}

void __tsan_free(void *p) {
  (void)p;
}

void __tsan_go_start(int pgoid, int chgoid, void *pc) {
  if (chgoid == 0)
    return;
  AllocGoroutine(chgoid);
  ThreadState *thr = goroutines[chgoid];
  ThreadState *parent = goroutines[pgoid];
  thr->in_rtl++;
  parent->in_rtl++;
  int goid2 = ThreadCreate(parent, (uptr)pc, 0, true);
  ThreadStart(thr, goid2, 0);
  parent->in_rtl--;
  thr->in_rtl--;
}

void __tsan_go_end(int goid) {
  ThreadState *thr = goroutines[goid];
  thr->in_rtl++;
  ThreadFinish(thr);
  thr->in_rtl--;
  internal_free(thr);
  goroutines[goid] = 0;
}

void __tsan_acquire(int goid, void *addr) {
  ThreadState *thr = goroutines[goid];
  thr->in_rtl++;
  Acquire(thr, 0, (uptr)addr);
  thr->in_rtl--;
}

void __tsan_release(int goid, void *addr) {
  ThreadState *thr = goroutines[goid];
  thr->in_rtl++;
  ReleaseStore(thr, 0, (uptr)addr);
  thr->in_rtl--;
}

void __tsan_release_merge(int goid, void *addr) {
  ThreadState *thr = goroutines[goid];
  thr->in_rtl++;
  Release(thr, 0, (uptr)addr);
  thr->in_rtl--;
}

void __tsan_finalizer_goroutine(int goid) {
  ThreadState *thr = goroutines[goid];
  AcquireGlobal(thr, 0);
}

#ifdef _WIN32
// MinGW gcc emits calls to the function.
void ___chkstk_ms(void) {
// The implementation must be along the lines of:
// .code64
// PUBLIC ___chkstk_ms
//     //cfi_startproc()
// ___chkstk_ms:
//     push rcx
//     //cfi_push(%rcx)
//     push rax
//     //cfi_push(%rax)
//     cmp rax, PAGE_SIZE
//     lea rcx, [rsp + 24]
//     jb l_LessThanAPage
// .l_MoreThanAPage:
//     sub rcx, PAGE_SIZE
//     or rcx, 0
//     sub rax, PAGE_SIZE
//     cmp rax, PAGE_SIZE
//     ja l_MoreThanAPage
// .l_LessThanAPage:
//     sub rcx, rax
//     or [rcx], 0
//     pop rax
//     //cfi_pop(%rax)
//     pop rcx
//     //cfi_pop(%rcx)
//     ret
//     //cfi_endproc()
// END
}
#endif

}  // extern "C"
}  // namespace __tsan
