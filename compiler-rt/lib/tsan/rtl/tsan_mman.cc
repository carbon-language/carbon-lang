//===-- tsan_mman.cc ------------------------------------------------------===//
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
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_placement_new.h"
#include "tsan_mman.h"
#include "tsan_rtl.h"
#include "tsan_report.h"
#include "tsan_flags.h"

// May be overriden by front-end.
extern "C" void WEAK __tsan_malloc_hook(void *ptr, uptr size) {
  (void)ptr;
  (void)size;
}

extern "C" void WEAK __tsan_free_hook(void *ptr) {
  (void)ptr;
}

namespace __tsan {

static char allocator_placeholder[sizeof(Allocator)] ALIGNED(64);
Allocator *allocator() {
  return reinterpret_cast<Allocator*>(&allocator_placeholder);
}

void InitializeAllocator() {
  allocator()->Init();
}

void AlloctorThreadFinish(ThreadState *thr) {
  allocator()->SwallowCache(&thr->alloc_cache);
}

static void SignalUnsafeCall(ThreadState *thr, uptr pc) {
  if (!thr->in_signal_handler || !flags()->report_signal_unsafe)
    return;
  StackTrace stack;
  stack.ObtainCurrent(thr, pc);
  ScopedReport rep(ReportTypeSignalUnsafe);
  rep.AddStack(&stack);
  OutputReport(rep, rep.GetReport()->stacks[0]);
}

void *user_alloc(ThreadState *thr, uptr pc, uptr sz, uptr align) {
  CHECK_GT(thr->in_rtl, 0);
  void *p = allocator()->Allocate(&thr->alloc_cache, sz, align);
  if (p == 0)
    return 0;
  MBlock *b = (MBlock*)allocator()->GetMetaData(p);
  b->size = sz;
  b->alloc_tid = thr->unique_id;
  b->alloc_stack_id = CurrentStackId(thr, pc);
  if (CTX() && CTX()->initialized) {
    MemoryRangeImitateWrite(thr, pc, (uptr)p, sz);
  }
  DPrintf("#%d: alloc(%zu) = %p\n", thr->tid, sz, p);
  SignalUnsafeCall(thr, pc);
  return p;
}

void user_free(ThreadState *thr, uptr pc, void *p) {
  CHECK_GT(thr->in_rtl, 0);
  CHECK_NE(p, (void*)0);
  DPrintf("#%d: free(%p)\n", thr->tid, p);
  MBlock *b = (MBlock*)allocator()->GetMetaData(p);
  if (b->head)   {
    Lock l(&b->mtx);
    for (SyncVar *s = b->head; s;) {
      SyncVar *res = s;
      s = s->next;
      StatInc(thr, StatSyncDestroyed);
      res->mtx.Lock();
      res->mtx.Unlock();
      DestroyAndFree(res);
    }
    b->head = 0;
  }
  if (CTX() && CTX()->initialized && thr->in_rtl == 1) {
    MemoryRangeFreed(thr, pc, (uptr)p, b->size);
  }
  allocator()->Deallocate(&thr->alloc_cache, p);
  SignalUnsafeCall(thr, pc);
}

void *user_realloc(ThreadState *thr, uptr pc, void *p, uptr sz) {
  CHECK_GT(thr->in_rtl, 0);
  void *p2 = 0;
  // FIXME: Handle "shrinking" more efficiently,
  // it seems that some software actually does this.
  if (sz) {
    p2 = user_alloc(thr, pc, sz);
    if (p2 == 0)
      return 0;
    if (p) {
      MBlock *b = user_mblock(thr, p);
      internal_memcpy(p2, p, min(b->size, sz));
    }
  }
  if (p) {
    user_free(thr, pc, p);
  }
  return p2;
}

MBlock *user_mblock(ThreadState *thr, void *p) {
  // CHECK_GT(thr->in_rtl, 0);
  CHECK_NE(p, (void*)0);
  return (MBlock*)allocator()->GetMetaData(p);
}

void invoke_malloc_hook(void *ptr, uptr size) {
  Context *ctx = CTX();
  ThreadState *thr = cur_thread();
  if (ctx == 0 || !ctx->initialized || thr->in_rtl)
    return;
  __tsan_malloc_hook(ptr, size);
}

void invoke_free_hook(void *ptr) {
  Context *ctx = CTX();
  ThreadState *thr = cur_thread();
  if (ctx == 0 || !ctx->initialized || thr->in_rtl)
    return;
  __tsan_free_hook(ptr);
}

void *internal_alloc(MBlockType typ, uptr sz) {
  ThreadState *thr = cur_thread();
  CHECK_GT(thr->in_rtl, 0);
  if (thr->nomalloc) {
    thr->nomalloc = 0;  // CHECK calls internal_malloc().
    CHECK(0);
  }
  return InternalAlloc(sz);
}

void internal_free(void *p) {
  ThreadState *thr = cur_thread();
  CHECK_GT(thr->in_rtl, 0);
  if (thr->nomalloc) {
    thr->nomalloc = 0;  // CHECK calls internal_malloc().
    CHECK(0);
  }
  InternalFree(p);
}

}  // namespace __tsan
