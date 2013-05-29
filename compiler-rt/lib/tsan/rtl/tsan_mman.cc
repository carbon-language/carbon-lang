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

COMPILER_CHECK(sizeof(MBlock) == 16);

void MBlock::Lock() {
  atomic_uintptr_t *a = reinterpret_cast<atomic_uintptr_t*>(this);
  uptr v = atomic_load(a, memory_order_relaxed);
  for (int iter = 0;; iter++) {
    if (v & 1) {
      if (iter < 10)
        proc_yield(20);
      else
        internal_sched_yield();
      v = atomic_load(a, memory_order_relaxed);
      continue;
    }
    if (atomic_compare_exchange_weak(a, &v, v | 1, memory_order_acquire))
      break;
  }
}

void MBlock::Unlock() {
  atomic_uintptr_t *a = reinterpret_cast<atomic_uintptr_t*>(this);
  uptr v = atomic_load(a, memory_order_relaxed);
  DCHECK(v & 1);
  atomic_store(a, v & ~1, memory_order_relaxed);
}

struct MapUnmapCallback {
  void OnMap(uptr p, uptr size) const { }
  void OnUnmap(uptr p, uptr size) const {
    // We are about to unmap a chunk of user memory.
    // Mark the corresponding shadow memory as not needed.
    DontNeedShadowFor(p, size);
  }
};

static char allocator_placeholder[sizeof(Allocator)] ALIGNED(64);
Allocator *allocator() {
  return reinterpret_cast<Allocator*>(&allocator_placeholder);
}

void InitializeAllocator() {
  allocator()->Init();
}

void AllocatorThreadStart(ThreadState *thr) {
  allocator()->InitCache(&thr->alloc_cache);
  internal_allocator()->InitCache(&thr->internal_alloc_cache);
}

void AllocatorThreadFinish(ThreadState *thr) {
  allocator()->DestroyCache(&thr->alloc_cache);
  internal_allocator()->DestroyCache(&thr->internal_alloc_cache);
}

void AllocatorPrintStats() {
  allocator()->PrintStats();
}

static void SignalUnsafeCall(ThreadState *thr, uptr pc) {
  if (!thr->in_signal_handler || !flags()->report_signal_unsafe)
    return;
  Context *ctx = CTX();
  StackTrace stack;
  stack.ObtainCurrent(thr, pc);
  ThreadRegistryLock l(ctx->thread_registry);
  ScopedReport rep(ReportTypeSignalUnsafe);
  if (!IsFiredSuppression(ctx, rep, stack)) {
    rep.AddStack(&stack);
    OutputReport(ctx, rep, rep.GetReport()->stacks[0]);
  }
}

void *user_alloc(ThreadState *thr, uptr pc, uptr sz, uptr align) {
  CHECK_GT(thr->in_rtl, 0);
  if ((sz >= (1ull << 40)) || (align >= (1ull << 40)))
    return 0;
  void *p = allocator()->Allocate(&thr->alloc_cache, sz, align);
  if (p == 0)
    return 0;
  MBlock *b = new(allocator()->GetMetaData(p)) MBlock;
  b->Init(sz, thr->tid, CurrentStackId(thr, pc));
  if (CTX() && CTX()->initialized)
    MemoryRangeImitateWrite(thr, pc, (uptr)p, sz);
  DPrintf("#%d: alloc(%zu) = %p\n", thr->tid, sz, p);
  SignalUnsafeCall(thr, pc);
  return p;
}

void user_free(ThreadState *thr, uptr pc, void *p) {
  CHECK_GT(thr->in_rtl, 0);
  CHECK_NE(p, (void*)0);
  DPrintf("#%d: free(%p)\n", thr->tid, p);
  MBlock *b = (MBlock*)allocator()->GetMetaData(p);
  if (b->ListHead()) {
    MBlock::ScopedLock l(b);
    for (SyncVar *s = b->ListHead(); s;) {
      SyncVar *res = s;
      s = s->next;
      StatInc(thr, StatSyncDestroyed);
      res->mtx.Lock();
      res->mtx.Unlock();
      DestroyAndFree(res);
    }
    b->ListReset();
  }
  if (CTX() && CTX()->initialized && thr->in_rtl == 1)
    MemoryRangeFreed(thr, pc, (uptr)p, b->Size());
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
      CHECK_NE(b, 0);
      internal_memcpy(p2, p, min(b->Size(), sz));
    }
  }
  if (p)
    user_free(thr, pc, p);
  return p2;
}

uptr user_alloc_usable_size(ThreadState *thr, uptr pc, void *p) {
  CHECK_GT(thr->in_rtl, 0);
  if (p == 0)
    return 0;
  MBlock *b = (MBlock*)allocator()->GetMetaData(p);
  return b ? b->Size() : 0;
}

MBlock *user_mblock(ThreadState *thr, void *p) {
  CHECK_NE(p, 0);
  Allocator *a = allocator();
  void *b = a->GetBlockBegin(p);
  if (b == 0)
    return 0;
  return (MBlock*)a->GetMetaData(b);
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
  CHECK_LE(sz, InternalSizeClassMap::kMaxSize);
  if (thr->nomalloc) {
    thr->nomalloc = 0;  // CHECK calls internal_malloc().
    CHECK(0);
  }
  return InternalAlloc(sz, &thr->internal_alloc_cache);
}

void internal_free(void *p) {
  ThreadState *thr = cur_thread();
  CHECK_GT(thr->in_rtl, 0);
  if (thr->nomalloc) {
    thr->nomalloc = 0;  // CHECK calls internal_malloc().
    CHECK(0);
  }
  InternalFree(p, &thr->internal_alloc_cache);
}

}  // namespace __tsan

using namespace __tsan;

extern "C" {
uptr __tsan_get_current_allocated_bytes() {
  u64 stats[AllocatorStatCount];
  allocator()->GetStats(stats);
  u64 m = stats[AllocatorStatMalloced];
  u64 f = stats[AllocatorStatFreed];
  return m >= f ? m - f : 1;
}

uptr __tsan_get_heap_size() {
  u64 stats[AllocatorStatCount];
  allocator()->GetStats(stats);
  u64 m = stats[AllocatorStatMmapped];
  u64 f = stats[AllocatorStatUnmapped];
  return m >= f ? m - f : 1;
}

uptr __tsan_get_free_bytes() {
  return 1;
}

uptr __tsan_get_unmapped_bytes() {
  return 1;
}

uptr __tsan_get_estimated_allocated_size(uptr size) {
  return size;
}

bool __tsan_get_ownership(void *p) {
  return allocator()->GetBlockBegin(p) != 0;
}

uptr __tsan_get_allocated_size(void *p) {
  if (p == 0)
    return 0;
  p = allocator()->GetBlockBegin(p);
  if (p == 0)
    return 0;
  MBlock *b = (MBlock*)allocator()->GetMetaData(p);
  return b->Size();
}

void __tsan_on_thread_idle() {
  ThreadState *thr = cur_thread();
  allocator()->SwallowCache(&thr->alloc_cache);
  internal_allocator()->SwallowCache(&thr->internal_alloc_cache);
}
}  // extern "C"
