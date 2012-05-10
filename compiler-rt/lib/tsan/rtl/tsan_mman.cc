//===-- tsan_mman.cc --------------------------------------------*- C++ -*-===//
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
#include "tsan_mman.h"
#include "tsan_allocator.h"
#include "tsan_rtl.h"
#include "tsan_report.h"
#include "tsan_flags.h"

namespace __tsan {

static void SignalUnsafeCall(ThreadState *thr, uptr pc) {
  if (!thr->in_signal_handler || !flags()->report_signal_unsafe)
    return;
  StackTrace stack;
  stack.ObtainCurrent(thr, pc);
  ScopedReport rep(ReportTypeSignalUnsafe);
  rep.AddStack(&stack);
  OutputReport(rep);
}

void *user_alloc(ThreadState *thr, uptr pc, uptr sz) {
  CHECK_GT(thr->in_rtl, 0);
  MBlock *b = (MBlock*)Alloc(sz + sizeof(MBlock));
  b->size = sz;
  void *p = b + 1;
  if (CTX() && CTX()->initialized) {
    MemoryResetRange(thr, pc, (uptr)p, sz);
  }
  DPrintf("#%d: alloc(%lu) = %p\n", thr->tid, sz, p);
  SignalUnsafeCall(thr, pc);
  return p;
}

void user_free(ThreadState *thr, uptr pc, void *p) {
  CHECK_GT(thr->in_rtl, 0);
  CHECK_NE(p, (void*)0);
  DPrintf("#%d: free(%p)\n", thr->tid, p);
  MBlock *b = user_mblock(thr, p);
  p = b + 1;
  if (CTX() && CTX()->initialized && thr->in_rtl == 1) {
    MemoryRangeFreed(thr, pc, (uptr)p, b->size);
  }
  Free(b);
  SignalUnsafeCall(thr, pc);
}

void *user_realloc(ThreadState *thr, uptr pc, void *p, uptr sz) {
  CHECK_GT(thr->in_rtl, 0);
  void *p2 = 0;
  // FIXME: Handle "shrinking" more efficiently,
  // it seems that some software actually does this.
  if (sz) {
    p2 = user_alloc(thr, pc, sz);
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

void *user_alloc_aligned(ThreadState *thr, uptr pc, uptr sz, uptr align) {
  CHECK_GT(thr->in_rtl, 0);
  void *p = user_alloc(thr, pc, sz + align);
  void *pa = RoundUp(p, align);
  DCHECK_LE((uptr)pa + sz, (uptr)p + sz + align);
  return pa;
}

MBlock *user_mblock(ThreadState *thr, void *p) {
  CHECK_GT(thr->in_rtl, 0);
  CHECK_NE(p, (void*)0);
  MBlock *b = (MBlock*)AllocBlock(p);
  // FIXME: Output a warning, it's a user error.
  if (p < (char*)(b + 1) || p > (char*)(b + 1) + b->size) {
    Printf("user_mblock p=%p b=%p size=%lu beg=%p end=%p\n",
        p, b, b->size, (char*)(b + 1), (char*)(b + 1) + b->size);
    CHECK_GE(p, (char*)(b + 1));
    CHECK_LE(p, (char*)(b + 1) + b->size);
  }
  return b;
}

#if TSAN_DEBUG
struct InternalMBlock {
  static u32 const kMagic = 0xBCEBC041;
  u32 magic;
  u32 typ;
  u64 sz;
};
#endif

void *internal_alloc(MBlockType typ, uptr sz) {
  ThreadState *thr = cur_thread();
  CHECK_GT(thr->in_rtl, 0);
#if TSAN_DEBUG
  InternalMBlock *b = (InternalMBlock*)Alloc(sizeof(InternalMBlock) + sz);
  b->magic = InternalMBlock::kMagic;
  b->typ = typ;
  b->sz = sz;
  thr->int_alloc_cnt[typ] += 1;
  thr->int_alloc_siz[typ] += sz;
  void *p = b + 1;
  return p;
#else
  return Alloc(sz);
#endif
}

void internal_free(void *p) {
  ThreadState *thr = cur_thread();
  CHECK_GT(thr->in_rtl, 0);
#if TSAN_DEBUG
  InternalMBlock *b = (InternalMBlock*)p - 1;
  CHECK_EQ(b->magic, InternalMBlock::kMagic);
  thr->int_alloc_cnt[b->typ] -= 1;
  thr->int_alloc_siz[b->typ] -= b->sz;
  Free(b);
#else
  Free(p);
#endif
}

}  // namespace __tsan
