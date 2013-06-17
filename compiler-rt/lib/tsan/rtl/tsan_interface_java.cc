//===-- tsan_interface_java.cc --------------------------------------------===//
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

#include "tsan_interface_java.h"
#include "tsan_rtl.h"
#include "tsan_mutex.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_placement_new.h"
#include "sanitizer_common/sanitizer_stacktrace.h"

using namespace __tsan;  // NOLINT

namespace __tsan {

const uptr kHeapShadow = 0x300000000000ull;
const uptr kHeapAlignment = 8;

struct BlockDesc {
  bool begin;
  Mutex mtx;
  SyncVar *head;

  BlockDesc()
      : mtx(MutexTypeJavaMBlock, StatMtxJavaMBlock)
      , head() {
    CHECK_EQ(begin, false);
    begin = true;
  }

  ~BlockDesc() {
    CHECK_EQ(begin, true);
    begin = false;
    ThreadState *thr = cur_thread();
    SyncVar *s = head;
    while (s) {
      SyncVar *s1 = s->next;
      StatInc(thr, StatSyncDestroyed);
      s->mtx.Lock();
      s->mtx.Unlock();
      thr->mset.Remove(s->GetId());
      DestroyAndFree(s);
      s = s1;
    }
  }
};

struct JavaContext {
  const uptr heap_begin;
  const uptr heap_size;
  BlockDesc *heap_shadow;

  JavaContext(jptr heap_begin, jptr heap_size)
      : heap_begin(heap_begin)
      , heap_size(heap_size) {
    uptr size = heap_size / kHeapAlignment * sizeof(BlockDesc);
    heap_shadow = (BlockDesc*)MmapFixedNoReserve(kHeapShadow, size);
    if ((uptr)heap_shadow != kHeapShadow) {
      Printf("ThreadSanitizer: failed to mmap Java heap shadow\n");
      Die();
    }
  }
};

class ScopedJavaFunc {
 public:
  ScopedJavaFunc(ThreadState *thr, uptr pc)
      : thr_(thr) {
    Initialize(thr_);
    FuncEntry(thr, pc);
    CHECK_EQ(thr_->in_rtl, 0);
    thr_->in_rtl++;
  }

  ~ScopedJavaFunc() {
    thr_->in_rtl--;
    CHECK_EQ(thr_->in_rtl, 0);
    FuncExit(thr_);
    // FIXME(dvyukov): process pending signals.
  }

 private:
  ThreadState *thr_;
};

static u64 jctx_buf[sizeof(JavaContext) / sizeof(u64) + 1];
static JavaContext *jctx;

static BlockDesc *getblock(uptr addr) {
  uptr i = (addr - jctx->heap_begin) / kHeapAlignment;
  return &jctx->heap_shadow[i];
}

static uptr USED getmem(BlockDesc *b) {
  uptr i = b - jctx->heap_shadow;
  uptr p = jctx->heap_begin + i * kHeapAlignment;
  CHECK_GE(p, jctx->heap_begin);
  CHECK_LT(p, jctx->heap_begin + jctx->heap_size);
  return p;
}

static BlockDesc *getblockbegin(uptr addr) {
  for (BlockDesc *b = getblock(addr);; b--) {
    CHECK_GE(b, jctx->heap_shadow);
    if (b->begin)
      return b;
  }
  return 0;
}

SyncVar* GetJavaSync(ThreadState *thr, uptr pc, uptr addr,
                     bool write_lock, bool create) {
  if (jctx == 0 || addr < jctx->heap_begin
      || addr >= jctx->heap_begin + jctx->heap_size)
    return 0;
  BlockDesc *b = getblockbegin(addr);
  DPrintf("#%d: GetJavaSync %p->%p\n", thr->tid, addr, b);
  Lock l(&b->mtx);
  SyncVar *s = b->head;
  for (; s; s = s->next) {
    if (s->addr == addr) {
      DPrintf("#%d: found existing sync for %p\n", thr->tid, addr);
      break;
    }
  }
  if (s == 0 && create) {
    DPrintf("#%d: creating new sync for %p\n", thr->tid, addr);
    s = CTX()->synctab.Create(thr, pc, addr);
    s->next = b->head;
    b->head = s;
  }
  if (s) {
    if (write_lock)
      s->mtx.Lock();
    else
      s->mtx.ReadLock();
  }
  return s;
}

SyncVar* GetAndRemoveJavaSync(ThreadState *thr, uptr pc, uptr addr) {
  // We do not destroy Java mutexes other than in __tsan_java_free().
  return 0;
}

}  // namespace __tsan

#define SCOPED_JAVA_FUNC(func) \
  ThreadState *thr = cur_thread(); \
  const uptr caller_pc = GET_CALLER_PC(); \
  const uptr pc = __sanitizer::StackTrace::GetCurrentPc(); \
  (void)pc; \
  ScopedJavaFunc scoped(thr, caller_pc); \
/**/

void __tsan_java_init(jptr heap_begin, jptr heap_size) {
  SCOPED_JAVA_FUNC(__tsan_java_init);
  DPrintf("#%d: java_init(%p, %p)\n", thr->tid, heap_begin, heap_size);
  CHECK_EQ(jctx, 0);
  CHECK_GT(heap_begin, 0);
  CHECK_GT(heap_size, 0);
  CHECK_EQ(heap_begin % kHeapAlignment, 0);
  CHECK_EQ(heap_size % kHeapAlignment, 0);
  CHECK_LT(heap_begin, heap_begin + heap_size);
  jctx = new(jctx_buf) JavaContext(heap_begin, heap_size);
}

int  __tsan_java_fini() {
  SCOPED_JAVA_FUNC(__tsan_java_fini);
  DPrintf("#%d: java_fini()\n", thr->tid);
  CHECK_NE(jctx, 0);
  // FIXME(dvyukov): this does not call atexit() callbacks.
  int status = Finalize(thr);
  DPrintf("#%d: java_fini() = %d\n", thr->tid, status);
  return status;
}

void __tsan_java_alloc(jptr ptr, jptr size) {
  SCOPED_JAVA_FUNC(__tsan_java_alloc);
  DPrintf("#%d: java_alloc(%p, %p)\n", thr->tid, ptr, size);
  CHECK_NE(jctx, 0);
  CHECK_NE(size, 0);
  CHECK_EQ(ptr % kHeapAlignment, 0);
  CHECK_EQ(size % kHeapAlignment, 0);
  CHECK_GE(ptr, jctx->heap_begin);
  CHECK_LE(ptr + size, jctx->heap_begin + jctx->heap_size);

  BlockDesc *b = getblock(ptr);
  new(b) BlockDesc();
}

void __tsan_java_free(jptr ptr, jptr size) {
  SCOPED_JAVA_FUNC(__tsan_java_free);
  DPrintf("#%d: java_free(%p, %p)\n", thr->tid, ptr, size);
  CHECK_NE(jctx, 0);
  CHECK_NE(size, 0);
  CHECK_EQ(ptr % kHeapAlignment, 0);
  CHECK_EQ(size % kHeapAlignment, 0);
  CHECK_GE(ptr, jctx->heap_begin);
  CHECK_LE(ptr + size, jctx->heap_begin + jctx->heap_size);

  BlockDesc *beg = getblock(ptr);
  BlockDesc *end = getblock(ptr + size);
  for (BlockDesc *b = beg; b != end; b++) {
    if (b->begin)
      b->~BlockDesc();
  }
}

void __tsan_java_move(jptr src, jptr dst, jptr size) {
  SCOPED_JAVA_FUNC(__tsan_java_move);
  DPrintf("#%d: java_move(%p, %p, %p)\n", thr->tid, src, dst, size);
  CHECK_NE(jctx, 0);
  CHECK_NE(size, 0);
  CHECK_EQ(src % kHeapAlignment, 0);
  CHECK_EQ(dst % kHeapAlignment, 0);
  CHECK_EQ(size % kHeapAlignment, 0);
  CHECK_GE(src, jctx->heap_begin);
  CHECK_LE(src + size, jctx->heap_begin + jctx->heap_size);
  CHECK_GE(dst, jctx->heap_begin);
  CHECK_LE(dst + size, jctx->heap_begin + jctx->heap_size);
  CHECK(dst >= src + size || src >= dst + size);

  // Assuming it's not running concurrently with threads that do
  // memory accesses and mutex operations (stop-the-world phase).
  {  // NOLINT
    BlockDesc *s = getblock(src);
    BlockDesc *d = getblock(dst);
    BlockDesc *send = getblock(src + size);
    for (; s != send; s++, d++) {
      CHECK_EQ(d->begin, false);
      if (s->begin) {
        DPrintf("#%d: moving block %p->%p\n", thr->tid, getmem(s), getmem(d));
        new(d) BlockDesc;
        d->head = s->head;
        for (SyncVar *sync = d->head; sync; sync = sync->next) {
          uptr newaddr = sync->addr - src + dst;
          DPrintf("#%d: moving sync %p->%p\n", thr->tid, sync->addr, newaddr);
          sync->addr = newaddr;
        }
        s->head = 0;
        s->~BlockDesc();
      }
    }
  }

  {  // NOLINT
    u64 *s = (u64*)MemToShadow(src);
    u64 *d = (u64*)MemToShadow(dst);
    u64 *send = (u64*)MemToShadow(src + size);
    for (; s != send; s++, d++) {
      *d = *s;
      *s = 0;
    }
  }
}

void __tsan_java_mutex_lock(jptr addr) {
  SCOPED_JAVA_FUNC(__tsan_java_mutex_lock);
  DPrintf("#%d: java_mutex_lock(%p)\n", thr->tid, addr);
  CHECK_NE(jctx, 0);
  CHECK_GE(addr, jctx->heap_begin);
  CHECK_LT(addr, jctx->heap_begin + jctx->heap_size);

  MutexCreate(thr, pc, addr, true, true, true);
  MutexLock(thr, pc, addr);
}

void __tsan_java_mutex_unlock(jptr addr) {
  SCOPED_JAVA_FUNC(__tsan_java_mutex_unlock);
  DPrintf("#%d: java_mutex_unlock(%p)\n", thr->tid, addr);
  CHECK_NE(jctx, 0);
  CHECK_GE(addr, jctx->heap_begin);
  CHECK_LT(addr, jctx->heap_begin + jctx->heap_size);

  MutexUnlock(thr, pc, addr);
}

void __tsan_java_mutex_read_lock(jptr addr) {
  SCOPED_JAVA_FUNC(__tsan_java_mutex_read_lock);
  DPrintf("#%d: java_mutex_read_lock(%p)\n", thr->tid, addr);
  CHECK_NE(jctx, 0);
  CHECK_GE(addr, jctx->heap_begin);
  CHECK_LT(addr, jctx->heap_begin + jctx->heap_size);

  MutexCreate(thr, pc, addr, true, true, true);
  MutexReadLock(thr, pc, addr);
}

void __tsan_java_mutex_read_unlock(jptr addr) {
  SCOPED_JAVA_FUNC(__tsan_java_mutex_read_unlock);
  DPrintf("#%d: java_mutex_read_unlock(%p)\n", thr->tid, addr);
  CHECK_NE(jctx, 0);
  CHECK_GE(addr, jctx->heap_begin);
  CHECK_LT(addr, jctx->heap_begin + jctx->heap_size);

  MutexReadUnlock(thr, pc, addr);
}

void __tsan_java_mutex_lock_rec(jptr addr, int rec) {
  SCOPED_JAVA_FUNC(__tsan_java_mutex_lock_rec);
  DPrintf("#%d: java_mutex_lock_rec(%p, %d)\n", thr->tid, addr, rec);
  CHECK_NE(jctx, 0);
  CHECK_GE(addr, jctx->heap_begin);
  CHECK_LT(addr, jctx->heap_begin + jctx->heap_size);
  CHECK_GT(rec, 0);

  MutexCreate(thr, pc, addr, true, true, true);
  MutexLock(thr, pc, addr, rec);
}

int __tsan_java_mutex_unlock_rec(jptr addr) {
  SCOPED_JAVA_FUNC(__tsan_java_mutex_unlock_rec);
  DPrintf("#%d: java_mutex_unlock_rec(%p)\n", thr->tid, addr);
  CHECK_NE(jctx, 0);
  CHECK_GE(addr, jctx->heap_begin);
  CHECK_LT(addr, jctx->heap_begin + jctx->heap_size);

  return MutexUnlock(thr, pc, addr, true);
}
