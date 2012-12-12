//===-- tsan_fd.cc --------------------------------------------------------===//
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

#include "tsan_fd.h"
#include "tsan_rtl.h"
#include <sanitizer_common/sanitizer_atomic.h>

namespace __tsan {

const int kTableSizeL1 = 1024;
const int kTableSizeL2 = 1024;
const int kTableSize = kTableSizeL1 * kTableSizeL2;

struct FdDesc {
  atomic_uint64_t rc;
};

struct FdContext {
  atomic_uintptr_t tab[kTableSizeL1];
  // Addresses used for synchronization.
  FdDesc globdesc;
  FdDesc filedesc;
  FdDesc sockdesc;
};

static FdContext fdctx;

static FdDesc *allocdesc() {
  FdDesc *pd = (FdDesc*)internal_alloc(MBlockFD, sizeof(FdDesc));
  atomic_store(&pd->rc, 1, memory_order_relaxed);
  return pd;
}

static FdDesc *ref(FdDesc *pd) {
  if (pd && atomic_load(&pd->rc, memory_order_relaxed) != (u64)-1)
    atomic_fetch_add(&pd->rc, 1, memory_order_relaxed);
  return pd;
}

static void unref(ThreadState *thr, uptr pc, FdDesc *pd) {
  if (pd && atomic_load(&pd->rc, memory_order_relaxed) != (u64)-1) {
    if (atomic_fetch_sub(&pd->rc, 1, memory_order_acq_rel) == 1) {
      CHECK_NE(pd, &fdctx.globdesc);
      CHECK_NE(pd, &fdctx.filedesc);
      CHECK_NE(pd, &fdctx.sockdesc);
      SyncVar *s = CTX()->synctab.GetAndRemove(thr, pc, (uptr)pd);
      if (s)
        DestroyAndFree(s);
      internal_free(pd);
    }
  }
}

static FdDesc **fdaddr(ThreadState *thr, uptr pc, int fd) {
  CHECK_LT(fd, kTableSize);
  atomic_uintptr_t *pl1 = &fdctx.tab[fd / kTableSizeL2];
  uptr l1 = atomic_load(pl1, memory_order_consume);
  if (l1 == 0) {
    uptr size = kTableSizeL2 * sizeof(uptr);
    void *p = internal_alloc(MBlockFD, size);
    internal_memset(p, 0, size);
    MemoryResetRange(thr, (uptr)&fdaddr, (uptr)p, size);
    if (atomic_compare_exchange_strong(pl1, &l1, (uptr)p, memory_order_acq_rel))
      l1 = (uptr)p;
    else
      internal_free(p);
  }
  return &((FdDesc**)l1)[fd % kTableSizeL2];  // NOLINT
}

// pd must be already ref'ed.
static void init(ThreadState *thr, uptr pc, int fd, FdDesc *d) {
  FdDesc **pd = fdaddr(thr, pc, fd);
  // As a matter of fact, we don't intercept all close calls.
  // See e.g. libc __res_iclose().
  if (*pd)
    unref(thr, pc, *pd);
  *pd = d;
  // To catch races between fd usage and open.
  MemoryRangeImitateWrite(thr, pc, (uptr)pd, 8);
}

void FdInit() {
  atomic_store(&fdctx.globdesc.rc, (u64)-1, memory_order_relaxed);
  atomic_store(&fdctx.filedesc.rc, (u64)-1, memory_order_relaxed);
  atomic_store(&fdctx.sockdesc.rc, (u64)-1, memory_order_relaxed);
}

void FdAcquire(ThreadState *thr, uptr pc, int fd) {
  FdDesc **pd = fdaddr(thr, pc, fd);
  FdDesc *d = *pd;
  DPrintf("#%d: FdAcquire(%d) -> %p\n", thr->tid, fd, d);
  MemoryRead8Byte(thr, pc, (uptr)pd);
  if (d)
    Acquire(thr, pc, (uptr)d);
}

void FdRelease(ThreadState *thr, uptr pc, int fd) {
  FdDesc **pd = fdaddr(thr, pc, fd);
  FdDesc *d = *pd;
  DPrintf("#%d: FdRelease(%d) -> %p\n", thr->tid, fd, d);
  if (d)
    Release(thr, pc, (uptr)d);
  MemoryRead8Byte(thr, pc, (uptr)pd);
}

void FdClose(ThreadState *thr, uptr pc, int fd) {
  DPrintf("#%d: FdClose(%d)\n", thr->tid, fd);
  FdDesc **pd = fdaddr(thr, pc, fd);
  // To catch races between fd usage and close.
  MemoryWrite8Byte(thr, pc, (uptr)pd);
  // We need to clear it, because if we do not intercept any call out there
  // that creates fd, we will hit false postives.
  MemoryResetRange(thr, pc, (uptr)pd, 8);
  unref(thr, pc, *pd);
  *pd = 0;
}

void FdFileCreate(ThreadState *thr, uptr pc, int fd) {
  DPrintf("#%d: FdFileCreate(%d)\n", thr->tid, fd);
  init(thr, pc, fd, &fdctx.filedesc);
}

void FdDup(ThreadState *thr, uptr pc, int oldfd, int newfd) {
  DPrintf("#%d: FdDup(%d, %d)\n", thr->tid, oldfd, newfd);
  // Ignore the case when user dups not yet connected socket.
  FdDesc **opd = fdaddr(thr, pc, oldfd);
  MemoryRead8Byte(thr, pc, (uptr)opd);
  FdClose(thr, pc, newfd);
  init(thr, pc, newfd, ref(*opd));
}

void FdPipeCreate(ThreadState *thr, uptr pc, int rfd, int wfd) {
  DPrintf("#%d: FdCreatePipe(%d, %d)\n", thr->tid, rfd, wfd);
  FdDesc *d = allocdesc();
  init(thr, pc, rfd, d);
  init(thr, pc, wfd, ref(d));
}

void FdEventCreate(ThreadState *thr, uptr pc, int fd) {
  DPrintf("#%d: FdEventCreate(%d)\n", thr->tid, fd);
  init(thr, pc, fd, allocdesc());
}

void FdPollCreate(ThreadState *thr, uptr pc, int fd) {
  DPrintf("#%d: FdPollCreate(%d)\n", thr->tid, fd);
  init(thr, pc, fd, allocdesc());
}

void FdSocketCreate(ThreadState *thr, uptr pc, int fd) {
  DPrintf("#%d: FdSocketCreate(%d)\n", thr->tid, fd);
  // It can be a UDP socket.
  init(thr, pc, fd, &fdctx.sockdesc);
}

void FdSocketAccept(ThreadState *thr, uptr pc, int fd, int newfd) {
  DPrintf("#%d: FdSocketAccept(%d, %d)\n", thr->tid, fd, newfd);
  init(thr, pc, newfd, &fdctx.sockdesc);
}

void FdSocketConnect(ThreadState *thr, uptr pc, int fd) {
  DPrintf("#%d: FdSocketConnect(%d)\n", thr->tid, fd);
  init(thr, pc, fd, &fdctx.sockdesc);
}

uptr File2addr(char *path) {
  (void)path;
  static u64 addr;
  return (uptr)&addr;
}

uptr Dir2addr(char *path) {
  (void)path;
  static u64 addr;
  return (uptr)&addr;
}

}  //  namespace __tsan
