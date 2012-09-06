//===-- tsan_sync.cc ------------------------------------------------------===//
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
#include "tsan_sync.h"
#include "tsan_rtl.h"
#include "tsan_mman.h"

namespace __tsan {

SyncVar::SyncVar(uptr addr)
  : mtx(MutexTypeSyncVar, StatMtxSyncVar)
  , addr(addr)
  , owner_tid(kInvalidTid)
  , last_lock()
  , recursion()
  , is_rw()
  , is_recursive()
  , is_broken()
  , is_linker_init() {
}

SyncTab::Part::Part()
  : mtx(MutexTypeSyncTab, StatMtxSyncTab)
  , val() {
}

SyncTab::SyncTab() {
}

SyncTab::~SyncTab() {
  for (int i = 0; i < kPartCount; i++) {
    while (tab_[i].val) {
      SyncVar *tmp = tab_[i].val;
      tab_[i].val = tmp->next;
      DestroyAndFree(tmp);
    }
  }
}

SyncVar* SyncTab::GetAndLock(ThreadState *thr, uptr pc,
                             uptr addr, bool write_lock) {
#ifndef TSAN_GO
  if (PrimaryAllocator::PointerIsMine((void*)addr)) {
    MBlock *b = user_mblock(thr, (void*)addr);
    Lock l(&b->mtx);
    SyncVar *res = 0;
    for (res = b->head; res; res = res->next) {
      if (res->addr == addr)
        break;
    }
    if (res == 0) {
      StatInc(thr, StatSyncCreated);
      void *mem = internal_alloc(MBlockSync, sizeof(SyncVar));
      res = new(mem) SyncVar(addr);
      res->creation_stack.ObtainCurrent(thr, pc);
      res->next = b->head;
      b->head = res;
    }
    if (write_lock)
      res->mtx.Lock();
    else
      res->mtx.ReadLock();
    return res;
  }
#endif

  Part *p = &tab_[PartIdx(addr)];
  {
    ReadLock l(&p->mtx);
    for (SyncVar *res = p->val; res; res = res->next) {
      if (res->addr == addr) {
        if (write_lock)
          res->mtx.Lock();
        else
          res->mtx.ReadLock();
        return res;
      }
    }
  }
  {
    Lock l(&p->mtx);
    SyncVar *res = p->val;
    for (; res; res = res->next) {
      if (res->addr == addr)
        break;
    }
    if (res == 0) {
      StatInc(thr, StatSyncCreated);
      void *mem = internal_alloc(MBlockSync, sizeof(SyncVar));
      res = new(mem) SyncVar(addr);
#ifndef TSAN_GO
      res->creation_stack.ObtainCurrent(thr, pc);
#endif
      res->next = p->val;
      p->val = res;
    }
    if (write_lock)
      res->mtx.Lock();
    else
      res->mtx.ReadLock();
    return res;
  }
}

SyncVar* SyncTab::GetAndRemove(ThreadState *thr, uptr pc, uptr addr) {
#ifndef TSAN_GO
  if (PrimaryAllocator::PointerIsMine((void*)addr)) {
    MBlock *b = user_mblock(thr, (void*)addr);
    SyncVar *res = 0;
    {
      Lock l(&b->mtx);
      SyncVar **prev = &b->head;
      res = *prev;
      while (res) {
        if (res->addr == addr) {
          if (res->is_linker_init)
            return 0;
          *prev = res->next;
          break;
        }
        prev = &res->next;
        res = *prev;
      }
    }
    if (res) {
      StatInc(thr, StatSyncDestroyed);
      res->mtx.Lock();
      res->mtx.Unlock();
    }
    return res;
  }
#endif

  Part *p = &tab_[PartIdx(addr)];
  SyncVar *res = 0;
  {
    Lock l(&p->mtx);
    SyncVar **prev = &p->val;
    res = *prev;
    while (res) {
      if (res->addr == addr) {
        if (res->is_linker_init)
          return 0;
        *prev = res->next;
        break;
      }
      prev = &res->next;
      res = *prev;
    }
  }
  if (res) {
    StatInc(thr, StatSyncDestroyed);
    res->mtx.Lock();
    res->mtx.Unlock();
  }
  return res;
}

uptr SyncVar::GetMemoryConsumption() {
  return sizeof(*this)
      + clock.size() * sizeof(u64)
      + read_clock.size() * sizeof(u64)
      + creation_stack.Size() * sizeof(uptr);
}

uptr SyncTab::GetMemoryConsumption(uptr *nsync) {
  uptr mem = 0;
  for (int i = 0; i < kPartCount; i++) {
    Part *p = &tab_[i];
    Lock l(&p->mtx);
    for (SyncVar *s = p->val; s; s = s->next) {
      *nsync += 1;
      mem += s->GetMemoryConsumption();
    }
  }
  return mem;
}

int SyncTab::PartIdx(uptr addr) {
  return (addr >> 3) % kPartCount;
}

StackTrace::StackTrace()
    : n_()
    , s_()
    , c_() {
}

StackTrace::StackTrace(uptr *buf, uptr cnt)
    : n_()
    , s_(buf)
    , c_(cnt) {
  CHECK_NE(buf, 0);
  CHECK_NE(cnt, 0);
}

StackTrace::~StackTrace() {
  Reset();
}

void StackTrace::Reset() {
  if (s_ && !c_) {
    CHECK_NE(n_, 0);
    internal_free(s_);
    s_ = 0;
  }
  n_ = 0;
}

void StackTrace::Init(const uptr *pcs, uptr cnt) {
  Reset();
  if (cnt == 0)
    return;
  if (c_) {
    CHECK_NE(s_, 0);
    CHECK_LE(cnt, c_);
  } else {
    s_ = (uptr*)internal_alloc(MBlockStackTrace, cnt * sizeof(s_[0]));
  }
  n_ = cnt;
  internal_memcpy(s_, pcs, cnt * sizeof(s_[0]));
}

void StackTrace::ObtainCurrent(ThreadState *thr, uptr toppc) {
  Reset();
  n_ = thr->shadow_stack_pos - thr->shadow_stack;
  if (n_ + !!toppc == 0)
    return;
  uptr start = 0;
  if (c_) {
    CHECK_NE(s_, 0);
    if (n_ + !!toppc > c_) {
      start = n_ - c_ + !!toppc;
      n_ = c_ - !!toppc;
    }
  } else {
    s_ = (uptr*)internal_alloc(MBlockStackTrace,
                               (n_ + !!toppc) * sizeof(s_[0]));
  }
  for (uptr i = 0; i < n_; i++)
    s_[i] = thr->shadow_stack[start + i];
  if (toppc) {
    s_[n_] = toppc;
    n_++;
  }
}

void StackTrace::CopyFrom(const StackTrace& other) {
  Reset();
  Init(other.Begin(), other.Size());
}

bool StackTrace::IsEmpty() const {
  return n_ == 0;
}

uptr StackTrace::Size() const {
  return n_;
}

uptr StackTrace::Get(uptr i) const {
  CHECK_LT(i, n_);
  return s_[i];
}

const uptr *StackTrace::Begin() const {
  return s_;
}

}  // namespace __tsan
