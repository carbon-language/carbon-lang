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

SyncVar::SyncVar(uptr addr, u64 uid)
  : mtx(MutexTypeSyncVar, StatMtxSyncVar)
  , addr(addr)
  , uid(uid)
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

SyncVar* SyncTab::GetOrCreateAndLock(ThreadState *thr, uptr pc,
                                     uptr addr, bool write_lock) {
  return GetAndLock(thr, pc, addr, write_lock, true);
}

SyncVar* SyncTab::GetIfExistsAndLock(uptr addr, bool write_lock) {
  return GetAndLock(0, 0, addr, write_lock, false);
}

SyncVar* SyncTab::Create(ThreadState *thr, uptr pc, uptr addr) {
  Context *ctx = CTX();
  StatInc(thr, StatSyncCreated);
  void *mem = internal_alloc(MBlockSync, sizeof(SyncVar));
  const u64 uid = atomic_fetch_add(&uid_gen_, 1, memory_order_relaxed);
  SyncVar *res = new(mem) SyncVar(addr, uid);
#ifndef TSAN_GO
  res->creation_stack_id = CurrentStackId(thr, pc);
#endif
  if (flags()->detect_deadlocks)
    ctx->dd->MutexInit(&res->dd, res->creation_stack_id, res->GetId());
  return res;
}

SyncVar* SyncTab::GetAndLock(ThreadState *thr, uptr pc,
                             uptr addr, bool write_lock, bool create) {
#ifndef TSAN_GO
  {  // NOLINT
    SyncVar *res = GetJavaSync(thr, pc, addr, write_lock, create);
    if (res)
      return res;
  }

  // Here we ask only PrimaryAllocator, because
  // SecondaryAllocator::PointerIsMine() is slow and we have fallback on
  // the hashmap anyway.
  if (PrimaryAllocator::PointerIsMine((void*)addr)) {
    MBlock *b = user_mblock(thr, (void*)addr);
    CHECK_NE(b, 0);
    MBlock::ScopedLock l(b);
    SyncVar *res = 0;
    for (res = b->ListHead(); res; res = res->next) {
      if (res->addr == addr)
        break;
    }
    if (res == 0) {
      if (!create)
        return 0;
      res = Create(thr, pc, addr);
      b->ListPush(res);
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
  if (!create)
    return 0;
  {
    Lock l(&p->mtx);
    SyncVar *res = p->val;
    for (; res; res = res->next) {
      if (res->addr == addr)
        break;
    }
    if (res == 0) {
      res = Create(thr, pc, addr);
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
  {  // NOLINT
    SyncVar *res = GetAndRemoveJavaSync(thr, pc, addr);
    if (res)
      return res;
  }
  if (PrimaryAllocator::PointerIsMine((void*)addr)) {
    MBlock *b = user_mblock(thr, (void*)addr);
    CHECK_NE(b, 0);
    SyncVar *res = 0;
    {
      MBlock::ScopedLock l(b);
      res = b->ListHead();
      if (res) {
        if (res->addr == addr) {
          if (res->is_linker_init)
            return 0;
          b->ListPop();
        } else {
          SyncVar **prev = &res->next;
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
      }
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
    // Cap potentially huge stacks.
    if (n_ + !!toppc > kTraceStackSize) {
      start = n_ - kTraceStackSize + !!toppc;
      n_ = kTraceStackSize - !!toppc;
    }
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
