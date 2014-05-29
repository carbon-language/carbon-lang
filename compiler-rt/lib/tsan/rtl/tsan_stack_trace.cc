//===-- tsan_stack_trace.cc -----------------------------------------------===//
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
//#include "sanitizer_common/sanitizer_placement_new.h"
#include "tsan_stack_trace.h"
#include "tsan_rtl.h"
#include "tsan_mman.h"

namespace __tsan {

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
