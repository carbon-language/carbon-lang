//===-- tsan_stack_trace.h --------------------------------------*- C++ -*-===//
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
#ifndef TSAN_STACK_TRACE_H
#define TSAN_STACK_TRACE_H

//#include "sanitizer_common/sanitizer_atomic.h"
//#include "sanitizer_common/sanitizer_common.h"
//#include "sanitizer_common/sanitizer_deadlock_detector_interface.h"
#include "tsan_defs.h"
//#include "tsan_clock.h"
//#include "tsan_mutex.h"
//#include "tsan_dense_alloc.h"

namespace __tsan {

class StackTrace {
 public:
  StackTrace();
  // Initialized the object in "static mode",
  // in this mode it never calls malloc/free but uses the provided buffer.
  StackTrace(uptr *buf, uptr cnt);
  ~StackTrace();
  void Reset();

  void Init(const uptr *pcs, uptr cnt);
  void ObtainCurrent(ThreadState *thr, uptr toppc);
  bool IsEmpty() const;
  uptr Size() const;
  uptr Get(uptr i) const;
  const uptr *Begin() const;
  void CopyFrom(const StackTrace& other);

 private:
  uptr n_;
  uptr *s_;
  const uptr c_;

  StackTrace(const StackTrace&);
  void operator = (const StackTrace&);
};

}  // namespace __tsan

#endif  // TSAN_STACK_TRACE_H
