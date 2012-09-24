//===-- tsan_interceptors.h -------------------------------------*- C++ -*-===//
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

#ifndef TSAN_INTERCEPTORS_H
#define TSAN_INTERCEPTORS_H

#include "interception/interception.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "tsan_rtl.h"

namespace __tsan {

class ScopedInterceptor {
 public:
  ScopedInterceptor(ThreadState *thr, const char *fname, uptr pc);
  ~ScopedInterceptor();
 private:
  ThreadState *const thr_;
  const int in_rtl_;
};

#define SCOPED_INTERCEPTOR_RAW(func, ...) \
    ThreadState *thr = cur_thread(); \
    StatInc(thr, StatInterceptor); \
    StatInc(thr, StatInt_##func); \
    const uptr caller_pc = GET_CALLER_PC(); \
    ScopedInterceptor si(thr, #func, caller_pc); \
    /* Subtract one from pc as we need current instruction address */ \
    const uptr pc = __sanitizer::StackTrace::GetCurrentPc() - 1; \
    (void)pc; \
/**/

#define SCOPED_TSAN_INTERCEPTOR(func, ...) \
    SCOPED_INTERCEPTOR_RAW(func, __VA_ARGS__); \
    if (thr->in_rtl > 1) \
      return REAL(func)(__VA_ARGS__); \
/**/

#define TSAN_INTERCEPTOR(ret, func, ...) INTERCEPTOR(ret, func, __VA_ARGS__)
#define TSAN_INTERCEPT(func) INTERCEPT_FUNCTION(func)

}  // namespace __tsan

#endif  // TSAN_INTERCEPTORS_H
