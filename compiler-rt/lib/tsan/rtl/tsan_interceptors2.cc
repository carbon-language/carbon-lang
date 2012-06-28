//===-- tsan_interceptors2.cc ---------------------------------------------===//
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

#include "interception/interception.h"
#include "sanitizer_common/sanitizer_common.h"

namespace __tsan {
  void *intercept_memset(uptr, void*, int, uptr);
  void *intercept_memcpy(uptr, void*, const void*, uptr);
  int intercept_memcmp(uptr, const void*, const void*, uptr);
}

using namespace __tsan;  // NOLINT

INTERCEPTOR(void*, memset, void *dst, int v, uptr size) {
  return intercept_memset((uptr)__builtin_return_address(0), dst, v, size);
}

INTERCEPTOR(void*, memcpy, void *dst, const void *src, uptr size) {
  return intercept_memcpy((uptr)__builtin_return_address(0), dst, src, size);
}

INTERCEPTOR(int, memcmp, const void *s1, const void *s2, uptr n) {
  return intercept_memcmp((uptr)__builtin_return_address(0), s1, s2, n);
}
