//===-- asan_interceptors_memintrinsics.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// ASan versions of memcpy, memmove, and memset.
//===---------------------------------------------------------------------===//

#include "asan_interceptors_memintrinsics.h"
#include "asan_report.h"
#include "asan_stack.h"
#include "asan_suppressions.h"

using namespace __asan;

void *__asan_memcpy(void *to, const void *from, uptr size) {
  ASAN_MEMCPY_IMPL(nullptr, to, from, size);
}

void *__asan_memset(void *block, int c, uptr size) {
  ASAN_MEMSET_IMPL(nullptr, block, c, size);
}

void *__asan_memmove(void *to, const void *from, uptr size) {
  ASAN_MEMMOVE_IMPL(nullptr, to, from, size);
}

#if SANITIZER_FUCHSIA

// Fuchsia doesn't use sanitizer_common_interceptors.inc, but
// the only things there it wants are these three.  Just define them
// as aliases here rather than repeating the contents.

extern "C" decltype(__asan_memcpy) memcpy[[gnu::alias("__asan_memcpy")]];
extern "C" decltype(__asan_memmove) memmove[[gnu::alias("__asan_memmove")]];
extern "C" decltype(__asan_memset) memset[[gnu::alias("__asan_memset")]];

#endif  // SANITIZER_FUCHSIA
