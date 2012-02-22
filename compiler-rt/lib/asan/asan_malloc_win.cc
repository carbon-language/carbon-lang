//===-- asan_malloc_win.cc --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// Windows-specific malloc interception.
//===----------------------------------------------------------------------===//
#ifdef _WIN32

#include "asan_allocator.h"
#include "asan_interceptors.h"
#include "asan_internal.h"
#include "asan_stack.h"

namespace __asan {
void ReplaceSystemMalloc() {
  // FIXME: investigate whether any action is needed.
  // Currenlty, we fail to intercept malloc() called from intercepted _strdup().
}
}  // namespace __asan

// ---------------------- Replacement functions ---------------- {{{1
using namespace __asan;  // NOLINT

// FIXME: Simply defining functions with the same signature in *.obj
// files overrides the standard functions in *.lib
// This works well for simple helloworld-like tests but might need to be
// revisited in the future.

extern "C" {
void free(void *ptr) {
  GET_STACK_TRACE_HERE_FOR_FREE(ptr);
  return asan_free(ptr, &stack);
}

void *malloc(size_t size) {
  GET_STACK_TRACE_HERE_FOR_MALLOC;
  return asan_malloc(size, &stack);
}

void *calloc(size_t nmemb, size_t size) {
  GET_STACK_TRACE_HERE_FOR_MALLOC;
  return asan_calloc(nmemb, size, &stack);
}

void *realloc(void *ptr, size_t size) {
  GET_STACK_TRACE_HERE_FOR_MALLOC;
  return asan_realloc(ptr, size, &stack);
}
}  // extern "C"

#endif  // _WIN32
