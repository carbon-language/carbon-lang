//===-- asan_interceptors.cc ----------------------------------------------===//
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
// Interceptors for operators new and delete.
//===----------------------------------------------------------------------===//

#include "asan_allocator.h"
#include "asan_internal.h"
#include "asan_stack.h"

#include <stddef.h>

namespace __asan {
// This function is a no-op. We need it to make sure that object file
// with our replacements will actually be loaded from static ASan
// run-time library at link-time.
void ReplaceOperatorsNewAndDelete() { }
}

using namespace __asan;  // NOLINT

// On Android new() goes through malloc interceptors.
// See also https://code.google.com/p/address-sanitizer/issues/detail?id=131.
#if !ASAN_ANDROID

// Fake std::nothrow_t to avoid including <new>.
namespace std {
struct nothrow_t {};
}  // namespace std

#define OPERATOR_NEW_BODY(type) \
  GET_STACK_TRACE_MALLOC;\
  return asan_memalign(0, size, &stack, type);

// On OS X it's not enough to just provide our own 'operator new' and
// 'operator delete' implementations, because they're going to be in the
// runtime dylib, and the main executable will depend on both the runtime
// dylib and libstdc++, each of those'll have its implementation of new and
// delete.
// To make sure that C++ allocation/deallocation operators are overridden on
// OS X we need to intercept them using their mangled names.
#if !defined(__APPLE__)
INTERCEPTOR_ATTRIBUTE
void *operator new(size_t size) { OPERATOR_NEW_BODY(FROM_NEW); }
INTERCEPTOR_ATTRIBUTE
void *operator new[](size_t size) { OPERATOR_NEW_BODY(FROM_NEW_BR); }
INTERCEPTOR_ATTRIBUTE
void *operator new(size_t size, std::nothrow_t const&)
{ OPERATOR_NEW_BODY(FROM_NEW); }
INTERCEPTOR_ATTRIBUTE
void *operator new[](size_t size, std::nothrow_t const&)
{ OPERATOR_NEW_BODY(FROM_NEW_BR); }

#else  // __APPLE__
INTERCEPTOR(void *, _Znwm, size_t size) {
  OPERATOR_NEW_BODY(FROM_NEW);
}
INTERCEPTOR(void *, _Znam, size_t size) {
  OPERATOR_NEW_BODY(FROM_NEW_BR);
}
INTERCEPTOR(void *, _ZnwmRKSt9nothrow_t, size_t size, std::nothrow_t const&) {
  OPERATOR_NEW_BODY(FROM_NEW);
}
INTERCEPTOR(void *, _ZnamRKSt9nothrow_t, size_t size, std::nothrow_t const&) {
  OPERATOR_NEW_BODY(FROM_NEW_BR);
}
#endif

#define OPERATOR_DELETE_BODY(type) \
  GET_STACK_TRACE_FREE;\
  asan_free(ptr, &stack, type);

#if !defined(__APPLE__)
INTERCEPTOR_ATTRIBUTE
void operator delete(void *ptr) { OPERATOR_DELETE_BODY(FROM_NEW); }
INTERCEPTOR_ATTRIBUTE
void operator delete[](void *ptr) { OPERATOR_DELETE_BODY(FROM_NEW_BR); }
INTERCEPTOR_ATTRIBUTE
void operator delete(void *ptr, std::nothrow_t const&)
{ OPERATOR_DELETE_BODY(FROM_NEW); }
INTERCEPTOR_ATTRIBUTE
void operator delete[](void *ptr, std::nothrow_t const&)
{ OPERATOR_DELETE_BODY(FROM_NEW_BR); }

#else  // __APPLE__
INTERCEPTOR(void, _ZdlPv, void *ptr) {
  OPERATOR_DELETE_BODY(FROM_NEW);
}
INTERCEPTOR(void, _ZdaPv, void *ptr) {
  OPERATOR_DELETE_BODY(FROM_NEW_BR);
}
INTERCEPTOR(void, _ZdlPvRKSt9nothrow_t, void *ptr, std::nothrow_t const&) {
  OPERATOR_DELETE_BODY(FROM_NEW);
}
INTERCEPTOR(void, _ZdaPvRKSt9nothrow_t, void *ptr, std::nothrow_t const&) {
  OPERATOR_DELETE_BODY(FROM_NEW_BR);
}
#endif

#endif
