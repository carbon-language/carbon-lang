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
#include <new>

namespace __asan {
// This function is a no-op. We need it to make sure that object file
// with our replacements will actually be loaded from static ASan
// run-time library at link-time.
void ReplaceOperatorsNewAndDelete() { }
}

using namespace __asan;  // NOLINT

// On Android new() goes through malloc interceptors.
#if !ASAN_ANDROID

#define OPERATOR_NEW_BODY \
  GET_STACK_TRACE_HERE_FOR_MALLOC;\
  return asan_memalign(0, size, &stack);

INTERCEPTOR_ATTRIBUTE
void *operator new(size_t size) throw(std::bad_alloc) { OPERATOR_NEW_BODY; }
INTERCEPTOR_ATTRIBUTE
void *operator new[](size_t size) throw(std::bad_alloc) { OPERATOR_NEW_BODY; }
INTERCEPTOR_ATTRIBUTE
void *operator new(size_t size, std::nothrow_t const&) throw()
{ OPERATOR_NEW_BODY; }
INTERCEPTOR_ATTRIBUTE
void *operator new[](size_t size, std::nothrow_t const&) throw()
{ OPERATOR_NEW_BODY; }

#define OPERATOR_DELETE_BODY \
  GET_STACK_TRACE_HERE_FOR_FREE(ptr);\
  asan_free(ptr, &stack);

INTERCEPTOR_ATTRIBUTE
void operator delete(void *ptr) throw() { OPERATOR_DELETE_BODY; }
INTERCEPTOR_ATTRIBUTE
void operator delete[](void *ptr) throw() { OPERATOR_DELETE_BODY; }
INTERCEPTOR_ATTRIBUTE
void operator delete(void *ptr, std::nothrow_t const&) throw()
{ OPERATOR_DELETE_BODY; }
INTERCEPTOR_ATTRIBUTE
void operator delete[](void *ptr, std::nothrow_t const&) throw()
{ OPERATOR_DELETE_BODY; }

#endif
