//===-- msan_new_delete.cc ------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of MemorySanitizer.
//
// Interceptors for operators new and delete.
//===----------------------------------------------------------------------===//

#include "msan.h"
#include "interception/interception.h"

#if MSAN_REPLACE_OPERATORS_NEW_AND_DELETE

#include <stddef.h>

using namespace __msan;  // NOLINT

// Fake std::nothrow_t to avoid including <new>.
namespace std {
  struct nothrow_t {};
}  // namespace std


#define OPERATOR_NEW_BODY \
  GET_MALLOC_STACK_TRACE; \
  return MsanReallocate(&stack, 0, size, sizeof(u64), false)

INTERCEPTOR_ATTRIBUTE
void *operator new(size_t size) { OPERATOR_NEW_BODY; }
INTERCEPTOR_ATTRIBUTE
void *operator new[](size_t size) { OPERATOR_NEW_BODY; }
INTERCEPTOR_ATTRIBUTE
void *operator new(size_t size, std::nothrow_t const&) { OPERATOR_NEW_BODY; }
INTERCEPTOR_ATTRIBUTE
void *operator new[](size_t size, std::nothrow_t const&) { OPERATOR_NEW_BODY; }

#define OPERATOR_DELETE_BODY \
  GET_MALLOC_STACK_TRACE; \
  if (ptr) MsanDeallocate(&stack, ptr)

INTERCEPTOR_ATTRIBUTE
void operator delete(void *ptr) throw() { OPERATOR_DELETE_BODY; }
INTERCEPTOR_ATTRIBUTE
void operator delete[](void *ptr) throw() { OPERATOR_DELETE_BODY; }
INTERCEPTOR_ATTRIBUTE
void operator delete(void *ptr, std::nothrow_t const&) { OPERATOR_DELETE_BODY; }
INTERCEPTOR_ATTRIBUTE
void operator delete[](void *ptr, std::nothrow_t const&) {
  OPERATOR_DELETE_BODY;
}

#endif // MSAN_REPLACE_OPERATORS_NEW_AND_DELETE
