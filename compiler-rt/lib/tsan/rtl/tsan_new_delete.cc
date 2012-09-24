//===-- tsan_new_delete.cc ------------------------------------------------===//
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
// Interceptors for operators new and delete.
//===----------------------------------------------------------------------===//

#include "tsan_interceptors.h"
#include "tsan_mman.h"
#include "tsan_rtl.h"

#include <stddef.h>
#include <new>

namespace __tsan {
// This function is a no-op. We need it to make sure that object file
// with our replacements will actually be loaded from static TSan
// run-time library at link-time.
void ReplaceOperatorsNewAndDelete() { }
}

using namespace __tsan;  // NOLINT

#define OPERATOR_NEW_BODY(mangled_name) \
  void *p = 0; \
  {  \
    SCOPED_INTERCEPTOR_RAW(mangled_name, size); \
    p = user_alloc(thr, pc, size); \
  }  \
  invoke_malloc_hook(p, size);  \
  return p;

void *operator new(size_t size) throw(std::bad_alloc) {
  OPERATOR_NEW_BODY(_Znwm);
}
void *operator new[](size_t size) throw(std::bad_alloc) {
  OPERATOR_NEW_BODY(_Znam);
}
void *operator new(size_t size, std::nothrow_t const&) throw() {
  OPERATOR_NEW_BODY(_ZnwmRKSt9nothrow_t);
}
void *operator new[](size_t size, std::nothrow_t const&) throw() {
  OPERATOR_NEW_BODY(_ZnamRKSt9nothrow_t);
}

#define OPERATOR_DELETE_BODY(mangled_name) \
  if (ptr == 0) return;  \
  invoke_free_hook(ptr);  \
  SCOPED_INTERCEPTOR_RAW(mangled_name, ptr);  \
  user_free(thr, pc, ptr);

void operator delete(void *ptr) throw() {
  OPERATOR_DELETE_BODY(_ZdlPv);
}
void operator delete[](void *ptr) throw() {
  OPERATOR_DELETE_BODY(_ZdlPvRKSt9nothrow_t);
}
void operator delete(void *ptr, std::nothrow_t const&) throw() {
  OPERATOR_DELETE_BODY(_ZdaPv);
}
void operator delete[](void *ptr, std::nothrow_t const&) throw() {
  OPERATOR_DELETE_BODY(_ZdaPvRKSt9nothrow_t);
}
