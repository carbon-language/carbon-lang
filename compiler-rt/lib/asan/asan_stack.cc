//===-- asan_stack.cc -----------------------------------------------------===//
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
// Code for ASan stack trace.
//===----------------------------------------------------------------------===//
#include "asan_internal.h"
#include "asan_flags.h"
#include "asan_stack.h"
#include "sanitizer_common/sanitizer_flags.h"

namespace __asan {

static bool MaybeCallAsanSymbolize(const void *pc, char *out_buffer,
                                   int out_size) {
  return (&__asan_symbolize) ? __asan_symbolize(pc, out_buffer, out_size)
                             : false;
}

void PrintStack(const uptr *trace, uptr size) {
  StackTrace::PrintStack(trace, size, common_flags()->symbolize,
                         MaybeCallAsanSymbolize);
}
void PrintStack(StackTrace *stack) {
  PrintStack(stack->trace, stack->size);
}

}  // namespace __asan

// ------------------ Interface -------------- {{{1

// Provide default implementation of __asan_symbolize that does nothing
// and may be overriden by user if he wants to use his own symbolization.
// ASan on Windows has its own implementation of this.
#if !SANITIZER_WINDOWS && !SANITIZER_SUPPORTS_WEAK_HOOKS
SANITIZER_INTERFACE_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE NOINLINE
bool __asan_symbolize(const void *pc, char *out_buffer, int out_size) {
  return false;
}
#endif
