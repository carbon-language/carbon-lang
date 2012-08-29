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
#include "asan_flags.h"
#include "asan_stack.h"
#include "sanitizer/asan_interface.h"

namespace __asan {

static __asan_symbolize_callback symbolize_callback;

void PrintStack(StackTrace *stack) {
  stack->PrintStack(stack->trace, stack->size, flags()->symbolize,
                    flags()->strip_path_prefix, symbolize_callback);
}


}  // namespace __asan

// ------------------ Interface -------------- {{{1
using namespace __asan;  // NOLINT

void NOINLINE __asan_set_symbolize_callback(
    __asan_symbolize_callback callback) {
  symbolize_callback = callback;
}
