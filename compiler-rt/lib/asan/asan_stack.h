//===-- asan_stack.h --------------------------------------------*- C++ -*-===//
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
// ASan-private header for asan_stack.cc.
//===----------------------------------------------------------------------===//
#ifndef ASAN_STACK_H
#define ASAN_STACK_H

#include "sanitizer_common/sanitizer_stacktrace.h"

namespace __asan {

void GetStackTrace(StackTrace *stack, uptr max_s, uptr pc, uptr bp);
void PrintStack(StackTrace *stack);

}  // namespace __asan

// Get the stack trace with the given pc and bp.
// The pc will be in the position 0 of the resulting stack trace.
// The bp may refer to the current frame or to the caller's frame.
// fast_unwind is currently unused.
#define GET_STACK_TRACE_WITH_PC_AND_BP(max_s, pc, bp)               \
  StackTrace stack;                                             \
  GetStackTrace(&stack, max_s, pc, bp)

// NOTE: A Rule of thumb is to retrieve stack trace in the interceptors
// as early as possible (in functions exposed to the user), as we generally
// don't want stack trace to contain functions from ASan internals.

#define GET_STACK_TRACE_HERE(max_size)                        \
  GET_STACK_TRACE_WITH_PC_AND_BP(max_size,                    \
      StackTrace::GetCurrentPc(), GET_CURRENT_FRAME())

#define GET_STACK_TRACE_HERE_FOR_MALLOC                             \
  GET_STACK_TRACE_HERE(flags()->malloc_context_size)

#define GET_STACK_TRACE_HERE_FOR_FREE(ptr)                          \
  GET_STACK_TRACE_HERE(flags()->malloc_context_size)

#define PRINT_CURRENT_STACK()                    \
  {                                              \
    GET_STACK_TRACE_HERE(kStackTraceMax);        \
    PrintStack(&stack);                          \
  }

#endif  // ASAN_STACK_H
