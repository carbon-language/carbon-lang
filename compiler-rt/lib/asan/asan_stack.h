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

#include "asan_flags.h"
#include "asan_thread.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_stacktrace.h"

namespace __asan {

void PrintStack(StackTrace *stack);

}  // namespace __asan

// Get the stack trace with the given pc and bp.
// The pc will be in the position 0 of the resulting stack trace.
// The bp may refer to the current frame or to the caller's frame.
#if SANITIZER_WINDOWS
#define GET_STACK_TRACE_WITH_PC_AND_BP(max_s, pc, bp, fast)     \
  StackTrace stack;                                             \
  GetStackTrace(&stack, max_s, pc, bp, 0, 0, fast)
#else
#define GET_STACK_TRACE_WITH_PC_AND_BP(max_s, pc, bp, fast)     \
  StackTrace stack;                                             \
  {                                                             \
    uptr stack_top = 0, stack_bottom = 0;                       \
    AsanThread *t;                                              \
    if (asan_inited && (t = GetCurrentThread())) {              \
      stack_top = t->stack_top();                               \
      stack_bottom = t->stack_bottom();                         \
    }                                                           \
    GetStackTrace(&stack, max_s, pc, bp,                        \
                  stack_top, stack_bottom, fast);               \
  }
#endif  // SANITIZER_WINDOWS

// NOTE: A Rule of thumb is to retrieve stack trace in the interceptors
// as early as possible (in functions exposed to the user), as we generally
// don't want stack trace to contain functions from ASan internals.

#define GET_STACK_TRACE(max_size, fast)                       \
  GET_STACK_TRACE_WITH_PC_AND_BP(max_size,                    \
      StackTrace::GetCurrentPc(), GET_CURRENT_FRAME(), fast)

#define GET_STACK_TRACE_FATAL(pc, bp)                                 \
  GET_STACK_TRACE_WITH_PC_AND_BP(kStackTraceMax, pc, bp,              \
                                 common_flags()->fast_unwind_on_fatal)

#define GET_STACK_TRACE_FATAL_HERE                                \
  GET_STACK_TRACE(kStackTraceMax, common_flags()->fast_unwind_on_fatal)

#define GET_STACK_TRACE_THREAD                                    \
  GET_STACK_TRACE(kStackTraceMax, true)

#define GET_STACK_TRACE_MALLOC                                    \
  GET_STACK_TRACE(common_flags()->malloc_context_size,            \
                  common_flags()->fast_unwind_on_malloc)

#define GET_STACK_TRACE_FREE GET_STACK_TRACE_MALLOC

#define PRINT_CURRENT_STACK()                    \
  {                                              \
    GET_STACK_TRACE(kStackTraceMax,              \
      common_flags()->fast_unwind_on_fatal);     \
    PrintStack(&stack);                          \
  }

#endif  // ASAN_STACK_H
