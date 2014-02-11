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

// Get the stack trace with the given pc and bp.
// The pc will be in the position 0 of the resulting stack trace.
// The bp may refer to the current frame or to the caller's frame.
#if SANITIZER_WINDOWS
#define GET_STACK_TRACE_WITH_PC_BP_AND_CONTEXT(max_s, pc, bp, context, fast) \
  StackTrace stack;                                                          \
  stack.Unwind(max_s, pc, bp, context, 0, 0, fast)
#else
#define GET_STACK_TRACE_WITH_PC_BP_AND_CONTEXT(max_s, pc, bp, context, fast)   \
  StackTrace stack;                                                            \
  {                                                                            \
    AsanThread *t;                                                             \
    stack.size = 0;                                                            \
    if (asan_inited) {                                                         \
      if ((t = GetCurrentThread()) && !t->isUnwinding()) {                     \
        uptr stack_top = t->stack_top();                                       \
        uptr stack_bottom = t->stack_bottom();                                 \
        ScopedUnwinding unwind_scope(t);                                       \
        stack.Unwind(max_s, pc, bp, context, stack_top, stack_bottom, fast);   \
      } else if (t == 0 && !fast) {                                            \
        /* If GetCurrentThread() has failed, try to do slow unwind anyways. */ \
        stack.Unwind(max_s, pc, bp, context, 0, 0, false);                     \
      }                                                                        \
    }                                                                          \
  }
#endif  // SANITIZER_WINDOWS

// NOTE: A Rule of thumb is to retrieve stack trace in the interceptors
// as early as possible (in functions exposed to the user), as we generally
// don't want stack trace to contain functions from ASan internals.

#define GET_STACK_TRACE(max_size, fast)                                        \
  GET_STACK_TRACE_WITH_PC_BP_AND_CONTEXT(max_size, StackTrace::GetCurrentPc(), \
                                         GET_CURRENT_FRAME(), 0, fast)

#define GET_STACK_TRACE_FATAL(pc, bp)                               \
  GET_STACK_TRACE_WITH_PC_BP_AND_CONTEXT(kStackTraceMax, pc, bp, 0, \
                                         common_flags()->fast_unwind_on_fatal)

#define GET_STACK_TRACE_SIGNAL(pc, bp, context)                           \
  GET_STACK_TRACE_WITH_PC_BP_AND_CONTEXT(kStackTraceMax, pc, bp, context, \
                                         common_flags()->fast_unwind_on_fatal)

#define GET_STACK_TRACE_FATAL_HERE                                \
  GET_STACK_TRACE(kStackTraceMax, common_flags()->fast_unwind_on_fatal)

#define GET_STACK_TRACE_THREAD                                    \
  GET_STACK_TRACE(kStackTraceMax, true)

#define GET_STACK_TRACE_MALLOC                                    \
  GET_STACK_TRACE(common_flags()->malloc_context_size,            \
                  common_flags()->fast_unwind_on_malloc)

#define GET_STACK_TRACE_FREE GET_STACK_TRACE_MALLOC

#define PRINT_CURRENT_STACK()   \
  {                             \
    GET_STACK_TRACE_FATAL_HERE; \
    stack.Print();              \
  }

#endif  // ASAN_STACK_H
