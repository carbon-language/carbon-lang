//===-- sanitizer_stacktrace.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries.
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_STACKTRACE_H
#define SANITIZER_STACKTRACE_H

#include "sanitizer_internal_defs.h"

namespace __sanitizer {

static const uptr kStackTraceMax = 256;

#if SANITIZER_LINUX && (defined(__arm__) || \
    defined(__powerpc__) || defined(__powerpc64__) || \
    defined(__sparc__) || \
    defined(__mips__))
#define SANITIZER_CAN_FAST_UNWIND 0
#else
#define SANITIZER_CAN_FAST_UNWIND 1
#endif

struct StackTrace {
  typedef bool (*SymbolizeCallback)(const void *pc, char *out_buffer,
                                     int out_size);
  uptr size;
  uptr max_size;
  uptr trace[kStackTraceMax];
  static void PrintStack(const uptr *addr, uptr size,
                         bool symbolize, const char *strip_file_prefix,
                         SymbolizeCallback symbolize_callback);
  void CopyTo(uptr *dst, uptr dst_size) {
    for (uptr i = 0; i < size && i < dst_size; i++)
      dst[i] = trace[i];
    for (uptr i = size; i < dst_size; i++)
      dst[i] = 0;
  }

  void CopyFrom(uptr *src, uptr src_size) {
    size = src_size;
    if (size > kStackTraceMax) size = kStackTraceMax;
    for (uptr i = 0; i < size; i++) {
      trace[i] = src[i];
    }
  }

  void FastUnwindStack(uptr pc, uptr bp, uptr stack_top, uptr stack_bottom);
  void SlowUnwindStack(uptr pc, uptr max_depth);

  void PopStackFrames(uptr count);

  static uptr GetCurrentPc();
  static uptr GetPreviousInstructionPc(uptr pc);

  static uptr CompressStack(StackTrace *stack,
                            u32 *compressed, uptr size);
  static void UncompressStack(StackTrace *stack,
                              u32 *compressed, uptr size);
};


const char *StripPathPrefix(const char *filepath,
                            const char *strip_file_prefix);

void GetStackTrace(StackTrace *stack, uptr max_s, uptr pc, uptr bp,
                   uptr stack_top, uptr stack_bottom, bool fast);

}  // namespace __sanitizer

// Use this macro if you want to print stack trace with the caller
// of the current function in the top frame.
#define GET_CALLER_PC_BP_SP \
  uptr bp = GET_CURRENT_FRAME();              \
  uptr pc = GET_CALLER_PC();                  \
  uptr local_stack;                           \
  uptr sp = (uptr)&local_stack

// Use this macro if you want to print stack trace with the current
// function in the top frame.
#define GET_CURRENT_PC_BP_SP \
  uptr bp = GET_CURRENT_FRAME();              \
  uptr pc = StackTrace::GetCurrentPc();   \
  uptr local_stack;                           \
  uptr sp = (uptr)&local_stack


#endif  // SANITIZER_STACKTRACE_H
