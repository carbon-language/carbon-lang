//===-- sanitizer_stacktrace.cc -------------------------------------------===//
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

#include "sanitizer_common.h"
#include "sanitizer_flags.h"
#include "sanitizer_stacktrace.h"

namespace __sanitizer {

uptr StackTrace::GetPreviousInstructionPc(uptr pc) {
#if defined(__arm__)
  // Cancel Thumb bit.
  pc = pc & (~1);
#endif
#if defined(__powerpc__) || defined(__powerpc64__)
  // PCs are always 4 byte aligned.
  return pc - 4;
#elif defined(__sparc__)
  return pc - 8;
#else
  return pc - 1;
#endif
}

uptr StackTrace::GetCurrentPc() {
  return GET_CALLER_PC();
}

void StackTrace::FastUnwindStack(uptr pc, uptr bp,
                                 uptr stack_top, uptr stack_bottom,
                                 uptr max_depth) {
  CHECK_GE(max_depth, 2);
  trace[0] = pc;
  size = 1;
  uhwptr *frame = (uhwptr *)bp;
  uhwptr *prev_frame = frame - 1;
  if (stack_top < 4096) return;  // Sanity check for stack top.
  // Avoid infinite loop when frame == frame[0] by using frame > prev_frame.
  while (frame > prev_frame &&
         frame < (uhwptr *)stack_top - 2 &&
         frame > (uhwptr *)stack_bottom &&
         IsAligned((uptr)frame, sizeof(*frame)) &&
         size < max_depth) {
    uhwptr pc1 = frame[1];
    if (pc1 != pc) {
      trace[size++] = (uptr) pc1;
    }
    prev_frame = frame;
    frame = (uhwptr *)frame[0];
  }
}

static bool MatchPc(uptr cur_pc, uptr trace_pc, uptr threshold) {
  return cur_pc - trace_pc <= threshold || trace_pc - cur_pc <= threshold;
}

void StackTrace::PopStackFrames(uptr count) {
  CHECK_LT(count, size);
  size -= count;
  for (uptr i = 0; i < size; ++i) {
    trace[i] = trace[i + count];
  }
}

uptr StackTrace::LocatePcInTrace(uptr pc) {
  // Use threshold to find PC in stack trace, as PC we want to unwind from may
  // slightly differ from return address in the actual unwinded stack trace.
  const int kPcThreshold = 288;
  for (uptr i = 0; i < size; ++i) {
    if (MatchPc(pc, trace[i], kPcThreshold))
      return i;
  }
  return 0;
}

}  // namespace __sanitizer
