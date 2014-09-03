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

// Check if given pointer points into allocated stack area.
static inline bool IsValidFrame(uptr frame, uptr stack_top, uptr stack_bottom) {
  return frame > stack_bottom && frame < stack_top - 2 * sizeof (uhwptr);
}

// In GCC on ARM bp points to saved lr, not fp, so we should check the next
// cell in stack to be a saved frame pointer. GetCanonicFrame returns the
// pointer to saved frame pointer in any case.
static inline uhwptr *GetCanonicFrame(uptr bp,
                                      uptr stack_top,
                                      uptr stack_bottom) {
#ifdef __arm__
  if (!IsValidFrame(bp, stack_top, stack_bottom)) return 0;
  uhwptr *bp_prev = (uhwptr *)bp;
  if (IsValidFrame((uptr)bp_prev[0], stack_top, stack_bottom)) return bp_prev;
  return bp_prev - 1;
#else
  return (uhwptr*)bp;
#endif
}

void StackTrace::FastUnwindStack(uptr pc, uptr bp,
                                 uptr stack_top, uptr stack_bottom,
                                 uptr max_depth) {
  CHECK_GE(max_depth, 2);
  trace[0] = pc;
  size = 1;
  if (stack_top < 4096) return;  // Sanity check for stack top.
  uhwptr *frame = GetCanonicFrame(bp, stack_top, stack_bottom);
  uhwptr *prev_frame = 0;
  // Avoid infinite loop when frame == frame[0] by using frame > prev_frame.
  while (frame > prev_frame &&
         IsValidFrame((uptr)frame, stack_top, stack_bottom) &&
         IsAligned((uptr)frame, sizeof(*frame)) &&
         size < max_depth) {
    uhwptr pc1 = frame[1];
    if (pc1 != pc) {
      trace[size++] = (uptr) pc1;
    }
    prev_frame = frame;
    frame = GetCanonicFrame((uptr)frame[0], stack_top, stack_bottom);
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
