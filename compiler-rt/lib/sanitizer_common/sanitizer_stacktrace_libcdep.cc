//===-- sanitizer_stacktrace_libcdep.cc -----------------------------------===//
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

#include "sanitizer_stacktrace.h"

namespace __sanitizer {

void StackTrace::Unwind(uptr max_depth, uptr pc, uptr bp, uptr stack_top,
                        uptr stack_bottom, bool fast) {
  // Check if fast unwind is available. Fast unwind is the only option on Mac.
  if (!SANITIZER_CAN_FAST_UNWIND)
    fast = false;
  else if (SANITIZER_MAC)
    fast = true;

  if (!fast)
    SlowUnwindStack(pc, max_depth);
  else
    FastUnwindStack(pc, bp, stack_top, stack_bottom, max_depth);
}

}  // namespace __sanitizer
