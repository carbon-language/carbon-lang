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
                        uptr stack_bottom, bool request_fast_unwind) {
  if (!WillUseFastUnwind(request_fast_unwind))
    SlowUnwindStack(pc, max_depth);
  else
    FastUnwindStack(pc, bp, stack_top, stack_bottom, max_depth);

  top_frame_bp = size ? bp : 0;
}

}  // namespace __sanitizer
