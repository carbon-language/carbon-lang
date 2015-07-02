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

#include "sanitizer_common.h"
#include "sanitizer_placement_new.h"
#include "sanitizer_stacktrace.h"
#include "sanitizer_stacktrace_printer.h"
#include "sanitizer_symbolizer.h"

namespace __sanitizer {

void StackTrace::Print() const {
  if (trace == nullptr || size == 0) {
    Printf("    <empty stack>\n\n");
    return;
  }
  InternalScopedString frame_desc(GetPageSizeCached() * 2);
  uptr frame_num = 0;
  for (uptr i = 0; i < size && trace[i]; i++) {
    // PCs in stack traces are actually the return addresses, that is,
    // addresses of the next instructions after the call.
    uptr pc = GetPreviousInstructionPc(trace[i]);
    SymbolizedStack *frames = Symbolizer::GetOrInit()->SymbolizePC(pc);
    CHECK(frames);
    for (SymbolizedStack *cur = frames; cur; cur = cur->next) {
      frame_desc.clear();
      RenderFrame(&frame_desc, common_flags()->stack_trace_format, frame_num++,
                  cur->info, common_flags()->symbolize_vs_style,
                  common_flags()->strip_path_prefix);
      Printf("%s\n", frame_desc.data());
    }
    frames->ClearAll();
  }
  // Always print a trailing empty line after stack trace.
  Printf("\n");
}

void BufferedStackTrace::Unwind(u32 max_depth, uptr pc, uptr bp, void *context,
                                uptr stack_top, uptr stack_bottom,
                                bool request_fast_unwind) {
  top_frame_bp = (max_depth > 0) ? bp : 0;
  // Avoid doing any work for small max_depth.
  if (max_depth == 0) {
    size = 0;
    return;
  }
  if (max_depth == 1) {
    size = 1;
    trace_buffer[0] = pc;
    return;
  }
  if (!WillUseFastUnwind(request_fast_unwind)) {
#if SANITIZER_CAN_SLOW_UNWIND
    if (context)
      SlowUnwindStackWithContext(pc, context, max_depth);
    else
      SlowUnwindStack(pc, max_depth);
#else
    UNREACHABLE("slow unwind requested but not available");
#endif
  } else {
    FastUnwindStack(pc, bp, stack_top, stack_bottom, max_depth);
  }
}

}  // namespace __sanitizer
