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
  const int kMaxAddrFrames = 64;
  InternalScopedBuffer<AddressInfo> addr_frames(kMaxAddrFrames);
  for (uptr i = 0; i < kMaxAddrFrames; i++)
    new(&addr_frames[i]) AddressInfo();
  InternalScopedString frame_desc(GetPageSizeCached() * 2);
  uptr frame_num = 0;
  for (uptr i = 0; i < size && trace[i]; i++) {
    // PCs in stack traces are actually the return addresses, that is,
    // addresses of the next instructions after the call.
    uptr pc = GetPreviousInstructionPc(trace[i]);
    uptr addr_frames_num = Symbolizer::GetOrInit()->SymbolizePC(
        pc, addr_frames.data(), kMaxAddrFrames);
    if (addr_frames_num == 0) {
      addr_frames[0].address = pc;
      addr_frames_num = 1;
    }
    for (uptr j = 0; j < addr_frames_num; j++) {
      AddressInfo &info = addr_frames[j];
      frame_desc.clear();
      RenderFrame(&frame_desc, "DEFAULT", frame_num++, info,
                  common_flags()->strip_path_prefix);
      Printf("%s\n", frame_desc.data());
      info.Clear();
    }
  }
  // Always print a trailing empty line after stack trace.
  Printf("\n");
}

void BufferedStackTrace::Unwind(uptr max_depth, uptr pc, uptr bp, void *context,
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
    if (context)
      SlowUnwindStackWithContext(pc, context, max_depth);
    else
      SlowUnwindStack(pc, max_depth);
  } else {
    FastUnwindStack(pc, bp, stack_top, stack_bottom, max_depth);
  }
}

}  // namespace __sanitizer
