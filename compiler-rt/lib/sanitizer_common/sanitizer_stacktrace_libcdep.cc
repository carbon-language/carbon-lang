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
#include "sanitizer_stacktrace.h"
#include "sanitizer_symbolizer.h"

namespace __sanitizer {

static void PrintStackFramePrefix(InternalScopedString *buffer, uptr frame_num,
                                  uptr pc) {
  buffer->append("    #%zu 0x%zx", frame_num, pc);
}

void StackTrace::Print() const {
  if (trace == nullptr || size == 0) {
    Printf("    <empty stack>\n\n");
    return;
  }
  InternalScopedBuffer<AddressInfo> addr_frames(64);
  InternalScopedString frame_desc(GetPageSizeCached() * 2);
  uptr frame_num = 0;
  for (uptr i = 0; i < size && trace[i]; i++) {
    // PCs in stack traces are actually the return addresses, that is,
    // addresses of the next instructions after the call.
    uptr pc = GetPreviousInstructionPc(trace[i]);
    uptr addr_frames_num = Symbolizer::GetOrInit()->SymbolizePC(
        pc, addr_frames.data(), addr_frames.size());
    if (addr_frames_num == 0) {
      frame_desc.clear();
      PrintStackFramePrefix(&frame_desc, frame_num++, pc);
      frame_desc.append(" (<unknown module>)");
      Printf("%s\n", frame_desc.data());
      continue;
    }
    for (uptr j = 0; j < addr_frames_num; j++) {
      AddressInfo &info = addr_frames[j];
      frame_desc.clear();
      PrintStackFramePrefix(&frame_desc, frame_num++, pc);
      if (info.function) {
        frame_desc.append(" in %s", info.function);
        // Print offset in function if we don't know the source file.
        if (!info.file && info.function_offset != AddressInfo::kUnknown)
          frame_desc.append("+0x%zx", info.function_offset);
      }
      if (info.file) {
        frame_desc.append(" ");
        PrintSourceLocation(&frame_desc, info.file, info.line, info.column);
      } else if (info.module) {
        frame_desc.append(" ");
        PrintModuleAndOffset(&frame_desc, info.module, info.module_offset);
      }
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
