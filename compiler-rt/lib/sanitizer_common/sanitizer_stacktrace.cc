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
#include "sanitizer_procmaps.h"
#include "sanitizer_stacktrace.h"
#include "sanitizer_symbolizer.h"

namespace __sanitizer {
const char *StripPathPrefix(const char *filepath,
                            const char *strip_file_prefix) {
  if (filepath == 0) return 0;
  if (filepath == internal_strstr(filepath, strip_file_prefix))
    return filepath + internal_strlen(strip_file_prefix);
  return filepath;
}

// ----------------------- StackTrace ----------------------------- {{{1
uptr StackTrace::GetPreviousInstructionPc(uptr pc) {
#ifdef __arm__
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

static void PrintStackFramePrefix(uptr frame_num, uptr pc) {
  Printf("    #%zu 0x%zx", frame_num, pc);
}

static void PrintSourceLocation(const char *file, int line, int column,
                                const char *strip_file_prefix) {
  CHECK(file);
  Printf(" %s", StripPathPrefix(file, strip_file_prefix));
  if (line > 0) {
    Printf(":%d", line);
    if (column > 0)
      Printf(":%d", column);
  }
}

static void PrintModuleAndOffset(const char *module, uptr offset,
                                 const char *strip_file_prefix) {
  Printf(" (%s+0x%zx)", StripPathPrefix(module, strip_file_prefix), offset);
}

void StackTrace::PrintStack(const uptr *addr, uptr size,
                            bool symbolize, const char *strip_file_prefix,
                            SymbolizeCallback symbolize_callback ) {
  MemoryMappingLayout proc_maps(/*cache_enabled*/true);
  InternalScopedBuffer<char> buff(GetPageSizeCached() * 2);
  InternalScopedBuffer<AddressInfo> addr_frames(64);
  uptr frame_num = 0;
  for (uptr i = 0; i < size && addr[i]; i++) {
    // PCs in stack traces are actually the return addresses, that is,
    // addresses of the next instructions after the call.
    uptr pc = GetPreviousInstructionPc(addr[i]);
    uptr addr_frames_num = 0;  // The number of stack frames for current
                               // instruction address.
    if (symbolize_callback) {
      if (symbolize_callback((void*)pc, buff.data(), buff.size())) {
        addr_frames_num = 1;
        PrintStackFramePrefix(frame_num, pc);
        // We can't know anything about the string returned by external
        // symbolizer, but if it starts with filename, try to strip path prefix
        // from it.
        Printf(" %s\n", StripPathPrefix(buff.data(), strip_file_prefix));
        frame_num++;
      }
    }
    if (symbolize && addr_frames_num == 0) {
      // Use our own (online) symbolizer, if necessary.
      addr_frames_num = SymbolizeCode(pc, addr_frames.data(),
                                      addr_frames.size());
      for (uptr j = 0; j < addr_frames_num; j++) {
        AddressInfo &info = addr_frames[j];
        PrintStackFramePrefix(frame_num, pc);
        if (info.function) {
          Printf(" in %s", info.function);
        }
        if (info.file) {
          PrintSourceLocation(info.file, info.line, info.column,
                              strip_file_prefix);
        } else if (info.module) {
          PrintModuleAndOffset(info.module, info.module_offset,
                               strip_file_prefix);
        }
        Printf("\n");
        info.Clear();
        frame_num++;
      }
    }
    if (addr_frames_num == 0) {
      // If online symbolization failed, try to output at least module and
      // offset for instruction.
      PrintStackFramePrefix(frame_num, pc);
      uptr offset;
      if (proc_maps.GetObjectNameAndOffset(pc, &offset,
                                           buff.data(), buff.size(),
                                           /* protection */0)) {
        PrintModuleAndOffset(buff.data(), offset, strip_file_prefix);
      }
      Printf("\n");
      frame_num++;
    }
  }
}

uptr StackTrace::GetCurrentPc() {
  return GET_CALLER_PC();
}

void StackTrace::FastUnwindStack(uptr pc, uptr bp,
                                 uptr stack_top, uptr stack_bottom) {
  CHECK(size == 0 && trace[0] == pc);
  size = 1;
  uhwptr *frame = (uhwptr *)bp;
  uhwptr *prev_frame = frame - 1;
  // Avoid infinite loop when frame == frame[0] by using frame > prev_frame.
  while (frame > prev_frame &&
         frame < (uhwptr *)stack_top - 2 &&
         frame > (uhwptr *)stack_bottom &&
         size < max_size) {
    uhwptr pc1 = frame[1];
    if (pc1 != pc) {
      trace[size++] = (uptr) pc1;
    }
    prev_frame = frame;
    frame = (uhwptr *)frame[0];
  }
}

void StackTrace::PopStackFrames(uptr count) {
  CHECK(size >= count);
  size -= count;
  for (uptr i = 0; i < size; i++) {
    trace[i] = trace[i + count];
  }
}

// On 32-bits we don't compress stack traces.
// On 64-bits we compress stack traces: if a given pc differes slightly from
// the previous one, we record a 31-bit offset instead of the full pc.
SANITIZER_INTERFACE_ATTRIBUTE
uptr StackTrace::CompressStack(StackTrace *stack, u32 *compressed, uptr size) {
#if SANITIZER_WORDSIZE == 32
  // Don't compress, just copy.
  uptr res = 0;
  for (uptr i = 0; i < stack->size && i < size; i++) {
    compressed[i] = stack->trace[i];
    res++;
  }
  if (stack->size < size)
    compressed[stack->size] = 0;
#else  // 64 bits, compress.
  uptr prev_pc = 0;
  const uptr kMaxOffset = (1ULL << 30) - 1;
  uptr c_index = 0;
  uptr res = 0;
  for (uptr i = 0, n = stack->size; i < n; i++) {
    uptr pc = stack->trace[i];
    if (!pc) break;
    if ((s64)pc < 0) break;
    // Printf("C pc[%zu] %zx\n", i, pc);
    if (prev_pc - pc < kMaxOffset || pc - prev_pc < kMaxOffset) {
      uptr offset = (s64)(pc - prev_pc);
      offset |= (1U << 31);
      if (c_index >= size) break;
      // Printf("C co[%zu] offset %zx\n", i, offset);
      compressed[c_index++] = offset;
    } else {
      uptr hi = pc >> 32;
      uptr lo = (pc << 32) >> 32;
      CHECK_EQ((hi & (1 << 31)), 0);
      if (c_index + 1 >= size) break;
      // Printf("C co[%zu] hi/lo: %zx %zx\n", c_index, hi, lo);
      compressed[c_index++] = hi;
      compressed[c_index++] = lo;
    }
    res++;
    prev_pc = pc;
  }
  if (c_index < size)
    compressed[c_index] = 0;
  if (c_index + 1 < size)
    compressed[c_index + 1] = 0;
#endif  // SANITIZER_WORDSIZE

  // debug-only code
#if 0
  StackTrace check_stack;
  UncompressStack(&check_stack, compressed, size);
  if (res < check_stack.size) {
    Printf("res %zu check_stack.size %zu; c_size %zu\n", res,
           check_stack.size, size);
  }
  // |res| may be greater than check_stack.size, because
  // UncompressStack(CompressStack(stack)) eliminates the 0x0 frames.
  CHECK(res >= check_stack.size);
  CHECK_EQ(0, REAL(memcmp)(check_stack.trace, stack->trace,
                          check_stack.size * sizeof(uptr)));
#endif

  return res;
}

SANITIZER_INTERFACE_ATTRIBUTE
void StackTrace::UncompressStack(StackTrace *stack,
                                 u32 *compressed, uptr size) {
#if SANITIZER_WORDSIZE == 32
  // Don't uncompress, just copy.
  stack->size = 0;
  for (uptr i = 0; i < size && i < kStackTraceMax; i++) {
    if (!compressed[i]) break;
    stack->size++;
    stack->trace[i] = compressed[i];
  }
#else  // 64 bits, uncompress
  uptr prev_pc = 0;
  stack->size = 0;
  for (uptr i = 0; i < size && stack->size < kStackTraceMax; i++) {
    u32 x = compressed[i];
    uptr pc = 0;
    if (x & (1U << 31)) {
      // Printf("U co[%zu] offset: %x\n", i, x);
      // this is an offset
      s32 offset = x;
      offset = (offset << 1) >> 1;  // remove the 31-byte and sign-extend.
      pc = prev_pc + offset;
      CHECK(pc);
    } else {
      // CHECK(i + 1 < size);
      if (i + 1 >= size) break;
      uptr hi = x;
      uptr lo = compressed[i+1];
      // Printf("U co[%zu] hi/lo: %zx %zx\n", i, hi, lo);
      i++;
      pc = (hi << 32) | lo;
      if (!pc) break;
    }
    // Printf("U pc[%zu] %zx\n", stack->size, pc);
    stack->trace[stack->size++] = pc;
    prev_pc = pc;
  }
#endif  // SANITIZER_WORDSIZE
}

}  // namespace __sanitizer
