//===-- asan_stack.cc -------------------------------------------*- C++ -*-===//
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
// Code for ASan stack trace.
//===----------------------------------------------------------------------===//
#include "asan_interceptors.h"
#include "asan_lock.h"
#include "asan_procmaps.h"
#include "asan_stack.h"
#include "asan_thread.h"
#include "asan_thread_registry.h"
#include "sanitizer_common/sanitizer_symbolizer.h"

using namespace __sanitizer; // NOLINT

#ifdef ASAN_USE_EXTERNAL_SYMBOLIZER
extern bool
ASAN_USE_EXTERNAL_SYMBOLIZER(const void *pc, char *out, int out_size);
#endif

namespace __asan {

// ----------------------- AsanStackTrace ----------------------------- {{{1
#if defined(ASAN_USE_EXTERNAL_SYMBOLIZER)
void AsanStackTrace::PrintStack(uptr *addr, uptr size) {
  for (uptr i = 0; i < size && addr[i]; i++) {
    uptr pc = addr[i];
    char buff[4096];
    ASAN_USE_EXTERNAL_SYMBOLIZER((void*)pc, buff, sizeof(buff));
    Printf("  #%zu 0x%zx %s\n", i, pc, buff);
  }
}

#else  // ASAN_USE_EXTERNAL_SYMBOLIZER
void AsanStackTrace::PrintStack(uptr *addr, uptr size) {
  AsanProcMaps proc_maps;
  uptr frame_num = 0;
  for (uptr i = 0; i < size && addr[i]; i++) {
    proc_maps.Reset();
    uptr pc = addr[i];
    uptr offset;
    char filename[4096];
    if (FLAG_symbolize) {
      AddressInfoList *address_info_list = SymbolizeCode(pc);
      for (AddressInfoList *entry = address_info_list; entry;
           entry = entry->next) {
        AddressInfo info = entry->info;
        Printf("    #%zu 0x%zx %s:%d:%d\n", frame_num, pc,
                                            (info.file) ? info.file : "",
                                            info.line, info.column);
        frame_num++;
      }
      address_info_list->Clear();
    } else {
      if (proc_maps.GetObjectNameAndOffset(pc, &offset,
                                           filename, sizeof(filename))) {
        Printf("    #%zu 0x%zx (%s+0x%zx)\n", frame_num, pc, filename,
                                              offset);
      } else {
        Printf("    #%zu 0x%zx\n", frame_num, pc);
      }
      frame_num++;
    }
  }
}
#endif  // ASAN_USE_EXTERNAL_SYMBOLIZER

uptr AsanStackTrace::GetCurrentPc() {
  return GET_CALLER_PC();
}

void AsanStackTrace::FastUnwindStack(uptr pc, uptr bp) {
  CHECK(size == 0 && trace[0] == pc);
  size = 1;
  if (!asan_inited) return;
  AsanThread *t = asanThreadRegistry().GetCurrent();
  if (!t) return;
  uptr *frame = (uptr*)bp;
  uptr *prev_frame = frame;
  uptr *top = (uptr*)t->stack_top();
  uptr *bottom = (uptr*)t->stack_bottom();
  while (frame >= prev_frame &&
         frame < top - 2 &&
         frame > bottom &&
         size < max_size) {
    uptr pc1 = frame[1];
    if (pc1 != pc) {
      trace[size++] = pc1;
    }
    prev_frame = frame;
    frame = (uptr*)frame[0];
  }
}

// On 32-bits we don't compress stack traces.
// On 64-bits we compress stack traces: if a given pc differes slightly from
// the previous one, we record a 31-bit offset instead of the full pc.
uptr AsanStackTrace::CompressStack(AsanStackTrace *stack,
                                   u32 *compressed, uptr size) {
#if __WORDSIZE == 32
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
      CHECK((hi & (1 << 31)) == 0);
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
#endif  // __WORDSIZE

  // debug-only code
#if 0
  AsanStackTrace check_stack;
  UncompressStack(&check_stack, compressed, size);
  if (res < check_stack.size) {
    Printf("res %zu check_stack.size %zu; c_size %zu\n", res,
           check_stack.size, size);
  }
  // |res| may be greater than check_stack.size, because
  // UncompressStack(CompressStack(stack)) eliminates the 0x0 frames.
  CHECK(res >= check_stack.size);
  CHECK(0 == REAL(memcmp)(check_stack.trace, stack->trace,
                          check_stack.size * sizeof(uptr)));
#endif

  return res;
}

void AsanStackTrace::UncompressStack(AsanStackTrace *stack,
                                     u32 *compressed, uptr size) {
#if __WORDSIZE == 32
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
#endif  // __WORDSIZE
}

}  // namespace __asan
