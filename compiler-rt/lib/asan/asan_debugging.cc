//===-- asan_debugging.cc -------------------------------------------------===//
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
// This file contains various functions that are generally useful to call when
// using a debugger (LLDB, GDB).
//===----------------------------------------------------------------------===//

#include "asan_allocator.h"
#include "asan_flags.h"
#include "asan_internal.h"
#include "asan_mapping.h"
#include "asan_thread.h"

namespace __asan {

uptr AsanGetStack(uptr addr, uptr *trace, uptr size, u32 *thread_id,
                  bool alloc_stack) {
  AsanChunkView chunk = FindHeapChunkByAddress(addr);
  if (!chunk.IsValid()) return 0;

  StackTrace stack;
  if (alloc_stack) {
    if (chunk.AllocTid() == kInvalidTid) return 0;
    chunk.GetAllocStack(&stack);
    if (thread_id) *thread_id = chunk.AllocTid();
  } else {
    if (chunk.FreeTid() == kInvalidTid) return 0;
    chunk.GetFreeStack(&stack);
    if (thread_id) *thread_id = chunk.FreeTid();
  }

  if (trace && size) {
    if (size > kStackTraceMax)
      size = kStackTraceMax;
    if (size > stack.size)
      size = stack.size;
    for (uptr i = 0; i < size; i++)
      trace[i] = StackTrace::GetPreviousInstructionPc(stack.trace[i]);

    return size;
  }

  return 0;
}

}  // namespace __asan

using namespace __asan;

SANITIZER_INTERFACE_ATTRIBUTE
uptr __asan_get_alloc_stack(uptr addr, uptr *trace, uptr size, u32 *thread_id) {
  return AsanGetStack(addr, trace, size, thread_id, /* alloc_stack */ true);
}

SANITIZER_INTERFACE_ATTRIBUTE
uptr __asan_get_free_stack(uptr addr, uptr *trace, uptr size, u32 *thread_id) {
  return AsanGetStack(addr, trace, size, thread_id, /* alloc_stack */ false);
}

SANITIZER_INTERFACE_ATTRIBUTE
void __asan_get_shadow_mapping(uptr *shadow_scale, uptr *shadow_offset) {
  if (shadow_scale)
    *shadow_scale = SHADOW_SCALE;
  if (shadow_offset)
    *shadow_offset = SHADOW_OFFSET;
}
