//===-- tsan_interface.cc -------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//

#include "tsan_interface.h"
#include "tsan_interface_ann.h"
#include "tsan_rtl.h"
#include "sanitizer_common/sanitizer_internal_defs.h"

#define CALLERPC ((uptr)__builtin_return_address(0))

using namespace __tsan;  // NOLINT

typedef u16 uint16_t;
typedef u32 uint32_t;
typedef u64 uint64_t;

void __tsan_init() {
  Initialize(cur_thread());
}

void __tsan_read16(void *addr) {
  MemoryRead(cur_thread(), CALLERPC, (uptr)addr, kSizeLog8);
  MemoryRead(cur_thread(), CALLERPC, (uptr)addr + 8, kSizeLog8);
}

void __tsan_write16(void *addr) {
  MemoryWrite(cur_thread(), CALLERPC, (uptr)addr, kSizeLog8);
  MemoryWrite(cur_thread(), CALLERPC, (uptr)addr + 8, kSizeLog8);
}

u16 __tsan_unaligned_read2(const uu16 *addr) {
  UnalignedMemoryAccess(cur_thread(), CALLERPC, (uptr)addr, 2, false, false);
  return *addr;
}

u32 __tsan_unaligned_read4(const uu32 *addr) {
  UnalignedMemoryAccess(cur_thread(), CALLERPC, (uptr)addr, 4, false, false);
  return *addr;
}

u64 __tsan_unaligned_read8(const uu64 *addr) {
  UnalignedMemoryAccess(cur_thread(), CALLERPC, (uptr)addr, 8, false, false);
  return *addr;
}

void __tsan_unaligned_write2(uu16 *addr, u16 v) {
  UnalignedMemoryAccess(cur_thread(), CALLERPC, (uptr)addr, 2, true, false);
  *addr = v;
}

void __tsan_unaligned_write4(uu32 *addr, u32 v) {
  UnalignedMemoryAccess(cur_thread(), CALLERPC, (uptr)addr, 4, true, false);
  *addr = v;
}

void __tsan_unaligned_write8(uu64 *addr, u64 v) {
  UnalignedMemoryAccess(cur_thread(), CALLERPC, (uptr)addr, 8, true, false);
  *addr = v;
}

extern "C" {
uint16_t __sanitizer_unaligned_load16(void *addr)
    ALIAS("__tsan_unaligned_read2") SANITIZER_INTERFACE_ATTRIBUTE;
uint32_t __sanitizer_unaligned_load32(void *addr)
    ALIAS("__tsan_unaligned_read4") SANITIZER_INTERFACE_ATTRIBUTE;
uint64_t __sanitizer_unaligned_load64(void *addr)
    ALIAS("__tsan_unaligned_read8") SANITIZER_INTERFACE_ATTRIBUTE;
void __sanitizer_unaligned_store16(void *addr, uint16_t v)
    ALIAS("__tsan_unaligned_write2") SANITIZER_INTERFACE_ATTRIBUTE;
void __sanitizer_unaligned_store32(void *addr, uint32_t v)
    ALIAS("__tsan_unaligned_write4") SANITIZER_INTERFACE_ATTRIBUTE;
void __sanitizer_unaligned_store64(void *addr, uint64_t v)
    ALIAS("__tsan_unaligned_write8") SANITIZER_INTERFACE_ATTRIBUTE;
}

void __tsan_acquire(void *addr) {
  Acquire(cur_thread(), CALLERPC, (uptr)addr);
}

void __tsan_release(void *addr) {
  Release(cur_thread(), CALLERPC, (uptr)addr);
}
