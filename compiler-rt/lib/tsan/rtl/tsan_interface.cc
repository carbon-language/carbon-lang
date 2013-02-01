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

#define CALLERPC ((uptr)__builtin_return_address(0))

using namespace __tsan;  // NOLINT

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

void __tsan_acquire(void *addr) {
  Acquire(cur_thread(), CALLERPC, (uptr)addr);
}

void __tsan_release(void *addr) {
  Release(cur_thread(), CALLERPC, (uptr)addr);
}
