//===-- tsan_interface_inl.h ------------------------------------*- C++ -*-===//
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
#include "tsan_rtl.h"

#define CALLERPC ((uptr)__builtin_return_address(0))

using namespace __tsan;  // NOLINT

void __tsan_read1(void *addr) {
  MemoryAccess(cur_thread(), CALLERPC, (uptr)addr, 0, 0);
}

void __tsan_read2(void *addr) {
  MemoryAccess(cur_thread(), CALLERPC, (uptr)addr, 1, 0);
}

void __tsan_read4(void *addr) {
  MemoryAccess(cur_thread(), CALLERPC, (uptr)addr, 2, 0);
}

void __tsan_read8(void *addr) {
  MemoryAccess(cur_thread(), CALLERPC, (uptr)addr, 3, 0);
}

void __tsan_write1(void *addr) {
  MemoryAccess(cur_thread(), CALLERPC, (uptr)addr, 0, 1);
}

void __tsan_write2(void *addr) {
  MemoryAccess(cur_thread(), CALLERPC, (uptr)addr, 1, 1);
}

void __tsan_write4(void *addr) {
  MemoryAccess(cur_thread(), CALLERPC, (uptr)addr, 2, 1);
}

void __tsan_write8(void *addr) {
  MemoryAccess(cur_thread(), CALLERPC, (uptr)addr, 3, 1);
}

void __tsan_vptr_update(void **vptr_p, void *new_val) {
  CHECK_EQ(sizeof(vptr_p), 8);
  if (*vptr_p != new_val)
    MemoryAccess(cur_thread(), CALLERPC, (uptr)vptr_p, 3, 1);
}

void __tsan_func_entry(void *pc) {
  FuncEntry(cur_thread(), (uptr)pc);
}

void __tsan_func_exit() {
  FuncExit(cur_thread());
}
