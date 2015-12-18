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

#define TSAN_MEM_ACCESS_FUNC(type, func, size) \
  void __tsan_##type(void *addr) {             \
    ThreadState *thr = cur_thread();           \
    DCHECK_EQ(thr->in_interceptor_count, 0);   \
    func(thr, CALLERPC, (uptr)addr, size);     \
  }

#define TSAN_MEM_ACCESS_FUNC_PC(type, func, size) \
  void __tsan_##type(void *addr, void *pc) {      \
    ThreadState *thr = cur_thread();              \
    DCHECK_EQ(thr->in_interceptor_count, 0);      \
    func(thr, (uptr)pc, (uptr)addr, size);        \
  }

TSAN_MEM_ACCESS_FUNC(read1, MemoryRead, kSizeLog1)
TSAN_MEM_ACCESS_FUNC(read2, MemoryRead, kSizeLog2)
TSAN_MEM_ACCESS_FUNC(read4, MemoryRead, kSizeLog4)
TSAN_MEM_ACCESS_FUNC(read8, MemoryRead, kSizeLog8)
TSAN_MEM_ACCESS_FUNC(write1, MemoryWrite, kSizeLog1)
TSAN_MEM_ACCESS_FUNC(write2, MemoryWrite, kSizeLog2)
TSAN_MEM_ACCESS_FUNC(write4, MemoryWrite, kSizeLog4)
TSAN_MEM_ACCESS_FUNC(write8, MemoryWrite, kSizeLog8)
TSAN_MEM_ACCESS_FUNC_PC(read1_pc, MemoryRead, kSizeLog1)
TSAN_MEM_ACCESS_FUNC_PC(read2_pc, MemoryRead, kSizeLog2)
TSAN_MEM_ACCESS_FUNC_PC(read4_pc, MemoryRead, kSizeLog4)
TSAN_MEM_ACCESS_FUNC_PC(read8_pc, MemoryRead, kSizeLog8)
TSAN_MEM_ACCESS_FUNC_PC(write1_pc, MemoryWrite, kSizeLog1)
TSAN_MEM_ACCESS_FUNC_PC(write2_pc, MemoryWrite, kSizeLog2)
TSAN_MEM_ACCESS_FUNC_PC(write4_pc, MemoryWrite, kSizeLog4)
TSAN_MEM_ACCESS_FUNC_PC(write8_pc, MemoryWrite, kSizeLog8)

void __tsan_vptr_update(void **vptr_p, void *new_val) {
  CHECK_EQ(sizeof(vptr_p), 8);
  if (*vptr_p != new_val) {
    ThreadState *thr = cur_thread();
    thr->is_vptr_access = true;
    MemoryWrite(thr, CALLERPC, (uptr)vptr_p, kSizeLog8);
    thr->is_vptr_access = false;
  }
}

void __tsan_vptr_read(void **vptr_p) {
  CHECK_EQ(sizeof(vptr_p), 8);
  ThreadState *thr = cur_thread();
  thr->is_vptr_access = true;
  MemoryRead(thr, CALLERPC, (uptr)vptr_p, kSizeLog8);
  thr->is_vptr_access = false;
}

void __tsan_func_entry(void *pc) {
  FuncEntry(cur_thread(), (uptr)pc);
}

void __tsan_func_exit() {
  FuncExit(cur_thread());
}

void __tsan_read_range(void *addr, uptr size) {
  MemoryAccessRange(cur_thread(), CALLERPC, (uptr)addr, size, false);
}

void __tsan_write_range(void *addr, uptr size) {
  MemoryAccessRange(cur_thread(), CALLERPC, (uptr)addr, size, true);
}
