//===-- tsan_allocator-------------------------------------------*- C++ -*-===//
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
#include "tsan_allocator.h"

// Provisional implementation.
extern "C" void *__libc_malloc(__tsan::uptr size);
extern "C" void __libc_free(void *ptr);

namespace __tsan {

u64 kBlockMagic = 0x6A6CB03ABCEBC041ull;

void AllocInit() {
}

void *Alloc(uptr sz) {
  void *p = __libc_malloc(sz + sizeof(u64));
  ((u64*)p)[0] = kBlockMagic;
  return (char*)p + sizeof(u64);
}

void Free(void *p) {
  CHECK_NE(p, (char*)0);
  p = (char*)p - sizeof(u64);
  CHECK_EQ(((u64*)p)[0], kBlockMagic);
  ((u64*)p)[0] = 0;
  __libc_free(p);
}

void *AllocBlock(void *p) {
  CHECK_NE(p, (void*)0);
  u64 *pp = (u64*)((uptr)p & ~0x7);
  for (; pp[0] != kBlockMagic; pp--) {}
  return pp + 1;
}

}  // namespace __tsan
