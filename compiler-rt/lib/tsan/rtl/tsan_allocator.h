//===-- tsan_allocator.h ----------------------------------------*- C++ -*-===//
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
#ifndef TSAN_ALLOCATOR_H
#define TSAN_ALLOCATOR_H

#include "tsan_defs.h"

namespace __tsan {

void AllocInit();
void *Alloc(uptr sz);
void Free(void *p);  // Does not accept NULL.
// Given the pointer p into a valid allocated block,
// returns a pointer to the beginning of the block.
void *AllocBlock(void *p);

}  // namespace __tsan

#endif  // TSAN_ALLOCATOR_H
