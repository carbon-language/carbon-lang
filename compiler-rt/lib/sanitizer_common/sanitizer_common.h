//===-- sanitizer_common.h --------------------------------------*- C++ -*-===//
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
// It declares common functions and classes that are used in both runtimes.
// Implementation of some functions are provided in sanitizer_common, while
// others must be defined by run-time library itself.
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_COMMON_H
#define SANITIZER_COMMON_H

#include "sanitizer_internal_defs.h"

namespace __sanitizer {

// Constants.
const uptr kWordSize = __WORDSIZE / 8;
const uptr kWordSizeInBits = 8 * kWordSize;
const uptr kPageSizeBits = 12;
const uptr kPageSize = 1UL << kPageSizeBits;

int GetPid();
void RawWrite(const char *buffer);

// Memory management
void *MmapOrDie(uptr size, const char *mem_type);
void UnmapOrDie(void *addr, uptr size);

void Printf(const char *format, ...);
int SNPrintf(char *buffer, uptr length, const char *format, ...);
void Report(const char *format, ...);

// Bit twiddling.
inline bool IsPowerOfTwo(uptr x) {
  return (x & (x - 1)) == 0;
}
inline uptr RoundUpTo(uptr size, uptr boundary) {
  // FIXME: Use CHECK here.
  RAW_CHECK(IsPowerOfTwo(boundary));
  return (size + boundary - 1) & ~(boundary - 1);
}

}  // namespace __sanitizer

#endif  // SANITIZER_COMMON_H
