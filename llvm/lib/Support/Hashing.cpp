//===-- llvm/ADT/Hashing.cpp - Utilities for hashing ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Hashing.h"

namespace llvm {

// Add a possibly unaligned sequence of bytes.
void GeneralHash::addUnaligned(const uint8_t *I, const uint8_t *E) {
  ptrdiff_t Length = E - I;
  if ((uintptr_t(I) & 3) == 0) {
    while (Length > 3) {
      mix(*reinterpret_cast<const uint32_t *>(I));
      I += 4;
      Length -= 4;
    }
  } else {
    while (Length > 3) {
      mix(
        uint32_t(I[0]) +
        (uint32_t(I[1]) << 8) +
        (uint32_t(I[2]) << 16) +
        (uint32_t(I[3]) << 24));
      I += 4;
      Length -= 4;
    }
  }

  if (Length & 3) {
    uint32_t Data = 0;
    switch (Length & 3) {
      case 3: Data |= uint32_t(I[2]) << 16;   // fall through
      case 2: Data |= uint32_t(I[1]) << 8;    // fall through
      case 1: Data |= uint32_t(I[0]); break;
    }
    mix(Data);
  }
}

}
