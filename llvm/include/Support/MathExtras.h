//===-- Support/MathExtras.h - Useful math functions ------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file contains some functions that are useful for math stuff.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_MATHEXTRAS_H
#define SUPPORT_MATHEXTRAS_H

#include "Support/DataTypes.h"

namespace llvm {

inline unsigned log2(uint64_t C) {
  unsigned getPow;
  for (getPow = 0; C > 1; ++getPow)
    C >>= 1;
  return getPow;
}

inline bool isPowerOf2(int64_t C, unsigned &getPow) {
  if (C < 0) C = -C;
  if (C > 0 && C == (C & ~(C - 1))) {
    getPow = log2((uint64_t)C);
    return true;
  }

  return false;
}

} // End llvm namespace

#endif
