//===-- Support/MathExtras.h - Useful math functions ------------*- C++ -*-===//
//
// This file contains some functions that are useful for math stuff.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_MATHEXTRAS_H
#define SUPPORT_MATHEXTRAS_H

#include "Support/DataTypes.h"

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

#endif
