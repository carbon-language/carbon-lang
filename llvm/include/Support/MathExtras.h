// $Id$ -*-c++-*-
//***************************************************************************
// File:
//	MathExtras.h
// 
// Purpose:
//	
// History:
//	8/25/01	 -  Vikram Adve  -  Created
//**************************************************************************/

#ifndef LLVM_SUPPORT_MATH_EXTRAS_H
#define LLVM_SUPPORT_MATH_EXTRAS_H

#include <Support/DataTypes.h>

inline unsigned
log2(uint64_t C)
{
  unsigned getPow;
  for (getPow = 0; C > 1; getPow++)
    C = C >> 1;
  return getPow;
}

inline bool
IsPowerOf2(int64_t C, unsigned& getPow)
{
  if (C < 0)
    C = -C;
  bool isPowerOf2 = C > 0 && (C == (C & ~(C - 1)));
  if (isPowerOf2)
    getPow = log2(C);
  
  return isPowerOf2;
}

#endif /*LLVM_SUPPORT_MATH_EXTRAS_H*/
