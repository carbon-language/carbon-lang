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

#include <sys/types.h>

inline bool	IsPowerOf2	(int64_t C, unsigned& getPow);

inline
bool IsPowerOf2(int64_t C, unsigned& getPow)
{
  if (C < 0)
    C = -C;
  bool isBool = C > 0 && (C == (C & ~(C - 1)));
  if (isBool)
    for (getPow = 0; C > 1; getPow++)
      C = C >> 1;
  
  return isBool;
}

#endif /*LLVM_SUPPORT_MATH_EXTRAS_H*/
