//===-- IsInf.cpp ---------------------------------------------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// Platform-independent wrapper around C99 isinf(). 
//
//===----------------------------------------------------------------------===//

#include "Config/config.h"
#if HAVE_ISINF_IN_MATH_H
# include <math.h>
#elif HAVE_ISINF_IN_CMATH
# include <cmath>
#elif HAVE_STD_ISINF_IN_CMATH
# include <cmath>
using std::isinf;
#else
# error "Don't know how to get isinf()"
#endif

namespace llvm {

int IsInf (float f)  { return isinf (f); }
int IsInf (double d) { return isinf (d); }

}; // end namespace llvm;
