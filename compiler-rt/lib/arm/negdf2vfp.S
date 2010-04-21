//===-- negdf2vfp.S - Implement negdf2vfp ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"

//
// extern double __negdf2vfp(double a, double b);
//
// Returns the negation a double precision floating point numbers using the 
// Darwin calling convention where double arguments are passsed in GPR pairs.
//
	.align 2
DEFINE_COMPILERRT_FUNCTION(__negdf2vfp)
	eor	r1, r1, #-2147483648	// flip sign bit on double in r0/r1 pair
	bx	lr
