//===-- negsf2vfp.S - Implement negsf2vfp ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"

//
// extern float __negsf2vfp(float a);
//
// Returns the negation of a single precision floating point numbers using the 
// Darwin calling convention where single arguments are passsed like 32-bit ints
//
	.globl ___negsf2vfp
___negsf2vfp:
	eor	r0, r0, #-2147483648	// flip sign bit on float in r0
	bx	lr
