//===-- fixunssfsivfp.S - Implement fixunssfsivfp -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"

//
// extern unsigned int __fixunssfsivfp(float a);
//
// Converts single precision float to a 32-bit unsigned int rounding towards 
// zero. All negative values become zero.
// Uses Darwin calling convention where a single precision parameter is 
// passed in a GPR..
//
	.globl ___fixunssfsivfp
___fixunssfsivfp:
	fmsr	s15, r0      // load float register from R0
	ftouizs	s15, s15     // convert single to 32-bit unsigned into s15
	fmrs	r0, s15	     // move s15 to result register
	bx	lr
