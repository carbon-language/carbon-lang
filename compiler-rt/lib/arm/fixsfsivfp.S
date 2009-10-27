//===-- fixsfsivfp.S - Implement fixsfsivfp -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"

//
// extern int __fixsfsivfp(float a);
//
// Converts single precision float to a 32-bit int rounding towards zero.
// Uses Darwin calling convention where a single precision parameter is 
// passed in a GPR..
//
DEFINE_COMPILERRT_FUNCTION(__fixsfsivfp)
	fmsr	s15, r0      // load float register from R0
	ftosizs	s15, s15     // convert single to 32-bit int into s15
	fmrs	r0, s15	     // move s15 to result register
	bx	lr
