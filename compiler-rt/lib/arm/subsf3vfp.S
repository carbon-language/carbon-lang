//===-- subsf3vfp.S - Implement subsf3vfp ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"

//
// extern float __subsf3vfp(float a, float b);
//
// Returns the difference between two single precision floating point numbers 
// using the Darwin calling convention where single arguments are passsed
// like 32-bit ints.
//
DEFINE_COMPILERRT_FUNCTION(__subsf3vfp)
	fmsr	s14, r0		// move first param from r0 into float register
	fmsr	s15, r1		// move second param from r1 into float register
	fsubs	s14, s14, s15
	fmrs	r0, s14		// move result back to r0
	bx	lr
