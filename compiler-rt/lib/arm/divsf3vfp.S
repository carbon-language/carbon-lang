//===-- divsf3vfp.S - Implement divsf3vfp ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"

//
// extern float __divsf3vfp(float a, float b);
//
// Divides two single precision floating point numbers using the Darwin
// calling convention where single arguments are passsed like 32-bit ints.
//
	.align 2
DEFINE_COMPILERRT_FUNCTION(__divsf3vfp)
	fmsr	s14, r0		// move first param from r0 into float register
	fmsr	s15, r1		// move second param from r1 into float register
	fdivs	s13, s14, s15
	fmrs	r0, s13		// move result back to r0
	bx	lr
