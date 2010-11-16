//===-- unordsf2vfp.S - Implement unordsf2vfp -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"

//
// extern int __unordsf2vfp(float a, float b);
//
// Returns one iff a or b is NaN
// Uses Darwin calling convention where single precision arguments are passsed 
// like 32-bit ints
//
	.align 2
DEFINE_COMPILERRT_FUNCTION(__unordsf2vfp)
	fmsr	s14, r0     // move from GPR 0 to float register
	fmsr	s15, r1	    // move from GPR 1 to float register
	fcmps	s14, s15
	fmstat
	movvs	r0, #1      // set result register to 1 if "overflow" (any NaNs)
	movvc	r0, #0
	bx	lr

