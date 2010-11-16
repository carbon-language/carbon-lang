//===-- unorddf2vfp.S - Implement unorddf2vfp ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"

//
// extern int __unorddf2vfp(double a, double b);
//
// Returns one iff a or b is NaN
// Uses Darwin calling convention where double precision arguments are passsed 
// like in GPR pairs.
//
	.align 2
DEFINE_COMPILERRT_FUNCTION(__unorddf2vfp)
	fmdrr	d6, r0, r1	// load r0/r1 pair in double register
	fmdrr	d7, r2, r3	// load r2/r3 pair in double register
	fcmpd	d6, d7		
	fmstat
	movvs	r0, #1      // set result register to 1 if "overflow" (any NaNs)
	movvc	r0, #0
	bx	lr
