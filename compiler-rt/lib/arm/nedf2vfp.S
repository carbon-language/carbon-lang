//===-- nedf2vfp.S - Implement nedf2vfp -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"

//
// extern double __nedf2vfp(double a, double b);
//
// Returns zero if a and b are unequal and neither is NaN.
// Uses Darwin calling convention where double precision arguments are passsed 
// like in GPR pairs.
//
	.globl ___nedf2vfp
___nedf2vfp:
	fmdrr	d6, r0, r1	// load r0/r1 pair in double register
	fmdrr	d7, r2, r3	// load r2/r3 pair in double register
	fcmpd	d6, d7		
	fmstat
	movne	r0, #1		// set result register to 0 if unequal
	moveq	r0, #0
	bx	lr
