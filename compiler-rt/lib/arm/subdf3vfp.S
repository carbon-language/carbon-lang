//===-- subdf3vfp.S - Implement subdf3vfp ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"

//
// extern double __subdf3vfp(double a, double b);
//
// Returns difference between two double precision floating point numbers using 
// the Darwin calling convention where double arguments are passsed in GPR pairs
//
		.globl ___subdf3vfp
___subdf3vfp:
	fmdrr	d6, r0, r1		// move first param from r0/r1 pair into d6
	fmdrr	d7, r2, r3		// move second param from r2/r3 pair into d7
	fsubd	d6, d6, d7		
	fmrrd	r0, r1, d6		// move result back to r0/r1 pair
	bx	lr
