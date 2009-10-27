//===-- truncdfsf2vfp.S - Implement truncdfsf2vfp -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"

//
// extern float __truncdfsf2vfp(double a);
//
// Converts double precision float to signle precision result.
// Uses Darwin calling convention where a double precision parameter is 
// passed in a R0/R1 pair and a signle precision result is returned in R0.
//
	.globl ___truncdfsf2vfp
___truncdfsf2vfp:
	fmdrr	d7, r0, r1   // load double from r0/r1 pair
	fcvtsd	s15, d7      // convert double to single (trucate precision)
	fmrs	r0, s15      // return result in r0
	bx	lr
