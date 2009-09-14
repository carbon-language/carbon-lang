//===-- fixdfsivfp.S - Implement fixdfsivfp -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


//
// extern int __fixdfsivfp(double a);
//
// Converts double precision float to a 32-bit int rounding towards zero.
// Uses Darwin calling convention where a double precision parameter is 
// passed in GPR register pair.
//
	.globl ___fixdfsivfp
___fixdfsivfp:
	fmdrr	d7, r0, r1    // load double register from R0/R1
	ftosizd	s15, d7       // convert double to 32-bit int into s15
	fmrs	r0, s15	      // move s15 to result register
	bx	lr
