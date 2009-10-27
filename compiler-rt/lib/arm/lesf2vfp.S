//===-- lesf2vfp.S - Implement lesf2vfp -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"

//
// extern int __lesf2vfp(float a, float b);
//
// Returns one iff a <= b and neither is NaN.
// Uses Darwin calling convention where single precision arguments are passsed 
// like 32-bit ints
//
DEFINE_COMPILERRT_FUNCTION(__lesf2vfp)
	fmsr	s14, r0     // move from GPR 0 to float register
	fmsr	s15, r1     // move from GPR 1 to float register
	fcmps	s14, s15
	fmstat
	movls	r0, #1      // set result register to 1 if equal
	movhi	r0, #0
	bx	lr

