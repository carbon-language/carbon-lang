//===-- extendsfdf2vfp.S - Implement extendsfdf2vfp -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"

//
// extern double __extendsfdf2vfp(float a);
//
// Converts single precision float to double precision result.
// Uses Darwin calling convention where a single precision parameter is 
// passed in a GPR and a double precision result is returned in R0/R1 pair.
//
DEFINE_COMPILERRT_FUNCTION(__extendsfdf2vfp)
	fmsr	s15, r0      // load float register from R0
	fcvtds	d7, s15      // convert single to double
	fmrrd	r0, r1, d7   // return result in r0/r1 pair
	bx	lr
