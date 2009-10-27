//===------- bswapdi2 - Implement bswapdi2 ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"

//
// extern uint64_t __bswapdi2(uint64_t);
//
// Reverse all the bytes in a 64-bit integer.
//
		.globl ___bswapdi2
___bswapdi2:
	rev	r2, r1		// reverse bytes in high 32-bits into temp2
	rev	r3, r0		// reverse bytes in low 32-bit into temp3
	mov	r0, r2		// set low 32-bits of result to temp2
	mov	r1, r3		// set high 32-bits of result to temp3
	bx	lr
