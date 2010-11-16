//===-------- modsi3.S - Implement modsi3 ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"

//
// extern int32_t __modsi3(int32_t a, int32_t b);
//
// Returns the remainder when dividing two 32-bit signed integers.
// Conceptually, the function is: { return a - (a / b) * b; }
// But if you write that in C, llvm compiles it to a call to __modsi3...
//
	.align 2
DEFINE_COMPILERRT_FUNCTION(__modsi3)
	push	{r4, r5, r7, lr}
	add	r7, sp, #8	// set stack frame
	mov	r5, r0		// save a
	mov	r4, r1		// save b
	bl      ___divsi3	// compute a/b
#if __ARM_ARCH_7A__
	mls     r0, r4, r0, r5  // mulitple result * b and subtract from a
#else	
	// before armv7, does not have "mls" instruction
	mul	r3, r0, r4	// multiple result * b
	sub     r0, r5, r3      // a - result
#endif
	pop     {r4, r5, r7, pc}
	


