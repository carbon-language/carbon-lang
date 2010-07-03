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
	.align 2
DEFINE_COMPILERRT_FUNCTION(__bswapdi2)
#if __ARM_ARCH_5TEJ__ || __ARM_ARCH_4T__
	// before armv6 does not have "rev" instruction
	stmfd	sp!, {r7, lr}
	mov	r7, sp
	mov	r2, #255, 24
	and	r3, r2, r1, lsr #8
	mov	r12, #255, 16
	and	lr, r12, r1, lsl #8
	orr	r3, r3, r1, lsr #24
	orr	r1, lr, r1, lsl #24
	and	r2, r2, r0, lsr #8
	orr	r3, r1, r3
	orr	r1, r2, r0, lsr #24
	and	r2, r12, r0, lsl #8
	orr	r0, r2, r0, lsl #24
	orr	r1, r0, r1
	mov	r0, r3
	ldmfd	sp!, {r7, pc}
#else
	rev	r2, r1		// reverse bytes in high 32-bits into temp2
	rev	r3, r0		// reverse bytes in low 32-bit into temp3
	mov	r0, r2		// set low 32-bits of result to temp2
	mov	r1, r3		// set high 32-bits of result to temp3
	bx	lr
#endif
