/*===-- umodsi3.S - 32-bit unsigned integer modulus -----------------------===//
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is dual licensed under the MIT and the University of Illinois Open
 * Source Licenses. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===//
 *
 * This file implements the __umodsi3 (32-bit unsigned integer modulus)
 * function for the ARM 32-bit architecture.
 *
 *===----------------------------------------------------------------------===*/

#include "../assembly.h"

	.syntax unified
	.text
#if __ARM_ARCH_ISA_THUMB == 2
	.thumb
#endif

	.p2align 2
DEFINE_COMPILERRT_FUNCTION(__umodsi3)
#if __ARM_ARCH_EXT_IDIV__
	tst     r1, r1
	beq     LOCAL_LABEL(divby0)
	mov 	r3, r0
	udiv	r0, r3, r1
	mls 	r1, r0, r1, r3
	str 	r1, [r2]
	bx  	lr
#else
	cmp	r1, #1
	bcc	LOCAL_LABEL(divby0)
	IT eq
	moveq	r0, #0
	JMPc(lr, eq)
	cmp	r0, r1
	JMPc(lr, cc)
	/*
	 * Implement division using binary long division algorithm.
	 *
	 * r0 is the numerator, r1 the denominator.
	 *
	 * For ARM:
	 * The code before JMP computes the correct shift I, so that
	 * r0 and (r1 << I) have the highest bit set in the same position.
	 * At the time of JMP, ip := .Ldiv0block - 8 * I.
	 * This depends on the fixed instruction size of block.
	 *
	 * For Thumb:
	 * Uses a jumptable to jump to the appropriate block.
	 *
	 * block(shift) implements the test-and-update-quotient core.
	 * It assumes (r0 << shift) can be computed without overflow and
	 * that (r0 << shift) < 2 * r1. The quotient is stored in r3.
	 */

#  ifdef __ARM_FEATURE_CLZ
	clz	ip, r0
	clz	r3, r1
	/* r0 >= r1 implies clz(r0) <= clz(r1), so ip <= r3. */
	sub	r3, r3, ip
#if __ARM_ARCH_ISA_THUMB == 2
	tbb	[pc, r3]
LOCAL_LABEL(JT):
	.byte (LOCAL_LABEL( 0) - LOCAL_LABEL(JT)) / 2
	.byte (LOCAL_LABEL( 1) - LOCAL_LABEL(JT)) / 2
	.byte (LOCAL_LABEL( 2) - LOCAL_LABEL(JT)) / 2
	.byte (LOCAL_LABEL( 3) - LOCAL_LABEL(JT)) / 2
	.byte (LOCAL_LABEL( 4) - LOCAL_LABEL(JT)) / 2
	.byte (LOCAL_LABEL( 5) - LOCAL_LABEL(JT)) / 2
	.byte (LOCAL_LABEL( 6) - LOCAL_LABEL(JT)) / 2
	.byte (LOCAL_LABEL( 7) - LOCAL_LABEL(JT)) / 2
	.byte (LOCAL_LABEL( 8) - LOCAL_LABEL(JT)) / 2
	.byte (LOCAL_LABEL( 9) - LOCAL_LABEL(JT)) / 2
	.byte (LOCAL_LABEL(10) - LOCAL_LABEL(JT)) / 2
	.byte (LOCAL_LABEL(11) - LOCAL_LABEL(JT)) / 2
	.byte (LOCAL_LABEL(12) - LOCAL_LABEL(JT)) / 2
	.byte (LOCAL_LABEL(13) - LOCAL_LABEL(JT)) / 2
	.byte (LOCAL_LABEL(14) - LOCAL_LABEL(JT)) / 2
	.byte (LOCAL_LABEL(15) - LOCAL_LABEL(JT)) / 2
	.byte (LOCAL_LABEL(16) - LOCAL_LABEL(JT)) / 2
	.byte (LOCAL_LABEL(17) - LOCAL_LABEL(JT)) / 2
	.byte (LOCAL_LABEL(18) - LOCAL_LABEL(JT)) / 2
	.byte (LOCAL_LABEL(19) - LOCAL_LABEL(JT)) / 2
	.byte (LOCAL_LABEL(20) - LOCAL_LABEL(JT)) / 2
	.byte (LOCAL_LABEL(21) - LOCAL_LABEL(JT)) / 2
	.byte (LOCAL_LABEL(22) - LOCAL_LABEL(JT)) / 2
	.byte (LOCAL_LABEL(23) - LOCAL_LABEL(JT)) / 2
	.byte (LOCAL_LABEL(24) - LOCAL_LABEL(JT)) / 2
	.byte (LOCAL_LABEL(25) - LOCAL_LABEL(JT)) / 2
	.byte (LOCAL_LABEL(26) - LOCAL_LABEL(JT)) / 2
	.byte (LOCAL_LABEL(27) - LOCAL_LABEL(JT)) / 2
	.byte (LOCAL_LABEL(28) - LOCAL_LABEL(JT)) / 2
	.byte (LOCAL_LABEL(29) - LOCAL_LABEL(JT)) / 2
	.byte (LOCAL_LABEL(30) - LOCAL_LABEL(JT)) / 2
	.byte (LOCAL_LABEL(31) - LOCAL_LABEL(JT)) / 2
#else
	adr	ip, LOCAL_LABEL(0)
	sub	ip, ip, r3, lsl #3
	bx	ip
#endif
#  else
#if __ARM_ARCH_ISA_THUMB == 2
#error unsupported configuration
#endif
	mov	r2, r0
	adr	ip, LOCAL_LABEL(0)

	lsr	r3, r2, #16
	cmp	r3, r1
	movhs	r2, r3
	subhs	ip, ip, #(16 * 8)

	lsr	r3, r2, #8
	cmp	r3, r1
	movhs	r2, r3
	subhs	ip, ip, #(8 * 8)

	lsr	r3, r2, #4
	cmp	r3, r1
	movhs	r2, r3
	subhs	ip, #(4 * 8)

	lsr	r3, r2, #2
	cmp	r3, r1
	movhs	r2, r3
	subhs	ip, ip, #(2 * 8)

	/* Last block, no need to update r2 or r3. */
	cmp	r1, r2, lsr #1
	subls	ip, ip, #(1 * 8)

	JMP(ip)
#  endif

#define	IMM	#

#define block(shift)                                                           \
LOCAL_LABEL(shift):                                                            \
	cmp r0, r1, lsl IMM shift;                                             \
	IT hs;                                                                 \
	subhs r0, r0, r1, lsl IMM shift

	block(31)
	block(30)
	block(29)
	block(28)
	block(27)
	block(26)
	block(25)
	block(24)
	block(23)
	block(22)
	block(21)
	block(20)
	block(19)
	block(18)
	block(17)
	block(16)
	block(15)
	block(14)
	block(13)
	block(12)
	block(11)
	block(10)
	block(9)
	block(8)
	block(7)
	block(6)
	block(5)
	block(4)
	block(3)
	block(2)
	block(1)
	block(0)
	JMP(lr)
#endif /* __ARM_ARCH_EXT_IDIV__ */

LOCAL_LABEL(divby0):
	mov	r0, #0
#ifdef __ARM_EABI__
	b	__aeabi_idiv0
#else
	JMP(lr)
#endif

END_COMPILERRT_FUNCTION(__umodsi3)
