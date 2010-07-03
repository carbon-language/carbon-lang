//===------- bswapsi2 - Implement bswapsi2 --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"

//
// extern uint32_t __bswapsi2(uint32_t);
//
// Reverse all the bytes in a 32-bit integer.
//
	.align 2
DEFINE_COMPILERRT_FUNCTION(__bswapsi2)
#if __ARM_ARCH_5TEJ__ || __ARM_ARCH_4T__
	// before armv6 does not have "rev" instruction
    mov r1, #0xff00
    and r2, r1, r0, lsr #8 
    and r1, r1, r0   
    orr r2, r2, r0, lsr #24
    orr r2, r2, r0, lsl #24
    orr r0, r2, r1, lsl #8
#else
	rev	r0, r0		// reverse bytes in parameter and put into result register
#endif
	bx	lr
