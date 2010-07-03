//===------- bswapdi2 - Implement bswapdi2 --------------------------------===//
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
    mov r12, #0xff00
    
    and r3,  r12, r0, lsr #8
    and r2,  r12, r0    
    orr r3,   r3, r0, lsr #24   
    orr r3,   r3, r0, lsl #24
    orr r2,   r3, r2, lsl #8  // r2 = rev(r0)
    
    and r3,  r12, r1, lsr #8
    and r0,  r12, r1
    orr r3,   r3, r1, lsr #24
    orr r3,   r3, r1, lsl #24
    orr r0,   r3, r0, lsl #8  // r0 = rev(r1)
#else
    rev r2,   r0  // r2 = rev(r0)
    rev r0,   r1  // r0 = rev(r1)
#endif
    mov r1,   r2  // r1 = r2 = rev(r0)
    bx  lr
