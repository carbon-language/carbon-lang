// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

// di_int __ashldi3(di_int input, int count);

// This routine has some extra memory traffic, loading the 64-bit input via two
// 32-bit loads, then immediately storing it back to the stack via a single 64-bit
// store.  This is to avoid a write-small, read-large stall.
// However, if callers of this routine can be safely assumed to store the argument
// via a 64-bt store, this is unnecessary memory traffic, and should be avoided.
// It can be turned off by defining the TRUST_CALLERS_USE_64_BIT_STORES macro.

#ifdef __i386__
#ifdef __SSE2__

.text
.align 4
.globl ___ashldi3
___ashldi3:
	movd	  12(%esp),		%xmm2	// Load count
#ifndef TRUST_CALLERS_USE_64_BIT_STORES
	movd	   4(%esp),		%xmm0
	movd	   8(%esp),		%xmm1
	punpckldq	%xmm1,		%xmm0	// Load input
#else
	movq	   4(%esp),		%xmm0	// Load input
#endif
	psllq		%xmm2,		%xmm0	// shift input by count
	movd		%xmm0,		%eax
	psrlq		$32,		%xmm0
	movd		%xmm0,		%edx
	ret

#else // Use GPRs instead of SSE2 instructions, if they aren't available.

.text
.align 4
.globl ___ashldi3
___ashldi3:
	movl	  12(%esp),		%ecx	// Load count
	movl	   8(%esp),		%edx	// Load high
	movl	   4(%esp),		%eax	// Load low
	
	testl		$0x20,		%ecx	// If count >= 32
	jnz			2f					//    goto 2
	testl		$0x1f,		%ecx	// If count == 0
	jz			1f					//    goto 1
	
	pushl		%ebx
	movl		%eax,		%ebx	// copy low
	shll		%cl,		%eax	// left shift low by count
	shll		%cl,		%edx	// left shift high by count
	neg			%cl
	shrl		%cl,		%ebx	// right shift low by 32 - count
	orl			%ebx,		%edx	// or the result into the high word
	popl		%ebx
1:	ret
	
2:	movl		%eax,		%edx	// Move low to high
	xorl		%eax,		%eax	// clear low
	shll		%cl,		%edx	// shift high by count - 32
	ret
	
#endif // __SSE2__
#endif // __i386__
