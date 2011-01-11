@ RUN: llvm-mc %s -triple=armv7-linux-gnueabi | FileCheck -check-prefix=ASM %s
	.syntax unified
	.text
	.globl	barf
	.align	2
	.type	barf,%function
barf:                                   @ @barf
@ BB#0:                                 @ %entry
	movw	r0, :lower16:GOT-(.LPC0_2+8)
	movt	r0, :upper16:GOT-(.LPC0_2+16)
.LPC0_2:
@ ASM:          movw    r0, :lower16:GOT-(.LPC0_2+8)
@ ASM-NEXT:     movt    r0, :upper16:GOT-(.LPC0_2+16)

