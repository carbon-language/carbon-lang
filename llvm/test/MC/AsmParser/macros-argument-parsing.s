# RUN: llvm-mc -triple i386 -filetype asm -o - %s | FileCheck %s

	.macro	it, cond
	.endm

	it ne
	.long 1

# CHECK: .long 1

