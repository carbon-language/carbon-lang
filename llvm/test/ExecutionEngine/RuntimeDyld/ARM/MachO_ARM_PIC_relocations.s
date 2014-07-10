# RUN: llvm-mc -triple=armv7s-apple-ios7.0.0 -relocation-model=pic -filetype=obj -o %t.o %s
# RUN: llvm-rtdyld -triple=armv7s-apple-ios7.0.0 -verify -check=%s %t.o
# RUN: rm %t.o

	.syntax unified
	.section	__TEXT,__text,regular,pure_instructions
	.globl	bar
	.align	2
bar:
# Check lower 16-bits of section difference relocation
# rtdyld-check: decode_operand(insn1, 1) = (foo-(nextPC+8))[15:0]
insn1:
	movw	r0, :lower16:(foo-(nextPC+8))
# Check upper 16-bits of section difference relocation
# rtdyld-check: decode_operand(insn2, 2) = (foo-(nextPC+8))[31:16]
insn2:
	movt	r0, :upper16:(foo-(nextPC+8))
nextPC:
	add	r0, pc, r0
	bx	lr

	.globl	foo
	.align	2
foo:
	bx	lr

.subsections_via_symbols
