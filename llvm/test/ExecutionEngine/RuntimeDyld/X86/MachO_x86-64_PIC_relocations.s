# RUN: llvm-mc -triple=x86_64-apple-macosx10.9 -relocation-model=pic -filetype=obj -o %t.o %s
# RUN: llvm-rtdyld -triple=x86_64-apple-macosx10.9 -verify -check=%s %t.o
# RUN: rm %t.o

        .section	__TEXT,__text,regular,pure_instructions
	.globl	foo
	.align	4, 0x90
foo:
        retq

	.globl	main
	.align	4, 0x90
main:
# Test PC-rel branch.
# rtdyld-check: decode_operand(insn1, 0) = foo - next_pc(insn1)
insn1:
        callq	foo

# Test PC-rel signed.
# rtdyld-check: decode_operand(insn2, 4) = x - next_pc(insn2)
insn2:
	movl	x(%rip), %eax
	movl	$0, %eax
	retq

        .section	__DATA,__data
	.globl	x
	.align	2
x:
        .long   5

.subsections_via_symbols
