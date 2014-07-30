# RUN: llvm-mc -triple=armv7s-apple-ios7.0.0 -relocation-model=pic -filetype=obj -o %T/foo.o %s
# RUN: llvm-rtdyld -triple=armv7s-apple-ios7.0.0 -verify -check=%s %/T/foo.o

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

# Check stub generation by referencing a common symbol, 'baz'. Check both the
# Content of the stub, and the reference to the stub.
# Stub should contain '0xe51ff004' (ldr pc, [pc, #-4]), followed by the target.
#
# rtdyld-check: *{4}(stub_addr(foo.o, __text, baz)) = 0xe51ff004
# rtdyld-check: *{4}(stub_addr(foo.o, __text, baz) + 4) = baz
#
# rtdyld-check: decode_operand(insn3, 0) = stub_addr(foo.o, __text, baz) - (insn3 + 8)
insn3:
        bl      baz
	bx	lr

	.globl	foo
	.align	2
foo:
	bx	lr

        .comm   baz, 4, 2

.subsections_via_symbols
