# RUN: llvm-mc -triple=powerpc-unknown-linux-gnu -filetype=obj %s | \
# RUN: llvm-readobj -r | FileCheck %s
	.section .text

	.globl foo
	.type foo,@function
	.align 2
foo:
	bl printf@plt
.LC1:
	.size foo, . - foo

# CHECK:      Relocations [
# CHECK-NEXT:   Section (2) .rela.text {
# CHECK-NEXT:     0x0 R_PPC_PLTREL24 printf 0x0
# CHECK-NEXT:   }
# CHECK-NEXT: ]
