@ RUN: llvm-mc -triple armv7-eabi -filetype obj -o - %s | llvm-readobj -r - \
@ RUN:     | FileCheck %s
@ RUN: llvm-mc -triple thumbv7-eabi -filetype obj -o - %s | llvm-readobj -r - \
@ RUN:     | FileCheck %s

	.syntax unified

	.byte abs8_0 -128
	.byte abs8_1 +255

@ CHECK: Relocations {
@ CHECK:   Section (2) .rel.text {
@ CHECK:     0x0 R_ARM_ABS8 abs8_0 0x0
@ CHECK:     0x1 R_ARM_ABS8 abs8_1 0x0
@ CHECK:   }
@ CHECK: }
