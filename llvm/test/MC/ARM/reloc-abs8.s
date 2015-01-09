@ RUN: llvm-mc -triple armv7-eabi -filetype obj -o - %s | llvm-objdump -r - \
@ RUN:     | FileCheck %s
@ RUN: llvm-mc -triple thumbv7-eabi -filetype obj -o - %s | llvm-objdump -r - \
@ RUN:     | FileCheck %s

	.syntax unified

	.byte abs8_0 -128
	.byte abs8_1 +255

@ CHECK: 0 R_ARM_ABS8 abs8_0
@ CHECK: 1 R_ARM_ABS8 abs8_1
