@ RUN: llvm-mc -triple armv7-eabi -filetype obj -o - %s | llvm-objdump -r - \
@ RUN:     | FileCheck %s
@ RUN: llvm-mc -triple thumbv7-eabi -filetype obj -o - %s | llvm-objdump -r - \
@ RUN:     | FileCheck %s

	.syntax unified

	.short abs16_0 -32768
	.short abs16_1 +65535

@ CHECK: 0 R_ARM_ABS16 abs16_0
@ CHECK: 2 R_ARM_ABS16 abs16_1

