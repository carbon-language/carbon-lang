@ RUN: llvm-mc -triple armv7-eabi -filetype obj -o - %s | llvm-readobj -r - \
@ RUN:     | FileCheck %s
@ RUN: llvm-mc -triple thumbv7-eabi -filetype obj -o - %s | llvm-readobj -r - \
@ RUN:     | FileCheck %s

	.syntax unified

	.short abs16_0 -32768
	.short abs16_1 +65535

@ CHECK: Relocations {
@ CHECK:   Section (2) .rel.text {
@ CHECK:     0x0 R_ARM_ABS16 abs16_0 0x0
@ CHECK:     0x2 R_ARM_ABS16 abs16_1 0x0
@ CHECK:   }
@ CHECK: }

