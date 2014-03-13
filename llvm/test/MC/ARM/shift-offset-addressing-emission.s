@ RUN: llvm-mc -triple armv7-elf -filetype obj -o - %s \
@ RUN:   | llvm-objdump -disassemble -no-show-raw-insn - | FileCheck %s

	cmp r0, #(.L2 - .L1)
.L1:
.L2:

@ CHECK: 0:	cmp	r0, #0

