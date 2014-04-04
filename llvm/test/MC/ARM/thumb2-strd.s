@ RUN: not llvm-mc -triple=armv7-linux-gnueabi %s 2>&1 | FileCheck %s
.text
.thumb

@ CHECK: error: invalid operand for instruction
@ CHECK: error: invalid operand for instruction
@ CHECK: error: invalid operand for instruction
strd r12, SP, [r0, #256]
strd r12, SP, [r0, #256]!
strd r12, SP, [r0], #256
