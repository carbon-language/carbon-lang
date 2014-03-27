@ RUN: not llvm-mc -triple=armv7-linux-gnuabi -filetype=obj < %s 2>&1 | FileCheck %s

.text
    cmp r2, #(l2 - l1) >> 6
@ CHECK: error: invalid operand for instruction

l1:
l2:
