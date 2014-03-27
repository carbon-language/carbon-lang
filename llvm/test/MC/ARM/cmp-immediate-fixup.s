// PR18931
// RUN: llvm-mc %s -triple=arm-linux-gnueabi -filetype=obj -o %t
// RUN: llvm-objdump --disassemble -arch=arm %t | FileCheck %s

    .text
// CHECK: cmp r2, #1
    cmp r2, #(l2 - l1 + 4) >> 2
l1:
l2:
