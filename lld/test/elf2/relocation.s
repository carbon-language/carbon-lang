// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t
// RUN: lld -flavor gnu2 %t -o %t2
// RUN: llvm-objdump -d %t2 | FileCheck %s
// REQUIRES: x86


.section       .text,"ax",@progbits,unique,1
.global _start
_start:
  call lulz

.section       .text,"ax",@progbits,unique,2
.zero 4
.global lulz
lulz:

// CHECK: e8 04 00 00 00  callq   4
