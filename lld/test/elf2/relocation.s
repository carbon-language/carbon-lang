// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t
// RUN: lld -flavor gnu2 %t -o %t2
// RUN: llvm-objdump -t -d %t2 | FileCheck %s
// REQUIRES: x86


.section       .text,"ax",@progbits,unique,1
.global _start
_start:
  call lulz

.section       .text,"ax",@progbits,unique,2
.zero 4
.global lulz
lulz:

.global bar
.text
bar:
  movl $bar, %edx // R_X86_64_32

// R_X86_64_32
// CHECK: bar:
// CHECK:    1000: ba 00 10 00 00 movl $4096, %edx

// CHECK: e8 04 00 00 00  callq   4

// Also check that symbols match.
// CHECK: 0000000000001000         .text           00000000 bar
