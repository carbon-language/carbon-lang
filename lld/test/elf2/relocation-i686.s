// RUN: llvm-mc -filetype=obj -triple=i686-pc-linux %s -o %t
// RUN: lld -flavor gnu2 %t -o %t2
// RUN: llvm-objdump -d %t2 | FileCheck %s
// REQUIRES: x86

.global _start
_start:

.section       .R_386_32,"ax",@progbits
.global R_386_32
R_386_32:
  movl $R_386_32, %edx

// CHECK: Disassembly of section .R_386_32:
// CHECK-NEXT: R_386_32:
// CHECK-NEXT:  11000: {{.*}} movl $69632, %edx
