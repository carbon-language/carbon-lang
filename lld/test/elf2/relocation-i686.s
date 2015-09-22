// RUN: llvm-mc -filetype=obj -triple=i686-pc-linux %s -o %t
// RUN: lld -flavor gnu2 %t -o %t2
// RUN: llvm-objdump -d %t2 | FileCheck %s
// REQUIRES: x86

.global _start
_start:

.section       .R_386_32,"ax",@progbits
.global R_386_32
R_386_32:
  movl $R_386_32 + 1, %edx


.section       .R_386_PC32,"ax",@progbits,unique,1
.global R_386_PC32
R_386_PC32:
  call R_386_PC32_2

.section       .R_386_PC32,"ax",@progbits,unique,2
.zero 4
R_386_PC32_2:
  nop

// CHECK: Disassembly of section .R_386_32:
// CHECK-NEXT: R_386_32:
// CHECK-NEXT:  11000: {{.*}} movl $69633, %edx

// CHECK: Disassembly of section .R_386_PC32:
// CHECK-NEXT: R_386_PC32:
// CHECK-NEXT:   11005:  e8 04 00 00 00  calll 4

// CHECK:      R_386_PC32_2:
// CHECK-NEXT:   1100e:  90  nop
