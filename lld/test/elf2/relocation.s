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
  nop

// CHECK: Disassembly of section .text:
// CHECK-NEXT: _start:
// CHECK-NEXT:   11000:  e8 04 00 00 00   callq 4
// CHECK-NEXT:   11005:

// CHECK:      lulz:
// CHECK-NEXT:   11009:  90  nop


.global bar
.section       .text2,"ax",@progbits
bar:
  movl $bar, %edx // R_X86_64_32

// FIXME: this would be far more self evident if llvm-objdump printed
// constants in hex.
// CHECK: Disassembly of section .text2:
// CHECK-NEXT: bar:
// CHECK-NEXT:  12000: {{.*}} movl $73728, %edx
