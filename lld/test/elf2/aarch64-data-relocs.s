// RUN: llvm-mc -filetype=obj -triple=aarch64-none-freebsd %s -o %t
// RUN: lld -flavor gnu2 -shared %t -o %t2
// RUN: llvm-objdump -d %t2 | FileCheck %s
// REQUIRES: aarch64

.section .R_AARCH64_ABS16, "ax",@progbits
  .hword sym+12

.section .R_AARCH64_ABS32, "ax",@progbits
  .word sym+24

.section .R_AARCH64_ABS64, "ax",@progbits
  .xword sym+36

// CHECK: Disassembly of section .R_AARCH64_ABS16:
// CHECK-NEXT: $d.0:
// CHECK-NEXT:     2000:       0c 00   .short
//                             ^-- A = 0xc

// CHECK-NEXT: Disassembly of section .R_AARCH64_ABS32:
// CHECK-NEXT: $d.1:
// CHECK-NEXT:     2002:       18 00 00 00     .word
//                             ^-- A = 0x18

// CHECK-NEXT: Disassembly of section .R_AARCH64_ABS64:
// CHECK-NEXT: $d.2:
// CHECK-NEXT:     2006:       24 00 00 00     .word
//                             ^-- A = 0x24
// CHECK-NEXT:     200a:       00 00 00 00     .word
