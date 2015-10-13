// RUN: llvm-mc -filetype=obj -triple=powerpc64-unknown-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=powerpc64-unknown-linux %p/Inputs/shared-ppc64.s -o %t2.o
// RUN: ld.lld2 -shared %t2.o -o %t2.so
// RUN: ld.lld2 %t.o %t2.so -o %t
// RUN: llvm-objdump -d %t | FileCheck %s
// REQUIRES: ppc

// CHECK: Disassembly of section .text:

.global _start
_start:
  bl bar
  nop

// CHECK: _start:
// CHECK: 20000:       48 00 00 21     bl .+32
// CHECK-NOT: 20004:       60 00 00 00     nop
// CHECK: 20004:       e8 41 00 28     ld 2, 40(1)

.global noret
noret:
  bl bar
  li 5, 7

// CHECK: noret:
// CHECK: 20008:       48 00 00 19     bl .+24
// CHECK: 2000c:       38 a0 00 07     li 5, 7

.global noretend
noretend:
  bl bar

// CHECK: noretend:
// CHECK: 20010:       48 00 00 11     bl .+16

.global noretb
noretb:
  b bar

// CHECK: noretb:
// CHECK: 20014:       48 00 00 0c     b .+12

// This should come last to check the end-of-buffer condition.
.global last
last:
  bl bar
  nop

// CHECK: last:
// CHECK: 20018:       48 00 00 09     bl .+8
// CHECK: 2001c:       e8 41 00 28     ld 2, 40(1)

// CHECK: Disassembly of section .plt:
// CHECK: .plt:
// CHECK: 20020:       f8 41 00 28     std 2, 40(1)
// CHECK: 20024:       3d 62 00 00     addis 11, 2, 0
// CHECK: 20028:       e9 8b 80 00     ld 12, -32768(11)
// CHECK: 2002c:       e9 6c 00 00     ld 11, 0(12)
// CHECK: 20030:       7d 69 03 a6     mtctr 11
// CHECK: 20034:       e8 4c 00 08     ld 2, 8(12)
// CHECK: 20038:       e9 6c 00 10     ld 11, 16(12)
// CHECK: 2003c:       4e 80 04 20     bctr
