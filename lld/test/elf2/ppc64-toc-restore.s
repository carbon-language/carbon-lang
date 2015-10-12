// RUN: llvm-mc -filetype=obj -triple=powerpc64-unknown-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=powerpc64-unknown-linux %p/Inputs/shared-ppc64.s -o %t2.o
// RUN: ld.lld2 -shared %t2.o -o %t2.so
// RUN: ld.lld2 -dynamic-linker /lib64/ld64.so.1 -rpath foo -rpath bar --export-dynamic %t.o %t2.so -o %t
// RUN: llvm-objdump -d %t | FileCheck %s
// REQUIRES: ppc

// CHECK: Disassembly of section .text:

.global _start
_start:
  bl bar
  nop

// CHECK: _start:
// CHECK: 10010000:       48 00 00 21     bl .+32
// CHECK-NOT: 10010004:       60 00 00 00     nop
// CHECK: 10010004:       e8 41 00 28     ld 2, 40(1)

.global noret
noret:
  bl bar
  li 5, 7

// CHECK: noret:
// CHECK: 10010008:       48 00 00 19     bl .+24
// CHECK: 1001000c:       38 a0 00 07     li 5, 7

.global noretend
noretend:
  bl bar

// CHECK: noretend:
// CHECK: 10010010:       48 00 00 11     bl .+16

.global noretb
noretb:
  b bar

// CHECK: noretb:
// CHECK: 10010014:       48 00 00 0c     b .+12

// CHECK: Disassembly of section .plt:
// CHECK: .plt:
// CHECK: 10010020:
