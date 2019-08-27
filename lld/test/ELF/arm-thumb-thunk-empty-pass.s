// REQUIRES: arm
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=thumbv7a-none-linux-gnueabi %s -o %t
// RUN: ld.lld %t -o %t2
// RUN: llvm-objdump -d %t2 -triple=thumbv7a | FileCheck %s
 .syntax unified
 .global _start, foo
 .type _start, %function
 .section .text.start,"ax",%progbits
_start:
 bl _start
 .section .text.dummy1,"ax",%progbits
 .space 0xfffffe
 .section .text.foo,"ax",%progbits
  .type foo, %function
foo:
 bl _start

// CHECK: Disassembly of section .text:
// CHECK-EMPTY:
// CHECK-NEXT: _start:
// CHECK-NEXT:    11000:       ff f7 fe ff     bl      #-4
// CHECK: __Thumbv7ABSLongThunk__start:
// CHECK-NEXT:    11004:       ff f7 fc bf     b.w     #-8 <_start>

// CHECK: __Thumbv7ABSLongThunk__start:
// CHECK:       1011008:       41 f2 01 0c     movw    r12, #4097
// CHECK-NEXT:  101100c:       c0 f2 01 0c     movt    r12, #1
// CHECK-NEXT:  1011010:       60 47   bx      r12
// CHECK: foo:
// CHECK-NEXT:  1011012:       ff f7 f9 ff     bl      #-14
