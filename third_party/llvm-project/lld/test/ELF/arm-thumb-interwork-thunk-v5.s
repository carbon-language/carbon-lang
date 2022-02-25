// REQUIRES: arm
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=armv5-none-linux-gnueabi %s -o %t
// RUN: ld.lld %t -o %t2
// RUN: llvm-objdump -d %t2 --triple=armv5-none-linux-gnueabi | FileCheck %s
// RUN: ld.lld %t -o %t3 --shared
// RUN: llvm-objdump -d %t3 --triple=armv5-none-linux-gnueabi | FileCheck --check-prefix=CHECK-PI %s

// Test ARM Thumb Interworking on older Arm architectures using Thunks that do
// not use MOVT/MOVW instructions.
// For pure interworking (not considering range extension) there is only the
// case of an Arm B to a Thumb Symbol to consider as in older Arm architectures
// there is no Thumb B.w that we can intercept with a Thunk and we still assume
// support for the blx instruction for Thumb BL and BLX to an Arm symbol.
        .arm
        .text
        .syntax unified
        .cpu    arm10tdmi

        .text
        .globl _start
        .type _start, %function
        .balign 0x1000
_start:
        b thumb_func
        bl thumb_func
        blx thumb_func
        bx lr

// CHECK: <_start>:
// CHECK-NEXT: 21000: 03 00 00 ea     b       0x21014 <__ARMv5ABSLongThunk_thumb_func>
// CHECK-NEXT: 21004: 01 00 00 fa     blx     0x21010 <thumb_func>
// CHECK-NEXT: 21008: 00 00 00 fa     blx     0x21010 <thumb_func>
// CHECK-NEXT: 2100c: 1e ff 2f e1     bx      lr

// CHECK: <thumb_func>:
// CHECK-NEXT: 21010: 70 47   bx      lr

// CHECK: <__ARMv5ABSLongThunk_thumb_func>:
// CHECK-NEXT: 21014: 04 f0 1f e5     ldr     pc, [pc, #-4]
// CHECK: <$d>:
// CHECK-NEXT: 21018: 11 10 02 00     .word   0x00021011

// CHECK-PI: <_start>:
// CHECK-PI-NEXT: 11000: 03 00 00 ea     b       0x11014 <__ARMV5PILongThunk_thumb_func>
// CHECK-PI-NEXT: 11004: 01 00 00 fa     blx     0x11010 <thumb_func>
// CHECK-PI-NEXT: 11008: 00 00 00 fa     blx     0x11010 <thumb_func>
// CHECK-PI-NEXT: 1100c: 1e ff 2f e1     bx      lr

// CHECK-PI: <thumb_func>:
// CHECK-PI-NEXT: 11010: 70 47   bx      lr

// CHECK-PI: <__ARMV5PILongThunk_thumb_func>:
// CHECK-PI-NEXT: 11014: 04 c0 9f e5     ldr     r12, [pc, #4]
// CHECK-PI-NEXT: 11018: 0c c0 8f e0     add     r12, pc, r12
// CHECK-PI-NEXT: 1101c: 1c ff 2f e1     bx      r12
// CHECK-PI: <$d>:
// CHECK-PI-NEXT: 11020: f1 ff ff ff     .word   0xfffffff1

        .section .text.1, "ax", %progbits
        .thumb
        .hidden thumb_func
        .type thumb_func, %function
thumb_func:
        bx lr
