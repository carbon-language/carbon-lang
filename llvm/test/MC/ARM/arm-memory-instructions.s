@ RUN: llvm-mc -triple=armv7-apple-darwin -show-encoding < %s | FileCheck %s
  .syntax unified
  .globl _func

@ Check that the assembler can handle the documented syntax from the ARM ARM
@ for loads and stores.

_func:
@ CHECK: _func

@------------------------------------------------------------------------------
@ LDR (immediate)
@------------------------------------------------------------------------------
        ldr r5, [r7]
        ldr r6, [r3, #63]
        ldr r2, [r4, #4095]!
        ldr r1, [r2], #30
        ldr r3, [r1], #-30

@ CHECK: ldr	r5, [r7]                @ encoding: [0x00,0x50,0x97,0xe5]
@ CHECK: ldr	r6, [r3, #63]           @ encoding: [0x3f,0x60,0x93,0xe5]
@ CHECK: ldr	r2, [r4, #4095]!        @ encoding: [0xff,0x2f,0xb4,0xe5]
@ CHECK: ldr	r1, [r2], #30           @ encoding: [0x1e,0x10,0x92,0xe4]
@ CHECK: ldr	r3, [r1], #-30          @ encoding: [0x1e,0x30,0x11,0xe4]


@------------------------------------------------------------------------------
@ LDR (register)
@------------------------------------------------------------------------------
        ldr r3, [r8, r1]
        ldr r2, [r5, -r3]
        ldr r1, [r5, r9]!
        ldr r6, [r7, -r8]!
        ldr r5, [r9], r2
        ldr r4, [r3], -r6

@ CHECK: ldr	r3, [r8, r1]            @ encoding: [0x01,0x30,0x98,0xe7]
@ CHECK: ldr	r2, [r5, -r3]           @ encoding: [0x03,0x20,0x15,0xe7]
@ CHECK: ldr	r1, [r5, r9]!           @ encoding: [0x09,0x10,0xb5,0xe7]
@ CHECK: ldr	r6, [r7, -r8]!          @ encoding: [0x08,0x60,0x37,0xe7]
@ CHECK: ldr	r5, [r9], r2            @ encoding: [0x02,0x50,0x99,0xe6]
@ CHECK: ldr	r4, [r3], -r6           @ encoding: [0x06,0x40,0x13,0xe6]
