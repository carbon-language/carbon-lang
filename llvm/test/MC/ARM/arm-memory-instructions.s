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


