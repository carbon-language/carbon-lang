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
@ FIXME: LDR (literal)
@------------------------------------------------------------------------------
@ label operands currently assert the show-encoding asm comment helper due
@ to the use of non-contiguous bit ranges for fixups in ARM. Once that's
@ cleaned up, we can write useful assembly testcases for these sorts of
@ instructions.

@------------------------------------------------------------------------------
@ LDR (register)
@------------------------------------------------------------------------------
        ldr r3, [r8, r1]
        ldr r2, [r5, -r3]
        ldr r1, [r5, r9]!
        ldr r6, [r7, -r8]!
        ldr r5, [r9], r2
        ldr r4, [r3], -r6
        ldr r3, [r8, -r2, lsl #15]
        ldr r1, [r5], r3, asr #15

@ CHECK: ldr	r3, [r8, r1]            @ encoding: [0x01,0x30,0x98,0xe7]
@ CHECK: ldr	r2, [r5, -r3]           @ encoding: [0x03,0x20,0x15,0xe7]
@ CHECK: ldr	r1, [r5, r9]!           @ encoding: [0x09,0x10,0xb5,0xe7]
@ CHECK: ldr	r6, [r7, -r8]!          @ encoding: [0x08,0x60,0x37,0xe7]
@ CHECK: ldr	r5, [r9], r2            @ encoding: [0x02,0x50,0x99,0xe6]
@ CHECK: ldr	r4, [r3], -r6           @ encoding: [0x06,0x40,0x13,0xe6]
@ CHECK: ldr	r3, [r8, -r2, lsl #15]  @ encoding: [0x82,0x37,0x18,0xe7]
@ CHECK: ldr	r1, [r5], r3, asr #15   @ encoding: [0xc3,0x17,0x95,0xe6]


@------------------------------------------------------------------------------
@ LDRB (immediate)
@------------------------------------------------------------------------------
        ldrb r3, [r8]
        ldrb r1, [sp, #63]
        ldrb r9, [r3, #4095]!
        ldrb r8, [r1], #22
        ldrb r2, [r7], #-19

@ CHECK: ldrb	r3, [r8]                @ encoding: [0x00,0x30,0xd8,0xe5]
@ CHECK: ldrb	r1, [sp, #63]           @ encoding: [0x3f,0x10,0xdd,0xe5]
@ CHECK: ldrb	r9, [r3, #4095]!        @ encoding: [0xff,0x9f,0xf3,0xe5]
@ CHECK: ldrb	r8, [r1], #22           @ encoding: [0x16,0x80,0xd1,0xe4]
@ CHECK: ldrb	r2, [r7], #-19          @ encoding: [0x13,0x20,0x57,0xe4]


@------------------------------------------------------------------------------
@ LDRB (register)
@------------------------------------------------------------------------------
        ldr r9, [r8, r5]
        ldr r1, [r5, -r1]
        ldr r3, [r5, r2]!
        ldr r6, [r9, -r3]!
        ldr r2, [r1], r4
        ldr r8, [r4], -r5
        ldr r7, [r12, -r1, lsl #15]
        ldr r5, [r2], r9, asr #15

@ CHECK: ldr	r9, [r8, r5]            @ encoding: [0x05,0x90,0x98,0xe7]
@ CHECK: ldr	r1, [r5, -r1]           @ encoding: [0x01,0x10,0x15,0xe7]
@ CHECK: ldr	r3, [r5, r2]!           @ encoding: [0x02,0x30,0xb5,0xe7]
@ CHECK: ldr	r6, [r9, -r3]!          @ encoding: [0x03,0x60,0x39,0xe7]
@ CHECK: ldr	r2, [r1], r4            @ encoding: [0x04,0x20,0x91,0xe6]
@ CHECK: ldr	r8, [r4], -r5           @ encoding: [0x05,0x80,0x14,0xe6]
@ CHECK: ldr	r7, [r12, -r1, lsl #15] @ encoding: [0x81,0x77,0x1c,0xe7]
@ CHECK: ldr	r5, [r2], r9, asr #15   @ encoding: [0xc9,0x57,0x92,0xe6]


@------------------------------------------------------------------------------
@ LDRBT
@------------------------------------------------------------------------------
@ FIXME: Optional offset operand.
        ldrbt r3, [r1], #4
        ldrbt r2, [r8], #-8
        ldrbt r8, [r7], r6
        ldrbt r1, [r2], -r6, lsl #12


@ CHECK: ldrbt	r3, [r1], #4            @ encoding: [0x04,0x30,0xf1,0xe4]
@ CHECK: ldrbt	r2, [r8], #-8           @ encoding: [0x08,0x20,0x78,0xe4]
@ CHECK: ldrbt	r8, [r7], r6            @ encoding: [0x06,0x80,0xf7,0xe6]
@ CHECK: ldrbt	r1, [r2], -r6, lsl #12  @ encoding: [0x06,0x16,0x72,0xe6]
