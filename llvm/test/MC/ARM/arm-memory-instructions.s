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


@------------------------------------------------------------------------------
@ LDRD (immediate)
@------------------------------------------------------------------------------
        ldrd r3, r4, [r5]
        ldrd r7, r8, [r2, #15]
        ldrd r1, r2, [r9, #32]!
        ldrd r6, r7, [r1], #8
        ldrd r1, r2, [r8], #0
        ldrd r1, r2, [r8], #+0
        ldrd r1, r2, [r8], #-0

@ CHECK: ldrd	r3, r4, [r5]            @ encoding: [0xd0,0x30,0xc5,0xe1]
@ CHECK: ldrd	r7, r8, [r2, #15]       @ encoding: [0xdf,0x70,0xc2,0xe1]
@ CHECK: ldrd	r1, r2, [r9, #32]!      @ encoding: [0xd0,0x12,0xe9,0xe1]
@ CHECK: ldrd	r6, r7, [r1], #8        @ encoding: [0xd8,0x60,0xc1,0xe0]
@ CHECK: ldrd	r1, r2, [r8], #0        @ encoding: [0xd0,0x10,0xc8,0xe0]
@ CHECK: ldrd	r1, r2, [r8], #0        @ encoding: [0xd0,0x10,0xc8,0xe0]
@ CHECK: ldrd	r1, r2, [r8], #-0       @ encoding: [0xd0,0x10,0x48,0xe0]


@------------------------------------------------------------------------------
@ FIXME: LDRD (label)
@------------------------------------------------------------------------------

@------------------------------------------------------------------------------
@ LDRD (register)
@------------------------------------------------------------------------------
        ldrd r3, r4, [r1, r3]
        ldrd r4, r5, [r7, r2]!
        ldrd r1, r2, [r8], r12
        ldrd r1, r2, [r8], -r12

@ CHECK: ldrd	r3, r4, [r1, r3]        @ encoding: [0xd3,0x30,0x81,0xe1]
@ CHECK: ldrd	r4, r5, [r7, r2]!       @ encoding: [0xd2,0x40,0xa7,0xe1]
@ CHECK: ldrd	r1, r2, [r8], r12       @ encoding: [0xdc,0x10,0x88,0xe0]
@ CHECK: ldrd	r1, r2, [r8], -r12      @ encoding: [0xdc,0x10,0x08,0xe0]


@------------------------------------------------------------------------------
@ LDRH (immediate)
@------------------------------------------------------------------------------
        ldrh r3, [r4]
        ldrh r2, [r7, #4]
        ldrh r1, [r8, #64]!
        ldrh r12, [sp], #4

@ CHECK: ldrh	r3, [r4]                @ encoding: [0xb0,0x30,0xd4,0xe1]
@ CHECK: ldrh	r2, [r7, #4]            @ encoding: [0xb4,0x20,0xd7,0xe1]
@ CHECK: ldrh	r1, [r8, #64]!          @ encoding: [0xb0,0x14,0xf8,0xe1]
@ CHECK: ldrh	r12, [sp], #4           @ encoding: [0xb4,0xc0,0xdd,0xe0]


@------------------------------------------------------------------------------
@ FIXME: LDRH (label)
@------------------------------------------------------------------------------


@------------------------------------------------------------------------------
@ LDRH (register)
@------------------------------------------------------------------------------
        ldrh r6, [r5, r4]
        ldrh r3, [r8, r11]!
        ldrh r1, [r2, -r1]!
        ldrh r9, [r7], r2
        ldrh r4, [r3], -r2

@ CHECK: ldrh	r6, [r5, r4]            @ encoding: [0xb4,0x60,0x95,0xe1]
@ CHECK: ldrh	r3, [r8, r11]!          @ encoding: [0xbb,0x30,0xb8,0xe1]
@ CHECK: ldrh	r1, [r2, -r1]!          @ encoding: [0xb1,0x10,0x32,0xe1]
@ CHECK: ldrh	r9, [r7], r2            @ encoding: [0xb2,0x90,0x97,0xe0]
@ CHECK: ldrh	r4, [r3], -r2           @ encoding: [0xb2,0x40,0x13,0xe0]


@------------------------------------------------------------------------------
@ LDRHT
@------------------------------------------------------------------------------
        ldrht r9, [r7], #128
        ldrht r4, [r3], #-75
        ldrht r9, [r7], r2
        ldrht r4, [r3], -r2

@ CHECK: ldrht	r9, [r7], #128          @ encoding: [0xb0,0x98,0xf7,0xe0]
@ CHECK: ldrht	r4, [r3], #-75          @ encoding: [0xbb,0x44,0x73,0xe0]
@ CHECK: ldrht	r9, [r7], r2            @ encoding: [0xb2,0x90,0xb7,0xe0]
@ CHECK: ldrht	r4, [r3], -r2           @ encoding: [0xb2,0x40,0x33,0xe0]


@------------------------------------------------------------------------------
@ LDRSB (immediate)
@------------------------------------------------------------------------------
        ldrsb r3, [r4]
        ldrsb r2, [r7, #17]
        ldrsb r1, [r8, #255]!
        ldrsb r12, [sp], #9

@ CHECK: ldrsb	r3, [r4]                @ encoding: [0xd0,0x30,0xd4,0xe1]
@ CHECK: ldrsb	r2, [r7, #17]           @ encoding: [0xd1,0x21,0xd7,0xe1]
@ CHECK: ldrsb	r1, [r8, #255]!         @ encoding: [0xdf,0x1f,0xf8,0xe1]
@ CHECK: ldrsb	r12, [sp], #9           @ encoding: [0xd9,0xc0,0xdd,0xe0]


@------------------------------------------------------------------------------
@ FIXME: LDRSB (label)
@------------------------------------------------------------------------------


@------------------------------------------------------------------------------
@ LDRSB (register)
@------------------------------------------------------------------------------
        ldrsb r6, [r5, r4]
        ldrsb r3, [r8, r11]!
        ldrsb r1, [r2, -r1]!
        ldrsb r9, [r7], r2
        ldrsb r4, [r3], -r2


@ CHECK: ldrsb	r6, [r5, r4]            @ encoding: [0xd4,0x60,0x95,0xe1]
@ CHECK: ldrsb	r3, [r8, r11]!          @ encoding: [0xdb,0x30,0xb8,0xe1]
@ CHECK: ldrsb	r1, [r2, -r1]!          @ encoding: [0xd1,0x10,0x32,0xe1]
@ CHECK: ldrsb	r9, [r7], r2            @ encoding: [0xd2,0x90,0x97,0xe0]
@ CHECK: ldrsb	r4, [r3], -r2           @ encoding: [0xd2,0x40,0x13,0xe0]


