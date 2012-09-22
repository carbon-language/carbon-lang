@ RUN: llvm-mc -mcpu=cortex-a8 -triple armv7 -show-encoding < %s | FileCheck %s

	ldr r0, [r0, r0]
	ldr r0, [r0, r0, lsr #32]
	ldr r0, [r0, r0, lsr #16]
	ldr r0, [r0, r0, lsl #0]
	ldr r0, [r0, r0, lsl #16]
	ldr r0, [r0, r0, asr #32]
	ldr r0, [r0, r0, asr #16]
	ldr r0, [r0, r0, rrx]
	ldr r0, [r0, r0, ror #16]

@ CHECK: ldr r0, [r0, r0]          @ encoding: [0x00,0x00,0x90,0xe7]
@ CHECK: ldr r0, [r0, r0, lsr #32] @ encoding: [0x20,0x00,0x90,0xe7]
@ CHECK: ldr r0, [r0, r0, lsr #16] @ encoding: [0x20,0x08,0x90,0xe7]
@ CHECK: ldr r0, [r0, r0]          @ encoding: [0x00,0x00,0x90,0xe7]
@ CHECK: ldr r0, [r0, r0, lsl #16] @ encoding: [0x00,0x08,0x90,0xe7]
@ CHECK: ldr r0, [r0, r0, asr #32] @ encoding: [0x40,0x00,0x90,0xe7]
@ CHECK: ldr r0, [r0, r0, asr #16] @ encoding: [0x40,0x08,0x90,0xe7]
@ CHECK: ldr r0, [r0, r0, rrx]     @ encoding: [0x60,0x00,0x90,0xe7]
@ CHECK: ldr r0, [r0, r0, ror #16] @ encoding: [0x60,0x08,0x90,0xe7]

	pld [r0, r0]
	pld [r0, r0, lsr #32]
	pld [r0, r0, lsr #16]
	pld [r0, r0, lsl #0]
	pld [r0, r0, lsl #16]
	pld [r0, r0, asr #32]
	pld [r0, r0, asr #16]
	pld [r0, r0, rrx]
	pld [r0, r0, ror #16]

@ CHECK: [r0, r0]          @ encoding: [0x00,0xf0,0xd0,0xf7]
@ CHECK: [r0, r0, lsr #32] @ encoding: [0x20,0xf0,0xd0,0xf7]
@ CHECK: [r0, r0, lsr #16] @ encoding: [0x20,0xf8,0xd0,0xf7]
@ CHECK: [r0, r0]          @ encoding: [0x00,0xf0,0xd0,0xf7]
@ CHECK: [r0, r0, lsl #16] @ encoding: [0x00,0xf8,0xd0,0xf7]
@ CHECK: [r0, r0, asr #32] @ encoding: [0x40,0xf0,0xd0,0xf7]
@ CHECK: [r0, r0, asr #16] @ encoding: [0x40,0xf8,0xd0,0xf7]
@ CHECK: [r0, r0, rrx]     @ encoding: [0x60,0xf0,0xd0,0xf7]
@ CHECK: [r0, r0, ror #16] @ encoding: [0x60,0xf8,0xd0,0xf7]

	str r0, [r0, r0]
	str r0, [r0, r0, lsr #32]
	str r0, [r0, r0, lsr #16]
	str r0, [r0, r0, lsl #0]
	str r0, [r0, r0, lsl #16]
	str r0, [r0, r0, asr #32]
	str r0, [r0, r0, asr #16]
	str r0, [r0, r0, rrx]
	str r0, [r0, r0, ror #16]

@ CHECK: str r0, [r0, r0]          @ encoding: [0x00,0x00,0x80,0xe7]
@ CHECK: str r0, [r0, r0, lsr #32] @ encoding: [0x20,0x00,0x80,0xe7]
@ CHECK: str r0, [r0, r0, lsr #16] @ encoding: [0x20,0x08,0x80,0xe7]
@ CHECK: str r0, [r0, r0]          @ encoding: [0x00,0x00,0x80,0xe7]
@ CHECK: str r0, [r0, r0, lsl #16] @ encoding: [0x00,0x08,0x80,0xe7]
@ CHECK: str r0, [r0, r0, asr #32] @ encoding: [0x40,0x00,0x80,0xe7]
@ CHECK: str r0, [r0, r0, asr #16] @ encoding: [0x40,0x08,0x80,0xe7]
@ CHECK: str r0, [r0, r0, rrx]     @ encoding: [0x60,0x00,0x80,0xe7]
@ CHECK: str r0, [r0, r0, ror #16] @ encoding: [0x60,0x08,0x80,0xe7]

@ Uses printAddrMode2OffsetOperand(), used by LDRBT_POST_IMM LDRBT_POST_REG
@ LDRB_POST_IMM LDRB_POST_REG LDRT_POST_IMM LDRT_POST_REG LDR_POST_IMM
@ LDR_POST_REG STRBT_POST_IMM STRBT_POST_REG STRB_POST_IMM STRB_POST_REG
@ STRT_POST_IMM STRT_POST_REG STR_POST_IMM STR_POST_REG

	ldr r0, [r1], r2, rrx
	ldr r3, [r4], r5, ror #0
	str r6, [r7], r8, lsl #0
	str r9, [r10], r11

@ CHECK: ldr r0, [r1], r2, rrx    @ encoding: [0x62,0x00,0x91,0xe6]
@ CHECK: ldr r3, [r4], r5         @ encoding: [0x05,0x30,0x94,0xe6]
@ CHECK: str r6, [r7], r8         @ encoding: [0x08,0x60,0x87,0xe6]
@ CHECK: str r9, [r10], r11       @ encoding: [0x0b,0x90,0x8a,0xe6]
