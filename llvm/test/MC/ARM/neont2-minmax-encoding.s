@ RUN: llvm-mc -mcpu=cortex-a8 -triple thumb-unknown-unkown -show-encoding < %s | FileCheck %s

.code 16

@ CHECK: vmin.s8	d16, d16, d17           @ encoding: [0xb1,0x06,0x40,0xef]
	vmin.s8	d16, d16, d17
@ CHECK: vmin.s16	d16, d16, d17   @ encoding: [0xb1,0x06,0x50,0xef]
	vmin.s16	d16, d16, d17
@ CHECK: vmin.s32	d16, d16, d17   @ encoding: [0xb1,0x06,0x60,0xef]
	vmin.s32	d16, d16, d17
@ CHECK: vmin.u8	d16, d16, d17           @ encoding: [0xb1,0x06,0x40,0xff]
	vmin.u8	d16, d16, d17
@ CHECK: vmin.u16	d16, d16, d17   @ encoding: [0xb1,0x06,0x50,0xff]
	vmin.u16	d16, d16, d17
@ CHECK: vmin.u32	d16, d16, d17   @ encoding: [0xb1,0x06,0x60,0xff]
	vmin.u32	d16, d16, d17
@ CHECK: vmin.f32	d16, d16, d17   @ encoding: [0xa1,0x0f,0x60,0xef]
	vmin.f32	d16, d16, d17
@ CHECK: vmin.s8	q8, q8, q9              @ encoding: [0xf2,0x06,0x40,0xef]
	vmin.s8	q8, q8, q9
@ CHECK: vmin.s16	q8, q8, q9      @ encoding: [0xf2,0x06,0x50,0xef]
	vmin.s16	q8, q8, q9
@ CHECK: vmin.s32	q8, q8, q9      @ encoding: [0xf2,0x06,0x60,0xef]
	vmin.s32	q8, q8, q9
@ CHECK: vmin.u8	q8, q8, q9              @ encoding: [0xf2,0x06,0x40,0xff]
	vmin.u8	q8, q8, q9
@ CHECK: vmin.u16	q8, q8, q9      @ encoding: [0xf2,0x06,0x50,0xff]
	vmin.u16	q8, q8, q9
@ CHECK: vmin.u32	q8, q8, q9      @ encoding: [0xf2,0x06,0x60,0xff]
	vmin.u32	q8, q8, q9
@ CHECK: vmin.f32	q8, q8, q9      @ encoding: [0xe2,0x0f,0x60,0xef]
	vmin.f32	q8, q8, q9
@ CHECK: vmax.s8	d16, d16, d17           @ encoding: [0xa1,0x06,0x40,0xef]
	vmax.s8	d16, d16, d17
@ CHECK: vmax.s16	d16, d16, d17   @ encoding: [0xa1,0x06,0x50,0xef]
	vmax.s16	d16, d16, d17
@ CHECK: vmax.s32	d16, d16, d17   @ encoding: [0xa1,0x06,0x60,0xef]
	vmax.s32	d16, d16, d17
@ CHECK: vmax.u8	d16, d16, d17           @ encoding: [0xa1,0x06,0x40,0xff]
	vmax.u8	d16, d16, d17
@ CHECK: vmax.u16	d16, d16, d17   @ encoding: [0xa1,0x06,0x50,0xff]
	vmax.u16	d16, d16, d17
@ CHECK: vmax.u32	d16, d16, d17   @ encoding: [0xa1,0x06,0x60,0xff]
	vmax.u32	d16, d16, d17
@ CHECK: vmax.f32	d16, d16, d17   @ encoding: [0xa1,0x0f,0x40,0xef]
	vmax.f32	d16, d16, d17
@ CHECK: vmax.s8	q8, q8, q9              @ encoding: [0xe2,0x06,0x40,0xef]
	vmax.s8	q8, q8, q9
@ CHECK: vmax.s16	q8, q8, q9      @ encoding: [0xe2,0x06,0x50,0xef]
	vmax.s16	q8, q8, q9
@ CHECK: vmax.s32	q8, q8, q9      @ encoding: [0xe2,0x06,0x60,0xef]
	vmax.s32	q8, q8, q9
@ CHECK: vmax.u8	q8, q8, q9              @ encoding: [0xe2,0x06,0x40,0xff]
	vmax.u8	q8, q8, q9
@ CHECK: vmax.u16	q8, q8, q9      @ encoding: [0xe2,0x06,0x50,0xff]
	vmax.u16	q8, q8, q9
@ CHECK: vmax.u32	q8, q8, q9      @ encoding: [0xe2,0x06,0x60,0xff]
	vmax.u32	q8, q8, q9
@ CHECK: vmax.f32	q8, q8, q9      @ encoding: [0xe2,0x0f,0x40,0xef]
	vmax.f32	q8, q8, q9
