@ RUN: llvm-mc -mcpu=cortex-a8 -triple thumb-unknown-unkown -show-encoding < %s | FileCheck %s

.code 16

@ CHECK: vmin.s8	d16, d16, d17           @ encoding: [0x40,0xef,0xb1,0x06]
	vmin.s8	d16, d16, d17
@ CHECK: vmin.s16	d16, d16, d17   @ encoding: [0x50,0xef,0xb1,0x06]
	vmin.s16	d16, d16, d17
@ CHECK: vmin.s32	d16, d16, d17   @ encoding: [0x60,0xef,0xb1,0x06]
	vmin.s32	d16, d16, d17
@ CHECK: vmin.u8	d16, d16, d17           @ encoding: [0x40,0xff,0xb1,0x06]
	vmin.u8	d16, d16, d17
@ CHECK: vmin.u16	d16, d16, d17   @ encoding: [0x50,0xff,0xb1,0x06]
	vmin.u16	d16, d16, d17
@ CHECK: vmin.u32	d16, d16, d17   @ encoding: [0x60,0xff,0xb1,0x06]
	vmin.u32	d16, d16, d17
@ CHECK: vmin.f32	d16, d16, d17   @ encoding: [0x60,0xef,0xa1,0x0f]
	vmin.f32	d16, d16, d17
@ CHECK: vmin.s8	q8, q8, q9              @ encoding: [0x40,0xef,0xf2,0x06]
	vmin.s8	q8, q8, q9
@ CHECK: vmin.s16	q8, q8, q9      @ encoding: [0x50,0xef,0xf2,0x06]
	vmin.s16	q8, q8, q9
@ CHECK: vmin.s32	q8, q8, q9      @ encoding: [0x60,0xef,0xf2,0x06]
	vmin.s32	q8, q8, q9
@ CHECK: vmin.u8	q8, q8, q9              @ encoding: [0x40,0xff,0xf2,0x06]
	vmin.u8	q8, q8, q9
@ CHECK: vmin.u16	q8, q8, q9      @ encoding: [0x50,0xff,0xf2,0x06]
	vmin.u16	q8, q8, q9
@ CHECK: vmin.u32	q8, q8, q9      @ encoding: [0x60,0xff,0xf2,0x06]
	vmin.u32	q8, q8, q9
@ CHECK: vmin.f32	q8, q8, q9      @ encoding: [0x60,0xef,0xe2,0x0f]
	vmin.f32	q8, q8, q9
@ CHECK: vmax.s8	d16, d16, d17           @ encoding: [0x40,0xef,0xa1,0x06]
	vmax.s8	d16, d16, d17
@ CHECK: vmax.s16	d16, d16, d17   @ encoding: [0x50,0xef,0xa1,0x06]
	vmax.s16	d16, d16, d17
@ CHECK: vmax.s32	d16, d16, d17   @ encoding: [0x60,0xef,0xa1,0x06]
	vmax.s32	d16, d16, d17
@ CHECK: vmax.u8	d16, d16, d17           @ encoding: [0x40,0xff,0xa1,0x06]
	vmax.u8	d16, d16, d17
@ CHECK: vmax.u16	d16, d16, d17   @ encoding: [0x50,0xff,0xa1,0x06]
	vmax.u16	d16, d16, d17
@ CHECK: vmax.u32	d16, d16, d17   @ encoding: [0x60,0xff,0xa1,0x06]
	vmax.u32	d16, d16, d17
@ CHECK: vmax.f32	d16, d16, d17   @ encoding: [0x40,0xef,0xa1,0x0f]
	vmax.f32	d16, d16, d17
@ CHECK: vmax.s8	q8, q8, q9              @ encoding: [0x40,0xef,0xe2,0x06]
	vmax.s8	q8, q8, q9
@ CHECK: vmax.s16	q8, q8, q9      @ encoding: [0x50,0xef,0xe2,0x06]
	vmax.s16	q8, q8, q9
@ CHECK: vmax.s32	q8, q8, q9      @ encoding: [0x60,0xef,0xe2,0x06]
	vmax.s32	q8, q8, q9
@ CHECK: vmax.u8	q8, q8, q9              @ encoding: [0x40,0xff,0xe2,0x06]
	vmax.u8	q8, q8, q9
@ CHECK: vmax.u16	q8, q8, q9      @ encoding: [0x50,0xff,0xe2,0x06]
	vmax.u16	q8, q8, q9
@ CHECK: vmax.u32	q8, q8, q9      @ encoding: [0x60,0xff,0xe2,0x06]
	vmax.u32	q8, q8, q9
@ CHECK: vmax.f32	q8, q8, q9      @ encoding: [0x40,0xef,0xe2,0x0f]
	vmax.f32	q8, q8, q9
