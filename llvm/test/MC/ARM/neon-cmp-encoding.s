@ RUN: llvm-mc -mcpu=cortex-a8 -triple arm-unknown-unknown -show-encoding < %s | FileCheck %s

	vceq.i8	d16, d16, d17
	vceq.i16	d16, d16, d17
	vceq.i32	d16, d16, d17
	vceq.f32	d16, d16, d17
	vceq.i8	q8, q8, q9
	vceq.i16	q8, q8, q9
	vceq.i32	q8, q8, q9
	vceq.f32	q8, q8, q9

@ CHECK: vceq.i8	d16, d16, d17   @ encoding: [0xb1,0x08,0x40,0xf3]
@ CHECK: vceq.i16	d16, d16, d17   @ encoding: [0xb1,0x08,0x50,0xf3]
@ CHECK: vceq.i32	d16, d16, d17   @ encoding: [0xb1,0x08,0x60,0xf3]
@ CHECK: vceq.f32	d16, d16, d17   @ encoding: [0xa1,0x0e,0x40,0xf2]
@ CHECK: vceq.i8	q8, q8, q9      @ encoding: [0xf2,0x08,0x40,0xf3]
@ CHECK: vceq.i16	q8, q8, q9      @ encoding: [0xf2,0x08,0x50,0xf3]
@ CHECK: vceq.i32	q8, q8, q9      @ encoding: [0xf2,0x08,0x60,0xf3]
@ CHECK: vceq.f32	q8, q8, q9      @ encoding: [0xe2,0x0e,0x40,0xf2]

	vcge.s8	d16, d16, d17
	vcge.s16	d16, d16, d17
	vcge.s32	d16, d16, d17
	vcge.u8	d16, d16, d17
	vcge.u16	d16, d16, d17
	vcge.u32	d16, d16, d17
	vcge.f32	d16, d16, d17
	vcge.s8	q8, q8, q9
	vcge.s16	q8, q8, q9
	vcge.s32	q8, q8, q9
	vcge.u8	q8, q8, q9
	vcge.u16	q8, q8, q9
	vcge.u32	q8, q8, q9
	vcge.f32	q8, q8, q9
	vacge.f32	d16, d16, d17
	vacge.f32	q8, q8, q9

@ CHECK: vcge.s8	d16, d16, d17   @ encoding: [0xb1,0x03,0x40,0xf2]
@ CHECK: vcge.s16	d16, d16, d17   @ encoding: [0xb1,0x03,0x50,0xf2]
@ CHECK: vcge.s32	d16, d16, d17   @ encoding: [0xb1,0x03,0x60,0xf2]
@ CHECK: vcge.u8	d16, d16, d17   @ encoding: [0xb1,0x03,0x40,0xf3]
@ CHECK: vcge.u16	d16, d16, d17   @ encoding: [0xb1,0x03,0x50,0xf3]
@ CHECK: vcge.u32	d16, d16, d17   @ encoding: [0xb1,0x03,0x60,0xf3]
@ CHECK: vcge.f32	d16, d16, d17   @ encoding: [0xa1,0x0e,0x40,0xf3]
@ CHECK: vcge.s8	q8, q8, q9      @ encoding: [0xf2,0x03,0x40,0xf2]
@ CHECK: vcge.s16	q8, q8, q9      @ encoding: [0xf2,0x03,0x50,0xf2]
@ CHECK: vcge.s32	q8, q8, q9      @ encoding: [0xf2,0x03,0x60,0xf2]
@ CHECK: vcge.u8	q8, q8, q9      @ encoding: [0xf2,0x03,0x40,0xf3]
@ CHECK: vcge.u16	q8, q8, q9      @ encoding: [0xf2,0x03,0x50,0xf3]
@ CHECK: vcge.u32	q8, q8, q9      @ encoding: [0xf2,0x03,0x60,0xf3]
@ CHECK: vcge.f32	q8, q8, q9      @ encoding: [0xe2,0x0e,0x40,0xf3]
@ CHECK: vacge.f32	d16, d16, d17   @ encoding: [0xb1,0x0e,0x40,0xf3]
@ CHECK: vacge.f32	q8, q8, q9      @ encoding: [0xf2,0x0e,0x40,0xf3]

	vcgt.s8	d16, d16, d17
	vcgt.s16	d16, d16, d17
	vcgt.s32	d16, d16, d17
	vcgt.u8	d16, d16, d17
	vcgt.u16	d16, d16, d17
	vcgt.u32	d16, d16, d17
	vcgt.f32	d16, d16, d17
	vcgt.s8	q8, q8, q9
	vcgt.s16	q8, q8, q9
	vcgt.s32	q8, q8, q9
	vcgt.u8	q8, q8, q9
	vcgt.u16	q8, q8, q9
	vcgt.u32	q8, q8, q9
	vcgt.f32	q8, q8, q9
	vacgt.f32	d16, d16, d17
	vacgt.f32	q8, q8, q9

@ CHECK: vcgt.s8	d16, d16, d17   @ encoding: [0xa1,0x03,0x40,0xf2]
@ CHECK: vcgt.s16	d16, d16, d17   @ encoding: [0xa1,0x03,0x50,0xf2]
@ CHECK: vcgt.s32	d16, d16, d17   @ encoding: [0xa1,0x03,0x60,0xf2]
@ CHECK: vcgt.u8	d16, d16, d17   @ encoding: [0xa1,0x03,0x40,0xf3]
@ CHECK: vcgt.u16	d16, d16, d17   @ encoding: [0xa1,0x03,0x50,0xf3]
@ CHECK: vcgt.u32	d16, d16, d17   @ encoding: [0xa1,0x03,0x60,0xf3]
@ CHECK: vcgt.f32	d16, d16, d17   @ encoding: [0xa1,0x0e,0x60,0xf3]
@ CHECK: vcgt.s8	q8, q8, q9      @ encoding: [0xe2,0x03,0x40,0xf2]
@ CHECK: vcgt.s16	q8, q8, q9      @ encoding: [0xe2,0x03,0x50,0xf2]
@ CHECK: vcgt.s32	q8, q8, q9      @ encoding: [0xe2,0x03,0x60,0xf2]
@ CHECK: vcgt.u8	q8, q8, q9      @ encoding: [0xe2,0x03,0x40,0xf3]
@ CHECK: vcgt.u16	q8, q8, q9      @ encoding: [0xe2,0x03,0x50,0xf3]
@ CHECK: vcgt.u32	q8, q8, q9      @ encoding: [0xe2,0x03,0x60,0xf3]
@ CHECK: vcgt.f32	q8, q8, q9      @ encoding: [0xe2,0x0e,0x60,0xf3]
@ CHECK: vacgt.f32	d16, d16, d17   @ encoding: [0xb1,0x0e,0x60,0xf3]
@ CHECK: vacgt.f32	q8, q8, q9      @ encoding: [0xf2,0x0e,0x60,0xf3]

	vtst.8	d16, d16, d17
	vtst.16	d16, d16, d17
	vtst.32	d16, d16, d17
	vtst.8	q8, q8, q9
	vtst.16	q8, q8, q9
	vtst.32	q8, q8, q9

@ CHECK: vtst.8	d16, d16, d17           @ encoding: [0xb1,0x08,0x40,0xf2]
@ CHECK: vtst.16	d16, d16, d17   @ encoding: [0xb1,0x08,0x50,0xf2]
@ CHECK: vtst.32	d16, d16, d17   @ encoding: [0xb1,0x08,0x60,0xf2]
@ CHECK: vtst.8	q8, q8, q9              @ encoding: [0xf2,0x08,0x40,0xf2]
@ CHECK: vtst.16	q8, q8, q9      @ encoding: [0xf2,0x08,0x50,0xf2]
@ CHECK: vtst.32	q8, q8, q9      @ encoding: [0xf2,0x08,0x60,0xf2]

	vceq.i8	d16, d16, #0
	vcge.s8	d16, d16, #0
	vcle.s8	d16, d16, #0
	vcgt.s8	d16, d16, #0
	vclt.s8	d16, d16, #0

@ CHECK: vceq.i8	d16, d16, #0    @ encoding: [0x20,0x01,0xf1,0xf3]
@ CHECK: vcge.s8	d16, d16, #0    @ encoding: [0xa0,0x00,0xf1,0xf3]
@ CHECK: vcle.s8	d16, d16, #0    @ encoding: [0xa0,0x01,0xf1,0xf3]
@ CHECK: vcgt.s8	d16, d16, #0    @ encoding: [0x20,0x00,0xf1,0xf3]
@ CHECK: vclt.s8	d16, d16, #0    @ encoding: [0x20,0x02,0xf1,0xf3]
