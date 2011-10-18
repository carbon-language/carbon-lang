@ RUN: llvm-mc -mcpu=cortex-a8 -triple arm-unknown-unknown -show-encoding < %s | FileCheck %s


	vmul.i8	d16, d16, d17
	vmul.i16	d16, d16, d17
	vmul.i32	d16, d16, d17
	vmul.f32	d16, d16, d17
	vmul.i8	q8, q8, q9
	vmul.i16	q8, q8, q9
	vmul.i32	q8, q8, q9
	vmul.f32	q8, q8, q9
	vmul.p8	d16, d16, d17
	vmul.p8	q8, q8, q9
	vmul.i16	d18, d8, d0[3]

@ CHECK: vmul.i8	d16, d16, d17   @ encoding: [0xb1,0x09,0x40,0xf2]
@ CHECK: vmul.i16	d16, d16, d17   @ encoding: [0xb1,0x09,0x50,0xf2]
@ CHECK: vmul.i32	d16, d16, d17   @ encoding: [0xb1,0x09,0x60,0xf2]
@ CHECK: vmul.f32	d16, d16, d17   @ encoding: [0xb1,0x0d,0x40,0xf3]
@ CHECK: vmul.i8	q8, q8, q9      @ encoding: [0xf2,0x09,0x40,0xf2]
@ CHECK: vmul.i16	q8, q8, q9      @ encoding: [0xf2,0x09,0x50,0xf2]
@ CHECK: vmul.i32	q8, q8, q9      @ encoding: [0xf2,0x09,0x60,0xf2]
@ CHECK: vmul.f32	q8, q8, q9      @ encoding: [0xf2,0x0d,0x40,0xf3]
@ CHECK: vmul.p8	d16, d16, d17   @ encoding: [0xb1,0x09,0x40,0xf3]
@ CHECK: vmul.p8	q8, q8, q9      @ encoding: [0xf2,0x09,0x40,0xf3]
@ CHECK: vmul.i16	d18, d8, d0[3]  @ encoding: [0x68,0x28,0xd8,0xf2]


	vqdmulh.s16	d16, d16, d17
	vqdmulh.s32	d16, d16, d17
	vqdmulh.s16	q8, q8, q9
	vqdmulh.s32	q8, q8, q9
	vqdmulh.s16	d11, d2, d3[0]

@ CHECK: vqdmulh.s16	d16, d16, d17   @ encoding: [0xa1,0x0b,0x50,0xf2]
@ CHECK: vqdmulh.s32	d16, d16, d17   @ encoding: [0xa1,0x0b,0x60,0xf2]
@ CHECK: vqdmulh.s16	q8, q8, q9      @ encoding: [0xe2,0x0b,0x50,0xf2]
@ CHECK: vqdmulh.s32	q8, q8, q9      @ encoding: [0xe2,0x0b,0x60,0xf2]
@ CHECK: vqdmulh.s16	d11, d2, d3[0]  @ encoding: [0x43,0xbc,0x92,0xf2]


	vqrdmulh.s16	d16, d16, d17
	vqrdmulh.s32	d16, d16, d17
	vqrdmulh.s16	q8, q8, q9
	vqrdmulh.s32	q8, q8, q9

@ CHECK: vqrdmulh.s16	d16, d16, d17   @ encoding: [0xa1,0x0b,0x50,0xf3]
@ CHECK: vqrdmulh.s32	d16, d16, d17   @ encoding: [0xa1,0x0b,0x60,0xf3]
@ CHECK: vqrdmulh.s16	q8, q8, q9      @ encoding: [0xe2,0x0b,0x50,0xf3]
@ CHECK: vqrdmulh.s32	q8, q8, q9      @ encoding: [0xe2,0x0b,0x60,0xf3]


	vmull.s8	q8, d16, d17
	vmull.s16	q8, d16, d17
	vmull.s32	q8, d16, d17
	vmull.u8	q8, d16, d17
	vmull.u16	q8, d16, d17
	vmull.u32	q8, d16, d17
	vmull.p8	q8, d16, d17

@ CHECK: vmull.s8	q8, d16, d17    @ encoding: [0xa1,0x0c,0xc0,0xf2]
@ CHECK: vmull.s16	q8, d16, d17    @ encoding: [0xa1,0x0c,0xd0,0xf2]
@ CHECK: vmull.s32	q8, d16, d17    @ encoding: [0xa1,0x0c,0xe0,0xf2]
@ CHECK: vmull.u8	q8, d16, d17    @ encoding: [0xa1,0x0c,0xc0,0xf3]
@ CHECK: vmull.u16	q8, d16, d17    @ encoding: [0xa1,0x0c,0xd0,0xf3]
@ CHECK: vmull.u32	q8, d16, d17    @ encoding: [0xa1,0x0c,0xe0,0xf3]
@ CHECK: vmull.p8	q8, d16, d17    @ encoding: [0xa1,0x0e,0xc0,0xf2]


	vqdmull.s16	q8, d16, d17
	vqdmull.s32	q8, d16, d17

@ CHECK: vqdmull.s16	q8, d16, d17    @ encoding: [0xa1,0x0d,0xd0,0xf2]
@ CHECK: vqdmull.s32	q8, d16, d17    @ encoding: [0xa1,0x0d,0xe0,0xf2]
