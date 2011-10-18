@ RUN: llvm-mc -mcpu=cortex-a8 -triple thumb-unknown-unknown -show-encoding < %s | FileCheck %s

.code 16

@ CHECK: vmul.i8	d16, d16, d17           @ encoding: [0x40,0xef,0xb1,0x09]
	vmul.i8	d16, d16, d17
@ CHECK: vmul.i16	d16, d16, d17   @ encoding: [0x50,0xef,0xb1,0x09]
	vmul.i16	d16, d16, d17
@ CHECK: vmul.i32	d16, d16, d17   @ encoding: [0x60,0xef,0xb1,0x09]
	vmul.i32	d16, d16, d17
@ CHECK: vmul.f32	d16, d16, d17   @ encoding: [0x40,0xff,0xb1,0x0d]
	vmul.f32	d16, d16, d17
@ CHECK: vmul.i8	q8, q8, q9              @ encoding: [0x40,0xef,0xf2,0x09]
	vmul.i8	q8, q8, q9
@ CHECK: vmul.i16	q8, q8, q9      @ encoding: [0x50,0xef,0xf2,0x09]
	vmul.i16	q8, q8, q9
@ CHECK: vmul.i32	q8, q8, q9      @ encoding: [0x60,0xef,0xf2,0x09]
	vmul.i32	q8, q8, q9
@ CHECK: vmul.f32	q8, q8, q9      @ encoding: [0x40,0xff,0xf2,0x0d]
	vmul.f32	q8, q8, q9
@ CHECK: vmul.p8	d16, d16, d17           @ encoding: [0x40,0xff,0xb1,0x09]
	vmul.p8	d16, d16, d17
@ CHECK: vmul.p8	q8, q8, q9              @ encoding: [0x40,0xff,0xf2,0x09]
	vmul.p8	q8, q8, q9

	vmul.i16	d18, d8, d0[3]
@ CHECK: vmul.i16	d18, d8, d0[3]    @ encoding: [0xd8,0xef,0x68,0x28]

@ CHECK: vqdmulh.s16	d16, d16, d17   @ encoding: [0x50,0xef,0xa1,0x0b]
	vqdmulh.s16	d16, d16, d17
@ CHECK: vqdmulh.s32	d16, d16, d17   @ encoding: [0x60,0xef,0xa1,0x0b]
	vqdmulh.s32	d16, d16, d17
@ CHECK: vqdmulh.s16	q8, q8, q9      @ encoding: [0x50,0xef,0xe2,0x0b]
	vqdmulh.s16	q8, q8, q9
@ CHECK: vqdmulh.s32	q8, q8, q9      @ encoding: [0x60,0xef,0xe2,0x0b]
	vqdmulh.s32	q8, q8, q9
@ CHECK: vqrdmulh.s16	d16, d16, d17   @ encoding: [0x50,0xff,0xa1,0x0b]
	vqrdmulh.s16	d16, d16, d17
@ CHECK: vqrdmulh.s32	d16, d16, d17   @ encoding: [0x60,0xff,0xa1,0x0b]
	vqrdmulh.s32	d16, d16, d17
@ CHECK: vqrdmulh.s16	q8, q8, q9      @ encoding: [0x50,0xff,0xe2,0x0b]
	vqrdmulh.s16	q8, q8, q9
@ CHECK: vqrdmulh.s32	q8, q8, q9      @ encoding: [0x60,0xff,0xe2,0x0b]
	vqrdmulh.s32	q8, q8, q9
@ CHECK: vmull.s8	q8, d16, d17    @ encoding: [0xc0,0xef,0xa1,0x0c]
	vmull.s8	q8, d16, d17
@ CHECK: vmull.s16	q8, d16, d17    @ encoding: [0xd0,0xef,0xa1,0x0c]
	vmull.s16	q8, d16, d17
@ CHECK: vmull.s32	q8, d16, d17    @ encoding: [0xe0,0xef,0xa1,0x0c]
	vmull.s32	q8, d16, d17
@ CHECK: vmull.u8	q8, d16, d17    @ encoding: [0xc0,0xff,0xa1,0x0c]
	vmull.u8	q8, d16, d17
@ CHECK: vmull.u16	q8, d16, d17    @ encoding: [0xd0,0xff,0xa1,0x0c]
	vmull.u16	q8, d16, d17
@ CHECK: vmull.u32	q8, d16, d17    @ encoding: [0xe0,0xff,0xa1,0x0c]
	vmull.u32	q8, d16, d17
@ CHECK: vmull.p8	q8, d16, d17    @ encoding: [0xc0,0xef,0xa1,0x0e]
	vmull.p8	q8, d16, d17
@ CHECK: vqdmull.s16	q8, d16, d17    @ encoding: [0xd0,0xef,0xa1,0x0d]
	vqdmull.s16	q8, d16, d17
@ CHECK: vqdmull.s32	q8, d16, d17    @ encoding: [0xe0,0xef,0xa1,0x0d]
	vqdmull.s32	q8, d16, d17

@	vmla.i32	q12, q8, d3[0]
@	vqdmulh.s16	d11, d2, d3[0]
@ FIXME: vmla.i32	q12, q8, d3[0]    @ encoding: [0xe0,0xff,0xc3,0x80]
@ FIXME: vqdmulh.s16	d11, d2, d3[0]    @ encoding: [0x92,0xef,0x43,0xbc]
