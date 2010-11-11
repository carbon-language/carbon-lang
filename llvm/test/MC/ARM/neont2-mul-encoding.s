@ RUN: llvm-mc -mcpu=cortex-a8 -triple thumb-unknown-unkown -show-encoding < %s | FileCheck %s

.code 16

@ CHECK: vmul.i8	d16, d16, d17           @ encoding: [0xb1,0x09,0x40,0xef]
	vmul.i8	d16, d16, d17
@ CHECK: vmul.i16	d16, d16, d17   @ encoding: [0xb1,0x09,0x50,0xef]
	vmul.i16	d16, d16, d17
@ CHECK: vmul.i32	d16, d16, d17   @ encoding: [0xb1,0x09,0x60,0xef]
	vmul.i32	d16, d16, d17
@ CHECK: vmul.f32	d16, d16, d17   @ encoding: [0xb1,0x0d,0x40,0xff]
	vmul.f32	d16, d16, d17
@ CHECK: vmul.i8	q8, q8, q9              @ encoding: [0xf2,0x09,0x40,0xef]
	vmul.i8	q8, q8, q9
@ CHECK: vmul.i16	q8, q8, q9      @ encoding: [0xf2,0x09,0x50,0xef]
	vmul.i16	q8, q8, q9
@ CHECK: vmul.i32	q8, q8, q9      @ encoding: [0xf2,0x09,0x60,0xef]
	vmul.i32	q8, q8, q9
@ CHECK: vmul.f32	q8, q8, q9      @ encoding: [0xf2,0x0d,0x40,0xff]
	vmul.f32	q8, q8, q9
@ CHECK: vmul.p8	d16, d16, d17           @ encoding: [0xb1,0x09,0x40,0xff]
	vmul.p8	d16, d16, d17
@ CHECK: vmul.p8	q8, q8, q9              @ encoding: [0xf2,0x09,0x40,0xff]
	vmul.p8	q8, q8, q9
@ CHECK: vqdmulh.s16	d16, d16, d17   @ encoding: [0xa1,0x0b,0x50,0xef]
	vqdmulh.s16	d16, d16, d17
@ CHECK: vqdmulh.s32	d16, d16, d17   @ encoding: [0xa1,0x0b,0x60,0xef]
	vqdmulh.s32	d16, d16, d17
@ CHECK: vqdmulh.s16	q8, q8, q9      @ encoding: [0xe2,0x0b,0x50,0xef]
	vqdmulh.s16	q8, q8, q9
@ CHECK: vqdmulh.s32	q8, q8, q9      @ encoding: [0xe2,0x0b,0x60,0xef]
	vqdmulh.s32	q8, q8, q9
@ CHECK: vqrdmulh.s16	d16, d16, d17   @ encoding: [0xa1,0x0b,0x50,0xff]
	vqrdmulh.s16	d16, d16, d17
@ CHECK: vqrdmulh.s32	d16, d16, d17   @ encoding: [0xa1,0x0b,0x60,0xff]
	vqrdmulh.s32	d16, d16, d17
@ CHECK: vqrdmulh.s16	q8, q8, q9      @ encoding: [0xe2,0x0b,0x50,0xff]
	vqrdmulh.s16	q8, q8, q9
@ CHECK: vqrdmulh.s32	q8, q8, q9      @ encoding: [0xe2,0x0b,0x60,0xff]
	vqrdmulh.s32	q8, q8, q9
@ CHECK: vmull.s8	q8, d16, d17    @ encoding: [0xa1,0x0c,0xc0,0xef]
	vmull.s8	q8, d16, d17
@ CHECK: vmull.s16	q8, d16, d17    @ encoding: [0xa1,0x0c,0xd0,0xef]
	vmull.s16	q8, d16, d17
@ CHECK: vmull.s32	q8, d16, d17    @ encoding: [0xa1,0x0c,0xe0,0xef]
	vmull.s32	q8, d16, d17
@ CHECK: vmull.u8	q8, d16, d17    @ encoding: [0xa1,0x0c,0xc0,0xff]
	vmull.u8	q8, d16, d17
@ CHECK: vmull.u16	q8, d16, d17    @ encoding: [0xa1,0x0c,0xd0,0xff]
	vmull.u16	q8, d16, d17
@ CHECK: vmull.u32	q8, d16, d17    @ encoding: [0xa1,0x0c,0xe0,0xff]
	vmull.u32	q8, d16, d17
@ CHECK: vmull.p8	q8, d16, d17    @ encoding: [0xa1,0x0e,0xc0,0xef]
	vmull.p8	q8, d16, d17
@ CHECK: vqdmull.s16	q8, d16, d17    @ encoding: [0xa1,0x0d,0xd0,0xef]
	vqdmull.s16	q8, d16, d17
@ CHECK: vqdmull.s32	q8, d16, d17    @ encoding: [0xa1,0x0d,0xe0,0xef]
	vqdmull.s32	q8, d16, d17
