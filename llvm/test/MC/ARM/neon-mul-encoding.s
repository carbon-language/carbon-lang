// RUN: llvm-mc -mcpu=cortex-a8 -triple arm-unknown-unkown -show-encoding < %s | FileCheck %s

// CHECK: vmul.i8	d16, d16, d17           @ encoding: [0xb1,0x09,0x40,0xf2]
	vmul.i8	d16, d16, d17
// CHECK: vmul.i16	d16, d16, d17   @ encoding: [0xb1,0x09,0x50,0xf2]
	vmul.i16	d16, d16, d17
// CHECK: vmul.i32	d16, d16, d17   @ encoding: [0xb1,0x09,0x60,0xf2]
	vmul.i32	d16, d16, d17
// CHECK: vmul.f32	d16, d16, d17   @ encoding: [0xb1,0x0d,0x40,0xf3]
	vmul.f32	d16, d16, d17
// CHECK: vmul.i8	q8, q8, q9              @ encoding: [0xf2,0x09,0x40,0xf2]
	vmul.i8	q8, q8, q9
// CHECK: vmul.i16	q8, q8, q9      @ encoding: [0xf2,0x09,0x50,0xf2]
	vmul.i16	q8, q8, q9
// CHECK: vmul.i32	q8, q8, q9      @ encoding: [0xf2,0x09,0x60,0xf2]
	vmul.i32	q8, q8, q9
// CHECK: vmul.f32	q8, q8, q9      @ encoding: [0xf2,0x0d,0x40,0xf3]
	vmul.f32	q8, q8, q9
// CHECK: vmul.p8	d16, d16, d17           @ encoding: [0xb1,0x09,0x40,0xf3]
	vmul.p8	d16, d16, d17
// CHECK: vmul.p8	q8, q8, q9              @ encoding: [0xf2,0x09,0x40,0xf3]
	vmul.p8	q8, q8, q9
// CHECK: vqdmulh.s16	d16, d16, d17   @ encoding: [0xa1,0x0b,0x50,0xf2]
	vqdmulh.s16	d16, d16, d17
// CHECK: vqdmulh.s32	d16, d16, d17   @ encoding: [0xa1,0x0b,0x60,0xf2]
	vqdmulh.s32	d16, d16, d17
// CHECK: vqdmulh.s16	q8, q8, q9      @ encoding: [0xe2,0x0b,0x50,0xf2]
	vqdmulh.s16	q8, q8, q9
// CHECK: vqdmulh.s32	q8, q8, q9      @ encoding: [0xe2,0x0b,0x60,0xf2]
	vqdmulh.s32	q8, q8, q9
// CHECK: vqrdmulh.s16	d16, d16, d17   @ encoding: [0xa1,0x0b,0x50,0xf3]
	vqrdmulh.s16	d16, d16, d17
// CHECK: vqrdmulh.s32	d16, d16, d17   @ encoding: [0xa1,0x0b,0x60,0xf3]
	vqrdmulh.s32	d16, d16, d17
// CHECK: vqrdmulh.s16	q8, q8, q9      @ encoding: [0xe2,0x0b,0x50,0xf3]
	vqrdmulh.s16	q8, q8, q9
// CHECK: vqrdmulh.s32	q8, q8, q9      @ encoding: [0xe2,0x0b,0x60,0xf3]
	vqrdmulh.s32	q8, q8, q9
// CHECK: vmull.s8	q8, d16, d17    @ encoding: [0xa1,0x0c,0xc0,0xf2]
	vmull.s8	q8, d16, d17
// CHECK: vmull.s16	q8, d16, d17    @ encoding: [0xa1,0x0c,0xd0,0xf2]
	vmull.s16	q8, d16, d17
// CHECK: vmull.s32	q8, d16, d17    @ encoding: [0xa1,0x0c,0xe0,0xf2]
	vmull.s32	q8, d16, d17
// CHECK: vmull.u8	q8, d16, d17    @ encoding: [0xa1,0x0c,0xc0,0xf3]
	vmull.u8	q8, d16, d17
// CHECK: vmull.u16	q8, d16, d17    @ encoding: [0xa1,0x0c,0xd0,0xf3]
	vmull.u16	q8, d16, d17
// CHECK: vmull.u32	q8, d16, d17    @ encoding: [0xa1,0x0c,0xe0,0xf3]
	vmull.u32	q8, d16, d17
// CHECK: vmull.p8	q8, d16, d17    @ encoding: [0xa1,0x0e,0xc0,0xf2]
	vmull.p8	q8, d16, d17
// CHECK: vqdmull.s16	q8, d16, d17    @ encoding: [0xa1,0x0d,0xd0,0xf2]
	vqdmull.s16	q8, d16, d17
// CHECK: vqdmull.s32	q8, d16, d17    @ encoding: [0xa1,0x0d,0xe0,0xf2]
	vqdmull.s32	q8, d16, d17
