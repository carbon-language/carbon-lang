@ RUN: llvm-mc -mcpu=cortex-a8 -triple thumb-unknown-unknown -show-encoding < %s | FileCheck %s
@ XFAIL: *

.code 16

@ CHECK: vmla.i8	d16, d18, d17           @ encoding: [0xa1,0x09,0x42,0xef]
	vmla.i8	d16, d18, d17
@ CHECK: vmla.i16	d16, d18, d17   @ encoding: [0xa1,0x09,0x52,0xef]
	vmla.i16	d16, d18, d17
@ CHECK: vmla.i16	d16, d18, d17   @ encoding: [0xa1,0x09,0x52,0xef]
	vmla.i32	d16, d18, d17
@ CHECK: vmla.f32	d16, d18, d17   @ encoding: [0xb1,0x0d,0x42,0xef]
	vmla.f32	d16, d18, d17
@ CHECK: vmla.i8	q9, q8, q10             @ encoding: [0xe4,0x29,0x40,0xef]
	vmla.i8	q9, q8, q10
@ CHECK: vmla.i16	q9, q8, q10     @ encoding: [0xe4,0x29,0x50,0xef]
	vmla.i16	q9, q8, q10
@ CHECK: vmla.i32	q9, q8, q10     @ encoding: [0xe4,0x29,0x60,0xef]
	vmla.i32	q9, q8, q10
@ CHECK: vmla.f32	q9, q8, q10     @ encoding: [0xf4,0x2d,0x40,0xef]
	vmla.f32	q9, q8, q10
@ CHECK: vmlal.s8	q8, d19, d18    @ encoding: [0xa2,0x08,0xc3,0xef]
	vmlal.s8	q8, d19, d18
@ CHECK: vmlal.s16	q8, d19, d18    @ encoding: [0xa2,0x08,0xd3,0xef]
	vmlal.s16	q8, d19, d18
@ CHECK: vmlal.s32	q8, d19, d18    @ encoding: [0xa2,0x08,0xe3,0xef]
	vmlal.s32	q8, d19, d18
@ CHECK: vmlal.u8	q8, d19, d18    @ encoding: [0xa2,0x08,0xc3,0xff]
	vmlal.u8	q8, d19, d18
@ CHECK: vmlal.u16	q8, d19, d18    @ encoding: [0xa2,0x08,0xd3,0xff]
	vmlal.u16	q8, d19, d18
@ CHECK: vmlal.u32	q8, d19, d18    @ encoding: [0xa2,0x08,0xe3,0xff]
	vmlal.u32	q8, d19, d18
@ CHECK: vqdmlal.s16	q8, d19, d18    @ encoding: [0xa2,0x09,0xd3,0xef]
	vqdmlal.s16	q8, d19, d18
@ CHECK: vqdmlal.s32	q8, d19, d18    @ encoding: [0xa2,0x09,0xe3,0xef]
	vqdmlal.s32	q8, d19, d18
@ CHECK: vmls.i8	d16, d18, d17           @ encoding: [0xa1,0x09,0x42,0xff]
	vmls.i8	d16, d18, d17
@ CHECK: vmls.i16	d16, d18, d17   @ encoding: [0xa1,0x09,0x52,0xff]
	vmls.i16	d16, d18, d17
@ CHECK: vmls.i32	d16, d18, d17   @ encoding: [0xa1,0x09,0x62,0xff]
	vmls.i32	d16, d18, d17
@ CHECK: vmls.f32	d16, d18, d17   @ encoding: [0xb1,0x0d,0x62,0xef]
	vmls.f32	d16, d18, d17
@ CHECK: vmls.i8	q9, q8, q10             @ encoding: [0xe4,0x29,0x40,0xff]
	vmls.i8	q9, q8, q10
@ CHECK: vmls.i16	q9, q8, q10     @ encoding: [0xe4,0x29,0x50,0xff]
	vmls.i16	q9, q8, q10
@ CHECK: vmls.i32	q9, q8, q10     @ encoding: [0xe4,0x29,0x60,0xff]
	vmls.i32	q9, q8, q10
@ CHECK: vmls.f32	q9, q8, q10     @ encoding: [0xf4,0x2d,0x60,0xef]
	vmls.f32	q9, q8, q10
@ CHECK: vmlsl.s8	q8, d19, d18    @ encoding: [0xa2,0x0a,0xc3,0xef]
	vmlsl.s8	q8, d19, d18
@ CHECK: vmlsl.s16	q8, d19, d18    @ encoding: [0xa2,0x0a,0xd3,0xef]
	vmlsl.s16	q8, d19, d18
@ CHECK: vmlsl.s32	q8, d19, d18    @ encoding: [0xa2,0x0a,0xe3,0xef]
	vmlsl.s32	q8, d19, d18
@ CHECK: vmlsl.u8	q8, d19, d18    @ encoding: [0xa2,0x0a,0xc3,0xff]
	vmlsl.u8	q8, d19, d18
@ CHECK: vmlsl.u16	q8, d19, d18    @ encoding: [0xa2,0x0a,0xd3,0xff]
	vmlsl.u16	q8, d19, d18
@ CHECK: vmlsl.u32	q8, d19, d18    @ encoding: [0xa2,0x0a,0xe3,0xff]
	vmlsl.u32	q8, d19, d18
@ CHECK: vqdmlsl.s16	q8, d19, d18    @ encoding: [0xa2,0x0b,0xd3,0xef]
	vqdmlsl.s16	q8, d19, d18
@ CHECK: vqdmlsl.s32	q8, d19, d18    @ encoding: [0xa2,0x0b,0xe3,0xef]
	vqdmlsl.s32	q8, d19, d18
