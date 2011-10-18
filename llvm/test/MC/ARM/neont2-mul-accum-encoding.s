@ RUN: llvm-mc -mcpu=cortex-a8 -triple thumb-unknown-unknown -show-encoding < %s | FileCheck %s

.code 16

	vmla.i8	d16, d18, d17
	vmla.i16	d16, d18, d17
	vmla.i32	d16, d18, d17
	vmla.f32	d16, d18, d17
	vmla.i8	q9, q8, q10
	vmla.i16	q9, q8, q10
	vmla.i32	q9, q8, q10
	vmla.f32	q9, q8, q10

@ CHECK: vmla.i8	d16, d18, d17   @ encoding: [0x42,0xef,0xa1,0x09]
@ CHECK: vmla.i16	d16, d18, d17   @ encoding: [0x52,0xef,0xa1,0x09]
@ CHECK: vmla.i32	d16, d18, d17   @ encoding: [0x62,0xef,0xa1,0x09]
@ CHECK: vmla.f32	d16, d18, d17   @ encoding: [0x42,0xef,0xb1,0x0d]
@ CHECK: vmla.i8	q9, q8, q10     @ encoding: [0x40,0xef,0xe4,0x29]
@ CHECK: vmla.i16	q9, q8, q10     @ encoding: [0x50,0xef,0xe4,0x29]
@ CHECK: vmla.i32	q9, q8, q10     @ encoding: [0x60,0xef,0xe4,0x29]
@ CHECK: vmla.f32	q9, q8, q10     @ encoding: [0x40,0xef,0xf4,0x2d]


	vmlal.s8	q8, d19, d18
	vmlal.s16	q8, d19, d18
	vmlal.s32	q8, d19, d18
	vmlal.u8	q8, d19, d18
	vmlal.u16	q8, d19, d18
	vmlal.u32	q8, d19, d18

@ CHECK: vmlal.s8	q8, d19, d18    @ encoding: [0xc3,0xef,0xa2,0x08]
@ CHECK: vmlal.s16	q8, d19, d18    @ encoding: [0xd3,0xef,0xa2,0x08]
@ CHECK: vmlal.s32	q8, d19, d18    @ encoding: [0xe3,0xef,0xa2,0x08]
@ CHECK: vmlal.u8	q8, d19, d18    @ encoding: [0xc3,0xff,0xa2,0x08]
@ CHECK: vmlal.u16	q8, d19, d18    @ encoding: [0xd3,0xff,0xa2,0x08]
@ CHECK: vmlal.u32	q8, d19, d18    @ encoding: [0xe3,0xff,0xa2,0x08]


	vqdmlal.s16	q8, d19, d18
	vqdmlal.s32	q8, d19, d18
        vqdmlal.s16 q11, d11, d7[0]
        vqdmlal.s16 q11, d11, d7[1]
        vqdmlal.s16 q11, d11, d7[2]
        vqdmlal.s16 q11, d11, d7[3]

@ CHECK: vqdmlal.s16	q8, d19, d18    @ encoding: [0xd3,0xef,0xa2,0x09]
@ CHECK: vqdmlal.s32	q8, d19, d18    @ encoding: [0xe3,0xef,0xa2,0x09]
@ CHECK: vqdmlal.s16	q11, d11, d7[0] @ encoding: [0xdb,0xef,0x47,0x63]
@ CHECK: vqdmlal.s16	q11, d11, d7[1] @ encoding: [0xdb,0xef,0x4f,0x63]
@ CHECK: vqdmlal.s16	q11, d11, d7[2] @ encoding: [0xdb,0xef,0x67,0x63]
@ CHECK: vqdmlal.s16	q11, d11, d7[3] @ encoding: [0xdb,0xef,0x6f,0x63]


	vmls.i8	d16, d18, d17
	vmls.i16	d16, d18, d17
	vmls.i32	d16, d18, d17
	vmls.f32	d16, d18, d17
	vmls.i8	q9, q8, q10
	vmls.i16	q9, q8, q10
	vmls.i32	q9, q8, q10
	vmls.f32	q9, q8, q10

@ CHECK: vmls.i8	d16, d18, d17   @ encoding: [0x42,0xff,0xa1,0x09]
@ CHECK: vmls.i16	d16, d18, d17   @ encoding: [0x52,0xff,0xa1,0x09]
@ CHECK: vmls.i32	d16, d18, d17   @ encoding: [0x62,0xff,0xa1,0x09]
@ CHECK: vmls.f32	d16, d18, d17   @ encoding: [0x62,0xef,0xb1,0x0d]
@ CHECK: vmls.i8	q9, q8, q10     @ encoding: [0x40,0xff,0xe4,0x29]
@ CHECK: vmls.i16	q9, q8, q10     @ encoding: [0x50,0xff,0xe4,0x29]
@ CHECK: vmls.i32	q9, q8, q10     @ encoding: [0x60,0xff,0xe4,0x29]
@ CHECK: vmls.f32	q9, q8, q10     @ encoding: [0x60,0xef,0xf4,0x2d]


	vmlsl.s8	q8, d19, d18
	vmlsl.s16	q8, d19, d18
	vmlsl.s32	q8, d19, d18
	vmlsl.u8	q8, d19, d18
	vmlsl.u16	q8, d19, d18
	vmlsl.u32	q8, d19, d18

@ CHECK: vmlsl.s8	q8, d19, d18    @ encoding: [0xc3,0xef,0xa2,0x0a]
@ CHECK: vmlsl.s16	q8, d19, d18    @ encoding: [0xd3,0xef,0xa2,0x0a]
@ CHECK: vmlsl.s32	q8, d19, d18    @ encoding: [0xe3,0xef,0xa2,0x0a]
@ CHECK: vmlsl.u8	q8, d19, d18    @ encoding: [0xc3,0xff,0xa2,0x0a]
@ CHECK: vmlsl.u16	q8, d19, d18    @ encoding: [0xd3,0xff,0xa2,0x0a]
@ CHECK: vmlsl.u32	q8, d19, d18    @ encoding: [0xe3,0xff,0xa2,0x0a]


	vqdmlsl.s16	q8, d19, d18
	vqdmlsl.s32	q8, d19, d18

@ CHECK: vqdmlsl.s16	q8, d19, d18    @ encoding: [0xd3,0xef,0xa2,0x0b]
@ CHECK: vqdmlsl.s32	q8, d19, d18    @ encoding: [0xe3,0xef,0xa2,0x0b]
