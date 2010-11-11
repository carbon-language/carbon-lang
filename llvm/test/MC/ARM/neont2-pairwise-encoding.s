@ RUN: llvm-mc -mcpu=cortex-a8 -triple thumb-unknown-unkown -show-encoding < %s | FileCheck %s
@ XFAIL: *

.code 16

@ CHECK: vpadd.i8	d16, d17, d16   @ encoding: [0xb0,0x0b,0x41,0xef]
	vpadd.i8	d16, d17, d16
@ CHECK: vpadd.i16	d16, d17, d16   @ encoding: [0xb0,0x0b,0x51,0xef]
	vpadd.i16	d16, d17, d16
@ CHECK: vpadd.i32	d16, d17, d16   @ encoding: [0xb0,0x0b,0x61,0xef]
	vpadd.i32	d16, d17, d16
@ CHECK: vpadd.f32	d16, d16, d17   @ encoding: [0xa1,0x0d,0x40,0xff]
	vpadd.f32	d16, d16, d17
@ CHECK: vpaddl.s8	d16, d16        @ encoding: [0x20,0x02,0xf0,0xff]
	vpaddl.s8	d16, d16
@ CHECK: vpaddl.s16	d16, d16        @ encoding: [0x20,0x02,0xf4,0xff]
	vpaddl.s16	d16, d16
@ CHECK: vpaddl.s32	d16, d16        @ encoding: [0x20,0x02,0xf8,0xff]
	vpaddl.s32	d16, d16
@ CHECK: vpaddl.u8	d16, d16        @ encoding: [0xa0,0x02,0xf0,0xff]
	vpaddl.u8	d16, d16
@ CHECK: vpaddl.u16	d16, d16        @ encoding: [0xa0,0x02,0xf4,0xff]
	vpaddl.u16	d16, d16
@ CHECK: vpaddl.u32	d16, d16        @ encoding: [0xa0,0x02,0xf8,0xff]
	vpaddl.u32	d16, d16
@ CHECK: vpaddl.s8	q8, q8          @ encoding: [0x60,0x02,0xf0,0xff]
	vpaddl.s8	q8, q8
@ CHECK: vpaddl.s16	q8, q8          @ encoding: [0x60,0x02,0xf4,0xff]
	vpaddl.s16	q8, q8
@ CHECK: vpaddl.s32	q8, q8          @ encoding: [0x60,0x02,0xf8,0xff]
	vpaddl.s32	q8, q8
@ CHECK: vpaddl.u8	q8, q8          @ encoding: [0xe0,0x02,0xf0,0xff]
	vpaddl.u8	q8, q8
@ CHECK: vpaddl.u16	q8, q8          @ encoding: [0xe0,0x02,0xf4,0xff]
	vpaddl.u16	q8, q8
@ CHECK: vpaddl.u32	q8, q8          @ encoding: [0xe0,0x02,0xf8,0xff]
	vpaddl.u32	q8, q8
@ CHECK: vpadal.s8	d16, d17        @ encoding: [0x21,0x06,0xf0,0xff]
	vpadal.s8	d16, d17
@ CHECK: vpadal.s16	d16, d17        @ encoding: [0x21,0x06,0xf4,0xff]
	vpadal.s16	d16, d17
@ CHECK: vpadal.s32	d16, d17        @ encoding: [0x21,0x06,0xf8,0xff]
	vpadal.s32	d16, d17
@ CHECK: vpadal.u8	d16, d17        @ encoding: [0xa1,0x06,0xf0,0xff]
	vpadal.u8	d16, d17
@ CHECK: vpadal.u16	d16, d17        @ encoding: [0xa1,0x06,0xf4,0xff]
	vpadal.u16	d16, d17
@ CHECK: vpadal.u32	d16, d17        @ encoding: [0xa1,0x06,0xf8,0xff]
	vpadal.u32	d16, d17
@ CHECK: vpadal.s8	q9, q8          @ encoding: [0x60,0x26,0xf0,0xff]
	vpadal.s8	q9, q8
@ CHECK: vpadal.s16	q9, q8          @ encoding: [0x60,0x26,0xf4,0xff]
	vpadal.s16	q9, q8
@ CHECK: vpadal.s32	q9, q8          @ encoding: [0x60,0x26,0xf8,0xff]
	vpadal.s32	q9, q8
@ CHECK: vpadal.u8	q9, q8          @ encoding: [0xe0,0x26,0xf0,0xff]
	vpadal.u8	q9, q8
@ CHECK: vpadal.u16	q9, q8          @ encoding: [0xe0,0x26,0xf4,0xff]
	vpadal.u16	q9, q8
@ CHECK: vpadal.u32	q9, q8          @ encoding: [0xe0,0x26,0xf8,0xff]
	vpadal.u32	q9, q8
@ CHECK: vpmin.s8	d16, d16, d17   @ encoding: [0xb1,0x0a,0x40,0xef]
	vpmin.s8	d16, d16, d17
@ CHECK: vpmin.s16	d16, d16, d17   @ encoding: [0xb1,0x0a,0x50,0xef]
	vpmin.s16	d16, d16, d17
@ CHECK: vpmin.s32	d16, d16, d17   @ encoding: [0xb1,0x0a,0x60,0xef]
	vpmin.s32	d16, d16, d17
@ CHECK: vpmin.u8	d16, d16, d17   @ encoding: [0xb1,0x0a,0x40,0xff]
	vpmin.u8	d16, d16, d17
@ CHECK: vpmin.u16	d16, d16, d17   @ encoding: [0xb1,0x0a,0x50,0xff]
	vpmin.u16	d16, d16, d17
@ CHECK: vpmin.u32	d16, d16, d17   @ encoding: [0xb1,0x0a,0x60,0xff]
	vpmin.u32	d16, d16, d17
@ CHECK: vpmin.f32	d16, d16, d17   @ encoding: [0xa1,0x0f,0x60,0xff]
	vpmin.f32	d16, d16, d17
@ CHECK: vpmax.s8	d16, d16, d17   @ encoding: [0xa1,0x0a,0x40,0xef]
	vpmax.s8	d16, d16, d17
@ CHECK: vpmax.s16	d16, d16, d17   @ encoding: [0xa1,0x0a,0x50,0xef]
	vpmax.s16	d16, d16, d17
@ CHECK: vpmax.s32	d16, d16, d17   @ encoding: [0xa1,0x0a,0x60,0xef]
	vpmax.s32	d16, d16, d17
@ CHECK: vpmax.u8	d16, d16, d17   @ encoding: [0xa1,0x0a,0x40,0xff]
	vpmax.u8	d16, d16, d17
@ CHECK: vpmax.u16	d16, d16, d17   @ encoding: [0xa1,0x0a,0x50,0xff]
	vpmax.u16	d16, d16, d17
@ CHECK: vpmax.u32	d16, d16, d17   @ encoding: [0xa1,0x0a,0x60,0xff]
	vpmax.u32	d16, d16, d17
@ CHECK: vpmax.f32	d16, d16, d17   @ encoding: [0xa1,0x0f,0x40,0xff]
	vpmax.f32	d16, d16, d17
