@ RUN: llvm-mc -mcpu=cortex-a8 -triple arm-unknown-unkown -show-encoding < %s | FileCheck %s

@ CHECK: vsub.i8	d16, d17, d16           @ encoding: [0xa0,0x08,0x41,0xf3]
	vsub.i8	d16, d17, d16
@ CHECK: vsub.i16	d16, d17, d16   @ encoding: [0xa0,0x08,0x51,0xf3]
	vsub.i16	d16, d17, d16
@ CHECK: vsub.i32	d16, d17, d16   @ encoding: [0xa0,0x08,0x61,0xf3]
	vsub.i32	d16, d17, d16
@ CHECK: vsub.i64	d16, d17, d16   @ encoding: [0xa0,0x08,0x71,0xf3]
	vsub.i64	d16, d17, d16
@ CHECK: vsub.f32	d16, d16, d17   @ encoding: [0xa1,0x0d,0x60,0xf2]
	vsub.f32	d16, d16, d17
@ CHECK: vsub.i8	q8, q8, q9              @ encoding: [0xe2,0x08,0x40,0xf3]
	vsub.i8	q8, q8, q9
@ CHECK: vsub.i16	q8, q8, q9      @ encoding: [0xe2,0x08,0x50,0xf3]
	vsub.i16	q8, q8, q9
@ CHECK: vsub.i32	q8, q8, q9      @ encoding: [0xe2,0x08,0x60,0xf3]
	vsub.i32	q8, q8, q9
@ CHECK: vsub.i64	q8, q8, q9      @ encoding: [0xe2,0x08,0x70,0xf3]
	vsub.i64	q8, q8, q9
@ CHECK: vsub.f32	q8, q8, q9      @ encoding: [0xe2,0x0d,0x60,0xf2]
	vsub.f32	q8, q8, q9
@ CHECK: vsubl.s8	q8, d17, d16    @ encoding: [0xa0,0x02,0xc1,0xf2]
	vsubl.s8	q8, d17, d16
@ CHECK: vsubl.s16	q8, d17, d16    @ encoding: [0xa0,0x02,0xd1,0xf2]
	vsubl.s16	q8, d17, d16
@ CHECK: vsubl.s32	q8, d17, d16    @ encoding: [0xa0,0x02,0xe1,0xf2]
	vsubl.s32	q8, d17, d16
@ CHECK: vsubl.u8	q8, d17, d16    @ encoding: [0xa0,0x02,0xc1,0xf3]
	vsubl.u8	q8, d17, d16
@ CHECK: vsubl.u16	q8, d17, d16    @ encoding: [0xa0,0x02,0xd1,0xf3]
	vsubl.u16	q8, d17, d16
@ CHECK: vsubl.u32	q8, d17, d16    @ encoding: [0xa0,0x02,0xe1,0xf3]
	vsubl.u32	q8, d17, d16
@ CHECK: vsubw.s8	q8, q8, d18     @ encoding: [0xa2,0x03,0xc0,0xf2]
	vsubw.s8	q8, q8, d18
@ CHECK: vsubw.s16	q8, q8, d18     @ encoding: [0xa2,0x03,0xd0,0xf2]
	vsubw.s16	q8, q8, d18
@ CHECK: vsubw.s32	q8, q8, d18     @ encoding: [0xa2,0x03,0xe0,0xf2]
	vsubw.s32	q8, q8, d18
@ CHECK: vsubw.u8	q8, q8, d18     @ encoding: [0xa2,0x03,0xc0,0xf3]
	vsubw.u8	q8, q8, d18
@ CHECK: vsubw.u16	q8, q8, d18     @ encoding: [0xa2,0x03,0xd0,0xf3]
	vsubw.u16	q8, q8, d18
@ CHECK: vsubw.u32	q8, q8, d18     @ encoding: [0xa2,0x03,0xe0,0xf3]
	vsubw.u32	q8, q8, d18
@ CHECK: vhsub.s8	d16, d16, d17   @ encoding: [0xa1,0x02,0x40,0xf2]
	vhsub.s8	d16, d16, d17
@ CHECK: vhsub.s16	d16, d16, d17   @ encoding: [0xa1,0x02,0x50,0xf2]
	vhsub.s16	d16, d16, d17
@ CHECK: vhsub.s32	d16, d16, d17   @ encoding: [0xa1,0x02,0x60,0xf2]
	vhsub.s32	d16, d16, d17
@ CHECK: vhsub.u8	d16, d16, d17   @ encoding: [0xa1,0x02,0x40,0xf3]
	vhsub.u8	d16, d16, d17
@ CHECK: vhsub.u16	d16, d16, d17   @ encoding: [0xa1,0x02,0x50,0xf3]
	vhsub.u16	d16, d16, d17
@ CHECK: vhsub.u32	d16, d16, d17   @ encoding: [0xa1,0x02,0x60,0xf3]
	vhsub.u32	d16, d16, d17
@ CHECK: vhsub.s8	q8, q8, q9      @ encoding: [0xe2,0x02,0x40,0xf2]
	vhsub.s8	q8, q8, q9
@ CHECK: vhsub.s16	q8, q8, q9      @ encoding: [0xe2,0x02,0x50,0xf2]
	vhsub.s16	q8, q8, q9
@ CHECK: vhsub.s32	q8, q8, q9      @ encoding: [0xe2,0x02,0x60,0xf2]
	vhsub.s32	q8, q8, q9
@ CHECK: vqsub.s8	d16, d16, d17   @ encoding: [0xb1,0x02,0x40,0xf2]
	vqsub.s8	d16, d16, d17
@ CHECK: vqsub.s16	d16, d16, d17   @ encoding: [0xb1,0x02,0x50,0xf2]
	vqsub.s16	d16, d16, d17
@ CHECK: vqsub.s32	d16, d16, d17   @ encoding: [0xb1,0x02,0x60,0xf2]
	vqsub.s32	d16, d16, d17
@ CHECK: vqsub.s64	d16, d16, d17   @ encoding: [0xb1,0x02,0x70,0xf2]
	vqsub.s64	d16, d16, d17
@ CHECK: vqsub.u8	d16, d16, d17   @ encoding: [0xb1,0x02,0x40,0xf3]
	vqsub.u8	d16, d16, d17
@ CHECK: vqsub.u16	d16, d16, d17   @ encoding: [0xb1,0x02,0x50,0xf3]
	vqsub.u16	d16, d16, d17
@ CHECK: vqsub.u32	d16, d16, d17   @ encoding: [0xb1,0x02,0x60,0xf3]
	vqsub.u32	d16, d16, d17
@ CHECK: vqsub.u64	d16, d16, d17   @ encoding: [0xb1,0x02,0x70,0xf3]
	vqsub.u64	d16, d16, d17
@ CHECK: vqsub.s8	q8, q8, q9      @ encoding: [0xf2,0x02,0x40,0xf2]
	vqsub.s8	q8, q8, q9
@ CHECK: vqsub.s16	q8, q8, q9      @ encoding: [0xf2,0x02,0x50,0xf2]
	vqsub.s16	q8, q8, q9
@ CHECK: vqsub.s32	q8, q8, q9      @ encoding: [0xf2,0x02,0x60,0xf2]
	vqsub.s32	q8, q8, q9
@ CHECK: vqsub.s64	q8, q8, q9      @ encoding: [0xf2,0x02,0x70,0xf2]
	vqsub.s64	q8, q8, q9
@ CHECK: vqsub.u8	q8, q8, q9      @ encoding: [0xf2,0x02,0x40,0xf3]
	vqsub.u8	q8, q8, q9
@ CHECK: vqsub.u16	q8, q8, q9      @ encoding: [0xf2,0x02,0x50,0xf3]
	vqsub.u16	q8, q8, q9
@ CHECK: vqsub.u32	q8, q8, q9      @ encoding: [0xf2,0x02,0x60,0xf3]
	vqsub.u32	q8, q8, q9
@ CHECK: vqsub.u64	q8, q8, q9      @ encoding: [0xf2,0x02,0x70,0xf3]
	vqsub.u64	q8, q8, q9
@ CHECK: vsubhn.i16	d16, q8, q9     @ encoding: [0xa2,0x06,0xc0,0xf2]
	vsubhn.i16	d16, q8, q9
@ CHECK: vsubhn.i32	d16, q8, q9     @ encoding: [0xa2,0x06,0xd0,0xf2]
	vsubhn.i32	d16, q8, q9
@ CHECK: vsubhn.i64	d16, q8, q9     @ encoding: [0xa2,0x06,0xe0,0xf2]
	vsubhn.i64	d16, q8, q9
@ CHECK: vrsubhn.i16	d16, q8, q9     @ encoding: [0xa2,0x06,0xc0,0xf3]
	vrsubhn.i16	d16, q8, q9
@ CHECK: vrsubhn.i32	d16, q8, q9     @ encoding: [0xa2,0x06,0xd0,0xf3]
	vrsubhn.i32	d16, q8, q9
@ CHECK: vrsubhn.i64	d16, q8, q9     @ encoding: [0xa2,0x06,0xe0,0xf3]
	vrsubhn.i64	d16, q8, q9
