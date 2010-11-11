@ RUN: llvm-mc -mcpu=cortex-a8 -triple thumb-unknown-unkown -show-encoding < %s | FileCheck %s

.code 16

@ CHECK: vshl.u8	d16, d17, d16           @ encoding: [0xa1,0x04,0x40,0xff]
	vshl.u8	d16, d17, d16
@ CHECK: vshl.u16	d16, d17, d16   @ encoding: [0xa1,0x04,0x50,0xff]
	vshl.u16	d16, d17, d16
@ CHECK: vshl.u32	d16, d17, d16   @ encoding: [0xa1,0x04,0x60,0xff]
	vshl.u32	d16, d17, d16
@ CHECK: vshl.u64	d16, d17, d16   @ encoding: [0xa1,0x04,0x70,0xff]
	vshl.u64	d16, d17, d16
@ CHECK: vshl.i8	d16, d16, #7            @ encoding: [0x30,0x05,0xcf,0xef]
	vshl.i8	d16, d16, #7
@ CHECK: vshl.i16	d16, d16, #15   @ encoding: [0x30,0x05,0xdf,0xef]
	vshl.i16	d16, d16, #15
@ CHECK: vshl.i32	d16, d16, #31   @ encoding: [0x30,0x05,0xff,0xef]
	vshl.i32	d16, d16, #31
@ CHECK: vshl.i64	d16, d16, #63   @ encoding: [0xb0,0x05,0xff,0xef]
	vshl.i64	d16, d16, #63
@ CHECK: vshl.u8	q8, q9, q8              @ encoding: [0xe2,0x04,0x40,0xff]
	vshl.u8	q8, q9, q8
@ CHECK: vshl.u16	q8, q9, q8      @ encoding: [0xe2,0x04,0x50,0xff]
	vshl.u16	q8, q9, q8
@ CHECK: vshl.u32	q8, q9, q8      @ encoding: [0xe2,0x04,0x60,0xff]
	vshl.u32	q8, q9, q8
@ CHECK: vshl.u64	q8, q9, q8      @ encoding: [0xe2,0x04,0x70,0xff]
	vshl.u64	q8, q9, q8
@ CHECK: vshl.i8	q8, q8, #7              @ encoding: [0x70,0x05,0xcf,0xef]
	vshl.i8	q8, q8, #7
@ CHECK: vshl.i16	q8, q8, #15     @ encoding: [0x70,0x05,0xdf,0xef]
	vshl.i16	q8, q8, #15
@ CHECK: vshl.i32	q8, q8, #31     @ encoding: [0x70,0x05,0xff,0xef]
	vshl.i32	q8, q8, #31
@ CHECK: vshl.i64	q8, q8, #63     @ encoding: [0xf0,0x05,0xff,0xef]
	vshl.i64	q8, q8, #63
@ CHECK: vshr.u8	d16, d16, #8            @ encoding: [0x30,0x00,0xc8,0xff]
	vshr.u8	d16, d16, #8
@ CHECK: vshr.u16	d16, d16, #16   @ encoding: [0x30,0x00,0xd0,0xff]
	vshr.u16	d16, d16, #16
@ CHECK: vshr.u32	d16, d16, #32   @ encoding: [0x30,0x00,0xe0,0xff]
	vshr.u32	d16, d16, #32
@ CHECK: vshr.u64	d16, d16, #64   @ encoding: [0xb0,0x00,0xc0,0xff]
	vshr.u64	d16, d16, #64
@ CHECK: vshr.u8	q8, q8, #8              @ encoding: [0x70,0x00,0xc8,0xff]
	vshr.u8	q8, q8, #8
@ CHECK: vshr.u16	q8, q8, #16     @ encoding: [0x70,0x00,0xd0,0xff]
	vshr.u16	q8, q8, #16
@ CHECK: vshr.u32	q8, q8, #32     @ encoding: [0x70,0x00,0xe0,0xff]
	vshr.u32	q8, q8, #32
@ CHECK: vshr.u64	q8, q8, #64     @ encoding: [0xf0,0x00,0xc0,0xff]
	vshr.u64	q8, q8, #64
@ CHECK: vshr.s8	d16, d16, #8            @ encoding: [0x30,0x00,0xc8,0xef]
	vshr.s8	d16, d16, #8
@ CHECK: vshr.s16	d16, d16, #16   @ encoding: [0x30,0x00,0xd0,0xef]
	vshr.s16	d16, d16, #16
@ CHECK: vshr.s32	d16, d16, #32   @ encoding: [0x30,0x00,0xe0,0xef]
	vshr.s32	d16, d16, #32
@ CHECK: vshr.s64	d16, d16, #64   @ encoding: [0xb0,0x00,0xc0,0xef]
	vshr.s64	d16, d16, #64
@ CHECK: vshr.s8	q8, q8, #8              @ encoding: [0x70,0x00,0xc8,0xef]
	vshr.s8	q8, q8, #8
@ CHECK: vshr.s16	q8, q8, #16     @ encoding: [0x70,0x00,0xd0,0xef]
	vshr.s16	q8, q8, #16
@ CHECK: vshr.s32	q8, q8, #32     @ encoding: [0x70,0x00,0xe0,0xef]
	vshr.s32	q8, q8, #32
@ CHECK: vshr.s64	q8, q8, #64     @ encoding: [0xf0,0x00,0xc0,0xef]
	vshr.s64	q8, q8, #64
@ CHECK: vshll.s8	q8, d16, #7     @ encoding: [0x30,0x0a,0xcf,0xef]
	vshll.s8	q8, d16, #7
@ CHECK: vshll.s16	q8, d16, #15    @ encoding: [0x30,0x0a,0xdf,0xef]
	vshll.s16	q8, d16, #15
@ CHECK: vshll.s32	q8, d16, #31    @ encoding: [0x30,0x0a,0xff,0xef]
	vshll.s32	q8, d16, #31
@ CHECK: vshll.u8	q8, d16, #7     @ encoding: [0x30,0x0a,0xcf,0xff]
	vshll.u8	q8, d16, #7
@ CHECK: vshll.u16	q8, d16, #15    @ encoding: [0x30,0x0a,0xdf,0xff]
	vshll.u16	q8, d16, #15
@ CHECK: vshll.u32	q8, d16, #31    @ encoding: [0x30,0x0a,0xff,0xff]
	vshll.u32	q8, d16, #31
@ CHECK: vshll.i8	q8, d16, #8     @ encoding: [0x20,0x03,0xf2,0xff]
	vshll.i8	q8, d16, #8
@ CHECK: vshll.i16	q8, d16, #16    @ encoding: [0x20,0x03,0xf6,0xff]
	vshll.i16	q8, d16, #16
@ CHECK: vshll.i32	q8, d16, #32    @ encoding: [0x20,0x03,0xfa,0xff]
	vshll.i32	q8, d16, #32
@ CHECK: vshrn.i16	d16, q8, #8     @ encoding: [0x30,0x08,0xc8,0xef]
	vshrn.i16	d16, q8, #8
@ CHECK: vshrn.i32	d16, q8, #16    @ encoding: [0x30,0x08,0xd0,0xef]
	vshrn.i32	d16, q8, #16
@ CHECK: vshrn.i64	d16, q8, #32    @ encoding: [0x30,0x08,0xe0,0xef]
	vshrn.i64	d16, q8, #32
@ CHECK: vrshl.s8	d16, d17, d16   @ encoding: [0xa1,0x05,0x40,0xef]
	vrshl.s8	d16, d17, d16
@ CHECK: vrshl.s16	d16, d17, d16   @ encoding: [0xa1,0x05,0x50,0xef]
	vrshl.s16	d16, d17, d16
@ CHECK: vrshl.s32	d16, d17, d16   @ encoding: [0xa1,0x05,0x60,0xef]
	vrshl.s32	d16, d17, d16
@ CHECK: vrshl.s64	d16, d17, d16   @ encoding: [0xa1,0x05,0x70,0
	vrshl.s64	d16, d17, d16
@ CHECK: vrshl.u8	d16, d17, d16   @ encoding: [0xa1,0x05,0x40,0xff]
	vrshl.u8	d16, d17, d16
@ CHECK: vrshl.u16	d16, d17, d16   @ encoding: [0xa1,0x05,0x50,0xff]
	vrshl.u16	d16, d17, d16
@ CHECK: vrshl.u32	d16, d17, d16   @ encoding: [0xa1,0x05,0x60,0xff]
	vrshl.u32	d16, d17, d16
@ CHECK: vrshl.u64	d16, d17, d16   @ encoding: [0xa1,0x05,0x70,0xff]
	vrshl.u64	d16, d17, d16
@ CHECK: vrshl.s8	q8, q9, q8      @ encoding: [0xe2,0x05,0x40,0xef]
	vrshl.s8	q8, q9, q8
@ CHECK: vrshl.s16	q8, q9, q8      @ encoding: [0xe2,0x05,0x50,0xef]
	vrshl.s16	q8, q9, q8
@ CHECK: vrshl.s32	q8, q9, q8      @ encoding: [0xe2,0x05,0x60,0xef]
	vrshl.s32	q8, q9, q8
@ CHECK: vrshl.s64	q8, q9, q8      @ encoding: [0xe2,0x05,0x70,0xef]
	vrshl.s64	q8, q9, q8
@ CHECK: vrshl.u8	q8, q9, q8      @ encoding: [0xe2,0x05,0x40,0xff]
	vrshl.u8	q8, q9, q8
@ CHECK: vrshl.u16	q8, q9, q8      @ encoding: [0xe2,0x05,0x50,0xff]
	vrshl.u16	q8, q9, q8
@ CHECK: vrshl.u32	q8, q9, q8      @ encoding: [0xe2,0x05,0x60,0xff]
	vrshl.u32	q8, q9, q8
@ CHECK: vrshl.u64	q8, q9, q8      @ encoding: [0xe2,0x05,0x70,0xff]
	vrshl.u64	q8, q9, q8
@ CHECK: vrshr.s8	d16, d16, #8    @ encoding: [0x30,0x02,0xc8,0xef]
	vrshr.s8	d16, d16, #8
@ CHECK: vrshr.s16	d16, d16, #16   @ encoding: [0x30,0x02,0xd0,0xef]
	vrshr.s16	d16, d16, #16
@ CHECK: vrshr.s32	d16, d16, #32   @ encoding: [0x30,0x02,0xe0,0xef]
	vrshr.s32	d16, d16, #32
@ CHECK: vrshr.s64	d16, d16, #64   @ encoding: [0xb0,0x02,0xc0,0xef]
	vrshr.s64	d16, d16, #64
@ CHECK: vrshr.u8	d16, d16, #8    @ encoding: [0x30,0x02,0xc8,0xff]
	vrshr.u8	d16, d16, #8
@ CHECK: vrshr.u16	d16, d16, #16   @ encoding: [0x30,0x02,0xd0,0xff]
	vrshr.u16	d16, d16, #16
@ CHECK: vrshr.u32	d16, d16, #32   @ encoding: [0x30,0x02,0xe0,0xff]
	vrshr.u32	d16, d16, #32
@ CHECK: vrshr.u64	d16, d16, #64   @ encoding: [0xb0,0x02,0xc0,0xff]
	vrshr.u64	d16, d16, #64
@ CHECK: vrshr.s8	q8, q8, #8      @ encoding: [0x70,0x02,0xc8,0xef]
	vrshr.s8	q8, q8, #8
@ CHECK: vrshr.s16	q8, q8, #16     @ encoding: [0x70,0x02,0xd0,0xef]
	vrshr.s16	q8, q8, #16
@ CHECK: vrshr.s32	q8, q8, #32     @ encoding: [0x70,0x02,0xe0,0xef]
	vrshr.s32	q8, q8, #32
@ CHECK: vrshr.s64	q8, q8, #64     @ encoding: [0xf0,0x02,0xc0,0xef]
	vrshr.s64	q8, q8, #64
@ CHECK: vrshr.u8	q8, q8, #8      @ encoding: [0x70,0x02,0xc8,0xff]
	vrshr.u8	q8, q8, #8
@ CHECK: vrshr.u16	q8, q8, #16     @ encoding: [0x70,0x02,0xd0,0xff]
	vrshr.u16	q8, q8, #16
@ CHECK: vrshr.u32	q8, q8, #32     @ encoding: [0x70,0x02,0xe0,0xff]
	vrshr.u32	q8, q8, #32
@ CHECK: vrshr.u64	q8, q8, #64     @ encoding: [0xf0,0x02,0xc0,0xff]
	vrshr.u64	q8, q8, #64
@ CHECK: vrshrn.i16	d16, q8, #8     @ encoding: [0x70,0x08,0xc8,0xef]
	vrshrn.i16	d16, q8, #8
@ CHECK: vrshrn.i32	d16, q8, #16    @ encoding: [0x70,0x08,0xd0,0xef]
	vrshrn.i32	d16, q8, #16
@ CHECK: vrshrn.i64	d16, q8, #32    @ encoding: [0x70,0x08,0xe0,0xef]
	vrshrn.i64	d16, q8, #32
