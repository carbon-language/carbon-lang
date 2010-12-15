@ RUN: llvm-mc -mcpu=cortex-a8 -triple thumb-unknown-unknown -show-encoding < %s | FileCheck %s
@ XFAIL: *
@ NOTE: This currently fails because the ASM parser doesn't parse vabal.

.code 16

@ CHECK: vabd.s8	d16, d16, d17           @ encoding: [0xa1,0x07,0x40,0xef]
	vabd.s8	d16, d16, d17
@ CHECK: vabd.s16	d16, d16, d17   @ encoding: [0xa1,0x07,0x50,0xef]
	vabd.s16	d16, d16, d17
@ CHECK: vabd.s32	d16, d16, d17   @ encoding: [0xa1,0x07,0x60,0xef]
	vabd.s32	d16, d16, d17
@ CHECK: vabd.u8	d16, d16, d17           @ encoding: [0xa1,0x07,0x40,0xff]
	vabd.u8	d16, d16, d17
@ CHECK: vabd.u16	d16, d16, d17   @ encoding: [0xa1,0x07,0x50,0xff]
	vabd.u16	d16, d16, d17
  @ CHECK: vabd.u32	d16, d16, d17   @ encoding: [0xa1,0x07,0x60,0xff]
	vabd.u32	d16, d16, d17
@ CHECK: vabd.f32	d16, d16, d17   @ encoding: [0xa1,0x0d,0x60,0xff]
	vabd.f32	d16, d16, d17
@ CHECK: vabd.s8	q8, q8, q9              @ encoding: [0xe2,0x07,0x40,0xef]
	vabd.s8	q8, q8, q9
@ CHECK: vabd.s16	q8, q8, q9      @ encoding: [0xe2,0x07,0x50,0xef]
	vabd.s16	q8, q8, q9
@ CHECK: vabd.s32	q8, q8, q9      @ encoding: [0xe2,0x07,0x60,0xef]
	vabd.s32	q8, q8, q9
@ CHECK: vabd.u8	q8, q8, q9              @ encoding: [0xe2,0x07,0x40,0xff]
	vabd.u8	q8, q8, q9
@ CHECK: vabd.u16	q8, q8, q9      @ encoding: [0xe2,0x07,0x50,0xff]
	vabd.u16	q8, q8, q9
@ CHECK: vabd.u32	q8, q8, q9      @ encoding: [0xe2,0x07,0x60,0xff]
	vabd.u32	q8, q8, q9
@ CHECK: vabd.f32	q8, q8, q9      @ encoding: [0xe2,0x0d,0x60,0xff]
	vabd.f32	q8, q8, q9

@ CHECK: vabdl.s8	q8, d16, d17    @ encoding: [0xa1,0x07,0xc0,0xef]
	vabdl.s8	q8, d16, d17
@ CHECK: vabdl.s16	q8, d16, d17    @ encoding: [0xa1,0x07,0xd0,0xef]
	vabdl.s16	q8, d16, d17
@ CHECK: vabdl.s32	q8, d16, d17    @ encoding: [0xa1,0x07,0xe0,0xef]
	vabdl.s32	q8, d16, d17
@ CHECK: vabdl.u8	q8, d16, d17    @ encoding: [0xa1,0x07,0xc0,0xff]
	vabdl.u8	q8, d16, d17
@ CHECK: vabdl.u16	q8, d16, d17    @ encoding: [0xa1,0x07,0xd0,0xff]
	vabdl.u16	q8, d16, d17
@ CHECK: vabdl.u32	q8, d16, d17    @ encoding: [0xa1,0x07,0xe0,0xff]
	vabdl.u32	q8, d16, d17

@ CHECK: vaba.s8	d16, d18, d17           @ encoding: [0xb1,0x07,0x42,0xef]
	vaba.s8	d16, d18, d17
@ CHECK: vaba.s16	d16, d18, d17   @ encoding: [0xb1,0x07,0x52,0xef]
	vaba.s16	d16, d18, d17
@ CHECK: vaba.s32	d16, d18, d17   @ encoding: [0xb1,0x07,0x62,0xef]
	vaba.s32	d16, d18, d17
@ CHECK: vaba.u8	d16, d18, d17           @ encoding: [0xb1,0x07,0x42,0xff]
	vaba.u8	d16, d18, d17
@ CHECK: vaba.u16	d16, d18, d17   @ encoding: [0xb1,0x07,0x52,0xff]
	vaba.u16	d16, d18, d17
@ CHECK: vaba.u32	d16, d18, d17   @ encoding: [0xb1,0x07,0x62,0xff]
	vaba.u32	d16, d18, d17
@ CHECK: vaba.s8	q9, q8, q10             @ encoding: [0xf4,0x27,0x40,0xef]
	vaba.s8	q9, q8, q10
@ CHECK: vaba.s16	q9, q8, q10     @ encoding: [0xf4,0x27,0x50,0xef]
	vaba.s16	q9, q8, q10
@ CHECK: vaba.s32	q9, q8, q10     @ encoding: [0xf4,0x27,0x60,0xef]
	vaba.s32	q9, q8, q10
@ CHECK: vaba.u8	q9, q8, q10             @ encoding: [0xf4,0x27,0x40,0xff]
	vaba.u8	q9, q8, q10
@ CHECK: vaba.u16	q9, q8, q10     @ encoding: [0xf4,0x27,0x50,0xff]
	vaba.u16	q9, q8, q10
@ CHECK: vaba.u32	q9, q8, q10     @ encoding: [0xf4,0x27,0x60,0xff]
	vaba.u32	q9, q8, q10

@ CHECK: vabal.s8	q8, d19, d18    @ encoding: [0xa2,0x05,0xc3,0xef]
	vabal.s8	q8, d19, d18
@ CHECK: vabal.s16	q8, d19, d18    @ encoding: [0xa2,0x05,0xd3,0xef]
	vabal.s16	q8, d19, d18
@ CHECK: vabal.s32	q8, d19, d18    @ encoding: [0xa2,0x05,0xe3,0xef]
	vabal.s32	q8, d19, d18
@ CHECK: vabal.u8	q8, d19, d18    @ encoding: [0xa2,0x05,0xc3,0xff]
	vabal.u8	q8, d19, d18
@ CHECK: 	vabal.u16	q8, d19, d18    @ encoding: [0xa2,0x05,0xd3,0xff]
	vabal.u16	q8, d19, d18
@ CHECK: vabal.u32	q8, d19, d18    @ encoding: [0xa2,0x05,0xe3,0xff]
	vabal.u32	q8, d19, d18

