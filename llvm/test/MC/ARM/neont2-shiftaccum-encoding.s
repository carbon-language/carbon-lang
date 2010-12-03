@ RUN: llvm-mc -mcpu=cortex-a8 -triple thumb-unknown-unkown -show-encoding < %s | FileCheck %s

.code 16

@ CHECK: vsra.s8	d17, d16, #8            @ encoding: [0xc8,0xef,0x30,0x11]
	vsra.s8	d17, d16, #8
@ CHECK: vsra.s16	d17, d16, #16   @ encoding: [0xd0,0xef,0x30,0x11]
	vsra.s16	d17, d16, #16
@ CHECK: vsra.s32	d17, d16, #32   @ encoding: [0xe0,0xef,0x30,0x11]
	vsra.s32	d17, d16, #32
@ CHECK: vsra.s64	d17, d16, #64   @ encoding: [0xc0,0xef,0xb0,0x11]
	vsra.s64	d17, d16, #64
@ CHECK: vsra.s8	q8, q9, #8              @ encoding: [0xc8,0xef,0x72,0x01]
	vsra.s8	q8, q9, #8
@ CHECK: vsra.s16	q8, q9, #16     @ encoding: [0xd0,0xef,0x72,0x01]
	vsra.s16	q8, q9, #16
@ CHECK: vsra.s32	q8, q9, #32     @ encoding: [0xe0,0xef,0x72,0x01]
	vsra.s32	q8, q9, #32
@ CHECK: vsra.s64	q8, q9, #64     @ encoding: [0xc0,0xef,0xf2,0x01]
	vsra.s64	q8, q9, #64
@ CHECK: vsra.u8	d17, d16, #8            @ encoding: [0xc8,0xff,0x30,0x11]
	vsra.u8	d17, d16, #8
@ CHECK: vsra.u16	d17, d16, #16   @ encoding: [0xd0,0xff,0x30,0x11]
	vsra.u16	d17, d16, #16
@ CHECK: vsra.u32	d17, d16, #32   @ encoding: [0xe0,0xff,0x30,0x11]
	vsra.u32	d17, d16, #32
@ CHECK: vsra.u64	d17, d16, #64   @ encoding: [0xc0,0xff,0xb0,0x11]
	vsra.u64	d17, d16, #64
@ CHECK: vsra.u8	q8, q9, #8              @ encoding: [0xc8,0xff,0x72,0x01]
	vsra.u8	q8, q9, #8
@ CHECK: vsra.u16	q8, q9, #16     @ encoding: [0xd0,0xff,0x72,0x01]
	vsra.u16	q8, q9, #16
@ CHECK: vsra.u32	q8, q9, #32     @ encoding: [0xe0,0xff,0x72,0x01]
	vsra.u32	q8, q9, #32
@ CHECK: vsra.u64	q8, q9, #64     @ encoding: [0xc0,0xff,0xf2,0x01]
	vsra.u64	q8, q9, #64
@ CHECK: vrsra.s8	d17, d16, #8    @ encoding: [0xc8,0xef,0x30,0x13]
	vrsra.s8	d17, d16, #8
@ CHECK: vrsra.s16	d17, d16, #16   @ encoding: [0xd0,0xef,0x30,0x13]
	vrsra.s16	d17, d16, #16
@ CHECK: vrsra.s32	d17, d16, #32   @ encoding: [0xe0,0xef,0x30,0x13]
	vrsra.s32	d17, d16, #32
@ CHECK: vrsra.s64	d17, d16, #64   @ encoding: [0xc0,0xef,0xb0,0x13]
	vrsra.s64	d17, d16, #64
@ CHECK: vrsra.u8	d17, d16, #8    @ encoding: [0xc8,0xff,0x30,0x13]
	vrsra.u8	d17, d16, #8
@ CHECK: vrsra.u16	d17, d16, #16   @ encoding: [0xd0,0xff,0x30,0x13]
	vrsra.u16	d17, d16, #16
@ CHECK: vrsra.u32	d17, d16, #32   @ encoding: [0xe0,0xff,0x30,0x13]
	vrsra.u32	d17, d16, #32
@ CHECK: vrsra.u64	d17, d16, #64   @ encoding: [0xc0,0xff,0xb0,0x13]
	vrsra.u64	d17, d16, #64
@ CHECK: vrsra.s8	q8, q9, #8      @ encoding: [0xc8,0xef,0x72,0x03]
	vrsra.s8	q8, q9, #8
@ CHECK: vrsra.s16	q8, q9, #16     @ encoding: [0xd0,0xef,0x72,0x03]
	vrsra.s16	q8, q9, #16
@ CHECK: vrsra.s32	q8, q9, #32     @ encoding: [0xe0,0xef,0x72,0x03]
	vrsra.s32	q8, q9, #32
@ CHECK: vrsra.s64	q8, q9, #64     @ encoding: [0xc0,0xef,0xf2,0x03]
	vrsra.s64	q8, q9, #64
@ CHECK: vrsra.u8	q8, q9, #8      @ encoding: [0xc8,0xff,0x72,0x03]
	vrsra.u8	q8, q9, #8
@ CHECK: vrsra.u16	q8, q9, #16     @ encoding: [0xd0,0xff,0x72,0x03]
	vrsra.u16	q8, q9, #16
@ CHECK: vrsra.u32	q8, q9, #32     @ encoding: [0xe0,0xff,0x72,0x03]
	vrsra.u32	q8, q9, #32
@ CHECK: vrsra.u64	q8, q9, #64     @ encoding: [0xc0,0xff,0xf2,0x03]
	vrsra.u64	q8, q9, #64
@ CHECK: vsli.8	d17, d16, #7            @ encoding: [0xcf,0xff,0x30,0x15]
	vsli.8	d17, d16, #7
@ CHECK: vsli.16	d17, d16, #15           @ encoding: [0xdf,0xff,0x30,0x15]
	vsli.16	d17, d16, #15
@ CHECK: vsli.32	d17, d16, #31           @ encoding: [0xff,0xff,0x30,0x15]
	vsli.32	d17, d16, #31
@ CHECK: vsli.64	d17, d16, #63           @ encoding: [0xff,0xff,0xb0,0x15]
	vsli.64	d17, d16, #63
@ CHECK: vsli.8	q9, q8, #7              @ encoding: [0xcf,0xff,0x70,0x25]
	vsli.8	q9, q8, #7
@ CHECK: vsli.16	q9, q8, #15             @ encoding: [0xdf,0xff,0x70,0x25]
	vsli.16	q9, q8, #15
@ CHECK: vsli.32	q9, q8, #31             @ encoding: [0xff,0xff,0x70,0x25]
	vsli.32	q9, q8, #31
@ CHECK: vsli.64	q9, q8, #63             @ encoding: [0xff,0xff,0xf0,0x25]
	vsli.64	q9, q8, #63
@ CHECK: vsri.8	d17, d16, #8            @ encoding: [0xc8,0xff,0x30,0x14]
	vsri.8	d17, d16, #8
@ CHECK: vsri.16	d17, d16, #16           @ encoding: [0xd0,0xff,0x30,0x14]
	vsri.16	d17, d16, #16
@ CHECK: vsri.32	d17, d16, #32           @ encoding: [0xe0,0xff,0x30,0x14]
	vsri.32	d17, d16, #32
@ CHECK: vsri.64	d17, d16, #64           @ encoding: [0xc0,0xff,0xb0,0x14]
	vsri.64	d17, d16, #64
@ CHECK: vsri.8	q9, q8, #8              @ encoding: [0xc8,0xff,0x70,0x24]
	vsri.8	q9, q8, #8
@ CHECK: vsri.16	q9, q8, #16             @ encoding: [0xd0,0xff,0x70,0x24]
	vsri.16	q9, q8, #16
@ CHECK: vsri.32	q9, q8, #32             @ encoding: [0xe0,0xff,0x70,0x24]
	vsri.32	q9, q8, #32
@ CHECK: vsri.64	q9, q8, #64             @ encoding: [0xc0,0xff,0xf0,0x24]
	vsri.64	q9, q8, #64
