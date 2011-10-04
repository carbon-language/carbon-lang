@ RUN: llvm-mc -mcpu=cortex-a8 -triple arm-unknown-unknown -show-encoding < %s | FileCheck %s

	vand	d16, d17, d16
	vand	q8, q8, q9

@ CHECK: vand	d16, d17, d16           @ encoding: [0xb0,0x01,0x41,0xf2]
@ CHECK: vand	q8, q8, q9              @ encoding: [0xf2,0x01,0x40,0xf2]

	veor	d16, d17, d16
	veor	q8, q8, q9

@ CHECK: veor	d16, d17, d16           @ encoding: [0xb0,0x01,0x41,0xf3]
@ CHECK: veor	q8, q8, q9              @ encoding: [0xf2,0x01,0x40,0xf3]

	vorr	d16, d17, d16
	vorr	q8, q8, q9

@ CHECK: vorr	d16, d17, d16           @ encoding: [0xb0,0x01,0x61,0xf2]
@ CHECK: vorr	q8, q8, q9              @ encoding: [0xf2,0x01,0x60,0xf2]

	vorr.i32	d16, #0x1000000
	vorr.i32	q8, #0x1000000
	vorr.i32	q8, #0x0

@ FIXME: vorr.i32	d16, #0x1000000 @ encoding: [0x11,0x07,0xc0,0xf2]
@ FIXME: vorr.i32	q8, #0x1000000  @ encoding: [0x51,0x07,0xc0,0xf2]
@ FIXME: vorr.i32	q8, #0x0        @ encoding: [0x50,0x01,0xc0,0xf2]

	vbic	d16, d17, d16
	vbic	q8, q8, q9
	vbic.i32	d16, #0xFF000000
	vbic.i32	q8, #0xFF000000

@ CHECK: vbic	d16, d17, d16           @ encoding: [0xb0,0x01,0x51,0xf2]
@ CHECK: vbic	q8, q8, q9              @ encoding: [0xf2,0x01,0x50,0xf2]
@ FIXME: vbic.i32	d16, #0xFF000000 @ encoding: [0x3f,0x07,0xc7,0xf3]
@ FIXME: vbic.i32	q8, #0xFF000000 @ encoding: [0x7f,0x07,0xc7,0xf3]

	vorn	d16, d17, d16
	vorn	q8, q8, q9

@ CHECK: vorn	d16, d17, d16           @ encoding: [0xb0,0x01,0x71,0xf2]
@ CHECK: vorn	q8, q8, q9              @ encoding: [0xf2,0x01,0x70,0xf2]

	vmvn	d16, d16
	vmvn	q8, q8

@ CHECK: vmvn	d16, d16                @ encoding: [0xa0,0x05,0xf0,0xf3]
@ CHECK: vmvn	q8, q8                  @ encoding: [0xe0,0x05,0xf0,0xf3]

	vbsl	d18, d17, d16
	vbsl	q8, q10, q9

@ CHECK: vbsl	d18, d17, d16           @ encoding: [0xb0,0x21,0x51,0xf3]
@ CHECK: vbsl	q8, q10, q9             @ encoding: [0xf2,0x01,0x54,0xf3]
