@ RUN: llvm-mc -mcpu=cortex-a8 -triple thumb-unknown-unknown -show-encoding < %s | FileCheck %s
@ XFAIL: *

.code 16

	vand	d16, d17, d16
	vand	q8, q8, q9

@ CHECK: vand	d16, d17, d16           @ encoding: [0xb0,0x01,0x41,0xef]
@ CHECK: vand	q8, q8, q9              @ encoding: [0xf2,0x01,0x40,0xef]


	veor	d16, d17, d16
	veor	q8, q8, q9

@ CHECK: veor	d16, d17, d16           @ encoding: [0xb0,0x01,0x41,0xff]
@ CHECK: veor	q8, q8, q9              @ encoding: [0xf2,0x01,0x40,0xff]


	vorr	d16, d17, d16
	vorr	q8, q8, q9
	vorr.i32	d16, #0x1000000
	vorr.i32	q8, #0x1000000
	vorr.i32	q8, #0x0

@ CHECK: vorr	d16, d17, d16           @ encoding: [0xb0,0x01,0x61,0xef]
@ CHECK: vorr	q8, q8, q9              @ encoding: [0xf2,0x01,0x60,0xef]
@ CHECK: vorr.i32	d16, #0x1000000 @ encoding: [0x11,0x07,0xc0,0xef]
@ CHECK: vorr.i32	q8, #0x1000000  @ encoding: [0x51,0x07,0xc0,0xef]
@ CHECK: vorr.i32	q8, #0x0        @ encoding: [0x50,0x01,0xc0,0xef]


	vbic	d16, d17, d16
	vbic	q8, q8, q9
	vbic.i32	d16, #0xFF000000
	vbic.i32	q8, #0xFF000000

@ CHECK: vbic	d16, d17, d16           @ encoding: [0xb0,0x01,0x51,0xef]
@ CHECK: vbic	q8, q8, q9              @ encoding: [0xf2,0x01,0x50,0xef]
@ CHECK: vbic.i32	d16, #0xFF000000 @ encoding: [0x3f,0x07,0xc7,0xff]
@ CHECK: vbic.i32	q8, #0xFF000000 @ encoding: [0x7f,0x07,0xc7,0xff]


	vorn	d16, d17, d16
	vorn	q8, q8, q9

@ CHECK: vorn	d16, d17, d16           @ encoding: [0xb0,0x01,0x71,0xef]
@ CHECK: vorn	q8, q8, q9              @ encoding: [0xf2,0x01,0x70,0xef]


	vmvn	d16, d16
	vmvn	q8, q8

@ CHECK: vmvn	d16, d16                @ encoding: [0xa0,0x05,0xf0,0xff]
@ CHECK: vmvn	q8, q8                  @ encoding: [0xe0,0x05,0xf0,0xff]


	vbsl	d18, d17, d16
	vbsl	q8, q10, q9

@ CHECK: vbsl	d18, d17, d16           @ encoding: [0xb0,0x21,0x51,0xff]
@ CHECK: vbsl	q8, q10, q9             @ encoding: [0xf2,0x01,0x54,0xff]
