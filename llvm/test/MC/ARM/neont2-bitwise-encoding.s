@ RUN: llvm-mc -mcpu=cortex-a8 -triple thumb-unknown-unkown -show-encoding < %s | FileCheck %s
@ XFAIL: *

.code 16

@ CHECK: vand	d16, d17, d16           @ encoding: [0xb0,0x01,0x41,0xef]
	vand	d16, d17, d16
@ CHECK: vand	q8, q8, q9              @ encoding: [0xf2,0x01,0x40,0xef]
	vand	q8, q8, q9

@ CHECK: veor	d16, d17, d16           @ encoding: [0xb0,0x01,0x41,0xff]
	veor	d16, d17, d16
@ CHECK: veor	q8, q8, q9              @ encoding: [0xf2,0x01,0x40,0xff]
	veor	q8, q8, q9

@ CHECK: vorr	d16, d17, d16           @ encoding: [0xb0,0x01,0x61,0xef]
	vorr	d16, d17, d16
@ CHECK: vorr	q8, q8, q9              @ encoding: [0xf2,0x01,0x60,0xef]
	vorr	q8, q8, q9
@ CHECK: vorr.i32	d16, #0x1000000 @ encoding: [0x11,0x07,0xc0,0xef]
  vorr.i32	d16, #0x1000000
@ CHECK: vorr.i32	q8, #0x1000000  @ encoding: [0x51,0x07,0xc0,0xef]
  vorr.i32	q8, #0x1000000
@ CHECK: vorr.i32	q8, #0x0        @ encoding: [0x50,0x01,0xc0,0xef]
  vorr.i32	q8, #0x0

@ CHECK: vbic	d16, d17, d16           @ encoding: [0xb0,0x01,0x51,0xef]
	vbic	d16, d17, d16
@ CHECK: vbic	q8, q8, q9              @ encoding: [0xf2,0x01,0x50,0xef]
	vbic	q8, q8, q9
@ CHECK: vbic.i32	d16, #0xFF000000 @ encoding: [0x3f,0x07,0xc7,0xff]
  vbic.i32	d16, #0xFF000000
@ CHECK: vbic.i32	q8, #0xFF000000 @ encoding: [0x7f,0x07,0xc7,0xff]
  vbic.i32	q8, #0xFF000000

@ CHECK: vorn	d16, d17, d16           @ encoding: [0xb0,0x01,0x71,0xef]
	vorn	d16, d17, d16
@ CHECK: vorn	q8, q8, q9              @ encoding: [0xf2,0x01,0x70,0xef]
	vorn	q8, q8, q9

@ CHECK: vmvn	d16, d16                @ encoding: [0xa0,0x05,0xf0,0xff]
	vmvn	d16, d16
@ CHECK: vmvn	q8, q8                  @ encoding: [0xe0,0x05,0xf0,0xff]
	vmvn	q8, q8

@ CHECK: vbsl	d18, d17, d16           @ encoding: [0xb0,0x21,0x51,0xff]
	vbsl	d18, d17, d16
@ CHECK: vbsl	q8, q10, q9             @ encoding: [0xf2,0x01,0x54,0xff]
	vbsl	q8, q10, q9
