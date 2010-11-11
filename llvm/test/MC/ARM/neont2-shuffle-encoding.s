@ RUN: llvm-mc -mcpu=cortex-a8 -triple thumb-unknown-unkown -show-encoding < %s | FileCheck %s

.code 16

@ CHECK: vext.8	d16, d17, d16, #3       @ encoding: [0xa0,0x03,0xf1,0xef]
	vext.8	d16, d17, d16, #3
@ CHECK: vext.8	d16, d17, d16, #5       @ encoding: [0xa0,0x05,0xf1,0xef]
	vext.8	d16, d17, d16, #5
@ CHECK: vext.8	q8, q9, q8, #3          @ encoding: [0xe0,0x03,0xf2,0xef]
	vext.8	q8, q9, q8, #3
@ CHECK: vext.8	q8, q9, q8, #7          @ encoding: [0xe0,0x07,0xf2,0xef]
	vext.8	q8, q9, q8, #7
@ CHECK: vext.16	d16, d17, d16, #3       @ encoding: [0xa0,0x06,0xf1,0xef]
	vext.16	d16, d17, d16, #3
@ CHECK: vext.32	q8, q9, q8, #3          @ encoding: [0xe0,0x0c,0xf2,0xef]
	vext.32	q8, q9, q8, #3
@ CHECK: vtrn.8	d17, d16                @ encoding: [0xa0,0x10,0xf2,0xff]
	vtrn.8	d17, d16
@ CHECK: vtrn.16	d17, d16                @ encoding: [0xa0,0x10,0xf6,0xff]
	vtrn.16	d17, d16
@ CHECK: vtrn.32	d17, d16                @ encoding: [0xa0,0x10,0xfa,0xff]
	vtrn.32	d17, d16
@ CHECK: vtrn.8	q9, q8                  @ encoding: [0xe0,0x20,0xf2,0xff]
	vtrn.8	q9, q8
@ CHECK: vtrn.16	q9, q8                  @ encoding: [0xe0,0x20,0xf6,0xff]
	vtrn.16	q9, q8
@ CHECK: vtrn.32	q9, q8                  @ encoding: [0xe0,0x20,0xfa,0xff]
	vtrn.32	q9, q8
@ CHECK: vuzp.8	d17, d16                @ encoding: [0x20,0x11,0xf2,0xff]
	vuzp.8	d17, d16
@ CHECK: vuzp.16	d17, d16                @ encoding: [0x20,0x11,0xf6,0xff]
	vuzp.16	d17, d16
@ CHECK: vuzp.8	q9, q8                  @ encoding: [0x60,0x21,0xf2,0xff]
	vuzp.8	q9, q8
@ CHECK: vuzp.16	q9, q8                  @ encoding: [0x60,0x21,0xf6,0xff]
	vuzp.16	q9, q8
@ CHECK: vuzp.32	q9, q8                  @ encoding: [0x60,0x21,0xfa,0xff]
	vuzp.32	q9, q8
@ CHECK: vzip.8	d17, d16                @ encoding: [0xa0,0x11,0xf2,0xff]
	vzip.8	d17, d16
@ CHECK: vzip.16	d17, d16                @ encoding: [0xa0,0x11,0xf6,0xff]
	vzip.16	d17, d16
@ CHECK: vzip.8	q9, q8                  @ encoding: [0xe0,0x21,0xf2,0xff]
	vzip.8	q9, q8
@ CHECK: vzip.16	q9, q8                  @ encoding: [0xe0,0x21,0xf6,0xff]
	vzip.16	q9, q8
@ CHECK: vzip.32	q9, q8                  @ encoding: [0xe0,0x21,0xfa,0xff]
	vzip.32	q9, q8
