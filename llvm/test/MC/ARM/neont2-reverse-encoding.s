@ RUN: llvm-mc -mcpu=cortex-a8 -triple thumb-unknown-unkown -show-encoding < %s | FileCheck %s

@ CHECK: vrev64.8	d16, d16        @ encoding: [0x20,0x00,0xf0,0xff]
	vrev64.8	d16, d16
@ CHECK: vrev64.16	d16, d16        @ encoding: [0x20,0x00,0xf4,0xff]
	vrev64.16	d16, d16
@ CHECK: vrev64.32	d16, d16        @ encoding: [0x20,0x00,0xf8,0xff]
	vrev64.32	d16, d16
@ CHECK: vrev64.8	q8, q8          @ encoding: [0x60,0x00,0xf0,0xff]
	vrev64.8	q8, q8
@ CHECK: vrev64.16	q8, q8          @ encoding: [0x60,0x00,0xf4,0xff]
	vrev64.16	q8, q8
@ CHECK: vrev64.32	q8, q8          @ encoding: [0x60,0x00,0xf8,0xff]
	vrev64.32	q8, q8
@ CHECK: vrev32.8	d16, d16        @ encoding: [0xa0,0x00,0xf0,0xff]
	vrev32.8	d16, d16
@ CHECK: vrev32.16	d16, d16        @ encoding: [0xa0,0x00,0xf4,0xff]
	vrev32.16	d16, d16
@ CHECK: vrev32.8	q8, q8          @ encoding: [0xe0,0x00,0xf0,0xff]
	vrev32.8	q8, q8
@ CHECK: vrev32.16	q8, q8          @ encoding: [0xe0,0x00,0xf4,0xff]
	vrev32.16	q8, q8
@ CHECK: vrev16.8	d16, d16        @ encoding: [0x20,0x01,0xf0,0xff]
	vrev16.8	d16, d16
@ CHECK: vrev16.8	q8, q8          @ encoding: [0x60,0x01,0xf0,0xff]
	vrev16.8	q8, q8
