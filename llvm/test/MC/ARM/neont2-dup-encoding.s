@ RUN: llvm-mc -mcpu=cortex-a8 -triple thumb-unknown-unkown -show-encoding < %s | FileCheck %s
@ XFAIL: *

.code 16

@ CHECK: vdup.8	d16, r0                 @ encoding: [0x90,0x0b,0xc0,0xee]
	vdup.8	d16, r0
@ CHECK: vdup.16	d16, r0                 @ encoding: [0xb0,0x0b,0x80,0xee]
	vdup.16	d16, r0
@ CHECK: vdup.32	d16, r0                 @ encoding: [0x90,0x0b,0x80,0xee]
	vdup.32	d16, r0
@ CHECK: vdup.8	q8, r0                  @ encoding: [0x90,0x0b,0xe0,0xee]
	vdup.8	q8, r0
@ CHECK: vdup.16	q8, r0                  @ encoding: [0xb0,0x0b,0xa0,0xee]
	vdup.16	q8, r0
@ CHECK: vdup.32	q8, r0                  @ encoding: [0x90,0x0b,0xa0,0xee]
	vdup.32	q8, r0
@ CHECK: vdup.8	d16, d16[1]             @ encoding: [0x20,0x0c,0xf3,0xff]
	vdup.8	d16, d16[1]
@ CHECK: vdup.16	d16, d16[1]             @ encoding: [0x20,0x0c,0xf6,0xff]
	vdup.16	d16, d16[1]
@ CHECK: vdup.32	d16, d16[1]             @ encoding: [0x20,0x0c,0xfc,0xff]
	vdup.32	d16, d16[1]
@ CHECK: vdup.8	q8, d16[1]              @ encoding: [0x60,0x0c,0xf3,0xff]
	vdup.8	q8, d16[1]
@ CHECK: vdup.16	q8, d16[1]              @ encoding: [0x60,0x0c,0xf6,0xff]
	vdup.16	q8, d16[1]
@ CHECK: vdup.32	q8, d16[1]              @ encoding: [0x60,0x0c,0xfc,0xff]
	vdup.32	q8, d16[1]
