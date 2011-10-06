@RUN: llvm-mc -triple thumbv7-unknown-unknown -show-encoding < %s | FileCheck %s

.code 16

	vdup.8	d16, r1
	vdup.16	d15, r2
	vdup.32	d14, r3
	vdup.8	q9, r4
	vdup.16	q8, r5
	vdup.32	q7, r6

@ CHECK: vdup.8	d16, r1                 @ encoding: [0xc0,0xee,0x90,0x1b]
@ CHECK: vdup.16	d15, r2         @ encoding: [0x8f,0xee,0x30,0x2b]
@ CHECK: vdup.32	d14, r3         @ encoding: [0x8e,0xee,0x10,0x3b]
@ CHECK: vdup.8	q9, r4                  @ encoding: [0xe2,0xee,0x90,0x4b]
@ CHECK: vdup.16	q8, r5          @ encoding: [0xa0,0xee,0xb0,0x5b]
@ CHECK: vdup.32	q7, r6          @ encoding: [0xae,0xee,0x10,0x6b]

@	vdup.8	d16, d16[1]
@	vdup.16	d16, d16[1]
@	vdup.32	d16, d16[1]
@	vdup.8	q8, d16[1]
@	vdup.16	q8, d16[1]
@	vdup.32	q8, d16[1]

@ FIXME: vdup.8	d16, d16[1]             @ encoding: [0x20,0x0c,0xf3,0xff]
@ FIXME: vdup.16 d16, d16[1]            @ encoding: [0x20,0x0c,0xf6,0xff]
@ FIXME: vdup.32 d16, d16[1]            @ encoding: [0x20,0x0c,0xfc,0xff]
@ FIXME: vdup.8	q8, d16[1]              @ encoding: [0x60,0x0c,0xf3,0xff]
@ FIXME: vdup.16 q8, d16[1]             @ encoding: [0x60,0x0c,0xf6,0xff]
@ FIXME: vdup.32 q8, d16[1]             @ encoding: [0x60,0x0c,0xfc,0xff]
