// RUN: llvm-mc -mcpu=cortex-a8 -triple arm-unknown-unkown -show-encoding < %s | FileCheck %s

// CHECK: vcvt.s32.f32	d16, d16        @ encoding: [0x20,0x07,0xfb,0xf3]
	vcvt.s32.f32	d16, d16
// CHECK: vcvt.u32.f32	d16, d16        @ encoding: [0xa0,0x07,0xfb,0xf3]
	vcvt.u32.f32	d16, d16
// CHECK: vcvt.f32.s32	d16, d16        @ encoding: [0x20,0x06,0xfb,0xf3]
	vcvt.f32.s32	d16, d16
// CHECK: vcvt.f32.u32	d16, d16        @ encoding: [0xa0,0x06,0xfb,0xf3]
	vcvt.f32.u32	d16, d16
// CHECK: vcvt.s32.f32	q8, q8          @ encoding: [0x60,0x07,0xfb,0xf3]
	vcvt.s32.f32	q8, q8
// CHECK: vcvt.u32.f32	q8, q8          @ encoding: [0xe0,0x07,0xfb,0xf3]
	vcvt.u32.f32	q8, q8
// CHECK: vcvt.f32.s32	q8, q8          @ encoding: [0x60,0x06,0xfb,0xf3]
	vcvt.f32.s32	q8, q8
// CHECK: vcvt.f32.u32	q8, q8          @ encoding: [0xe0,0x06,0xfb,0xf3]
	vcvt.f32.u32	q8, q8
// CHECK: vcvt.s32.f32	d16, d16, #1    @ encoding: [0x30,0x0f,0xff,0xf2]
	vcvt.s32.f32	d16, d16, #1
// CHECK: vcvt.u32.f32	d16, d16, #1    @ encoding: [0x30,0x0f,0xff,0xf3]
	vcvt.u32.f32	d16, d16, #1
// CHECK: vcvt.f32.s32	d16, d16, #1    @ encoding: [0x30,0x0e,0xff,0xf2]
	vcvt.f32.s32	d16, d16, #1
// CHECK: vcvt.f32.u32	d16, d16, #1    @ encoding: [0x30,0x0e,0xff,0xf3]
	vcvt.f32.u32	d16, d16, #1
// CHECK: vcvt.s32.f32	q8, q8, #1      @ encoding: [0x70,0x0f,0xff,0xf2]
	vcvt.s32.f32	q8, q8, #1
// CHECK: vcvt.u32.f32	q8, q8, #1      @ encoding: [0x70,0x0f,0xff,0xf3]
	vcvt.u32.f32	q8, q8, #1
// CHECK: vcvt.f32.s32	q8, q8, #1      @ encoding: [0x70,0x0e,0xff,0xf2]
	vcvt.f32.s32	q8, q8, #1
// CHECK: vcvt.f32.u32	q8, q8, #1      @ encoding: [0x70,0x0e,0xff,0xf3]
	vcvt.f32.u32	q8, q8, #1
