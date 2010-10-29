// RUN: llvm-mc -triple arm-unknown-unkown -show-encoding < %s | FileCheck %s

// CHECK: vand	d16, d17, d16           @ encoding: [0xb0,0x01,0x41,0xf2]
	vand	d16, d17, d16
// CHECK: vand	q8, q8, q9              @ encoding: [0xf2,0x01,0x40,0xf2]
	vand	q8, q8, q9

// CHECK: veor	d16, d17, d16           @ encoding: [0xb0,0x01,0x41,0xf3]
	veor	d16, d17, d16
// CHECK: veor	q8, q8, q9              @ encoding: [0xf2,0x01,0x40,0xf3]
	veor	q8, q8, q9

// CHECK: vorr	d16, d17, d16           @ encoding: [0xb0,0x01,0x61,0xf2]
	vorr	d16, d17, d16
// CHECK: vorr	q8, q8, q9              @ encoding: [0xf2,0x01,0x60,0xf2]
	vorr	q8, q8, q9

// CHECK: vbic	d16, d17, d16           @ encoding: [0xb0,0x01,0x51,0xf2]
	vbic	d16, d17, d16
// CHECK: vbic	q8, q8, q9              @ encoding: [0xf2,0x01,0x50,0xf2]
	vbic	q8, q8, q9

// CHECK: vorn	d16, d17, d16           @ encoding: [0xb0,0x01,0x71,0xf2]
	vorn	d16, d17, d16
// CHECK: vorn	q8, q8, q9              @ encoding: [0xf2,0x01,0x70,0xf2]
	vorn	q8, q8, q9

// CHECK: vmvn	d16, d16                @ encoding: [0xa0,0x05,0xf0,0xf3]
	vmvn	d16, d16
// CHECK: vmvn	q8, q8                  @ encoding: [0xe0,0x05,0xf0,0xf3]
	vmvn	q8, q8

// CHECK: vbsl	d18, d17, d16           @ encoding: [0xb0,0x21,0x51,0xf3]
	vbsl	d18, d17, d16
// CHECK: vbsl	q8, q10, q9             @ encoding: [0xf2,0x01,0x54,0xf3]
	vbsl	q8, q10, q9
