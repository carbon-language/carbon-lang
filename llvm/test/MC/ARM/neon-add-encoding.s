// RUN: llvm-mc -mcpu=cortex-a8 -triple armv7-apple-darwin -show-encoding < %s | FileCheck %s


// CHECK: vadd.i8	d16, d17, d16           @ encoding: [0xa0,0x08,0x41,0xf2]
	vadd.i8	d16, d17, d16
// CHECK: vadd.i16	d16, d17, d16   @ encoding: [0xa0,0x08,0x51,0xf2]
	vadd.i16	d16, d17, d16
// CHECK: vadd.i64	d16, d17, d16   @ encoding: [0xa0,0x08,0x71,0xf2]
	vadd.i64	d16, d17, d16
// CHECK: vadd.i32	d16, d17, d16   @ encoding: [0xa0,0x08,0x61,0xf2]
	vadd.i32	d16, d17, d16
// CHECK: vadd.f32	d16, d16, d17   @ encoding: [0xa1,0x0d,0x40,0xf2]
	vadd.f32	d16, d16, d17
// CHECK: vadd.f32	q8, q8, q9      @ encoding: [0xe2,0x0d,0x40,0xf2]
	vadd.f32	q8, q8, q9

// CHECK: vaddl.s8	q8, d17, d16    @ encoding: [0xa0,0x00,0xc1,0xf2]
	vaddl.s8	q8, d17, d16
// CHECK: vaddl.s16	q8, d17, d16    @ encoding: [0xa0,0x00,0xd1,0xf2]
	vaddl.s16	q8, d17, d16
// CHECK: vaddl.s32	q8, d17, d16    @ encoding: [0xa0,0x00,0xe1,0xf2]
	vaddl.s32	q8, d17, d16
// CHECK: vaddl.u8	q8, d17, d16    @ encoding: [0xa0,0x00,0xc1,0xf3]
	vaddl.u8	q8, d17, d16
// CHECK: vaddl.u16	q8, d17, d16    @ encoding: [0xa0,0x00,0xd1,0xf3]
	vaddl.u16	q8, d17, d16
// CHECK: vaddl.u32	q8, d17, d16    @ encoding: [0xa0,0x00,0xe1,0xf3]
	vaddl.u32	q8, d17, d16

// CHECK: vaddw.s8	q8, q8, d18     @ encoding: [0xa2,0x01,0xc0,0xf2]
	vaddw.s8	q8, q8, d18
// CHECK: vaddw.s16	q8, q8, d18     @ encoding: [0xa2,0x01,0xd0,0xf2]
	vaddw.s16	q8, q8, d18
// CHECK: vaddw.s32	q8, q8, d18     @ encoding: [0xa2,0x01,0xe0,0xf2]
	vaddw.s32	q8, q8, d18
// CHECK: vaddw.u8	q8, q8, d18     @ encoding: [0xa2,0x01,0xc0,0xf3]
	vaddw.u8	q8, q8, d18
// CHECK: vaddw.u16	q8, q8, d18     @ encoding: [0xa2,0x01,0xd0,0xf3]
	vaddw.u16	q8, q8, d18
// CHECK: vaddw.u32	q8, q8, d18     @ encoding: [0xa2,0x01,0xe0,0xf3]
	vaddw.u32	q8, q8, d18

// CHECK: vhadd.s8	d16, d16, d17   @ encoding: [0xa1,0x00,0x40,0xf2]
	vhadd.s8	d16, d16, d17
// CHECK: vhadd.s16	d16, d16, d17   @ encoding: [0xa1,0x00,0x50,0xf2]
	vhadd.s16	d16, d16, d17
// CHECK: vhadd.s32	d16, d16, d17   @ encoding: [0xa1,0x00,0x60,0xf2]
	vhadd.s32	d16, d16, d17
// CHECK: vhadd.u8	d16, d16, d17   @ encoding: [0xa1,0x00,0x40,0xf3]
	vhadd.u8	d16, d16, d17
// CHECK: vhadd.u16	d16, d16, d17   @ encoding: [0xa1,0x00,0x50,0xf3]
	vhadd.u16	d16, d16, d17
// CHECK: vhadd.u32	d16, d16, d17   @ encoding: [0xa1,0x00,0x60,0xf3]
	vhadd.u32	d16, d16, d17
// CHECK: vhadd.s8	q8, q8, q9      @ encoding: [0xe2,0x00,0x40,0xf2]
	vhadd.s8	q8, q8, q9
// CHECK: vhadd.s16	q8, q8, q9      @ encoding: [0xe2,0x00,0x50,0xf2]
	vhadd.s16	q8, q8, q9
// CHECK: vhadd.s32	q8, q8, q9      @ encoding: [0xe2,0x00,0x60,0xf2]
	vhadd.s32	q8, q8, q9
  // CHECK: vhadd.u8	q8, q8, q9      @ encoding: [0xe2,0x00,0x40,0xf3]
	vhadd.u8	q8, q8, q9
// CHECK: vhadd.u16	q8, q8, q9      @ encoding: [0xe2,0x00,0x50,0xf3]
	vhadd.u16	q8, q8, q9
// CHECK: vhadd.u32	q8, q8, q9      @ encoding: [0xe2,0x00,0x60,0xf3]
	vhadd.u32	q8, q8, q9
	
// CHECK: vrhadd.s8	d16, d16, d17   @ encoding: [0xa1,0x01,0x40,0xf2]
	vrhadd.s8	d16, d16, d17
// CHECK: vrhadd.s16	d16, d16, d17   @ encoding: [0xa1,0x01,0x50,0xf2]
	vrhadd.s16	d16, d16, d17
// CHECK: vrhadd.s32	d16, d16, d17   @ encoding: [0xa1,0x01,0x60,0xf2]
	vrhadd.s32	d16, d16, d17
// CHECK: vrhadd.u8	d16, d16, d17   @ encoding: [0xa1,0x01,0x40,0xf3]
	vrhadd.u8	d16, d16, d17
// CHECK: vrhadd.u16	d16, d16, d17   @ encoding: [0xa1,0x01,0x50,0xf3]
	vrhadd.u16	d16, d16, d17
// CHECK: vrhadd.u32	d16, d16, d17   @ encoding: [0xa1,0x01,0x60,0xf3]
	vrhadd.u32	d16, d16, d17
// CHECK: vrhadd.s8	q8, q8, q9      @ encoding: [0xe2,0x01,0x40,0xf2]
	vrhadd.s8	q8, q8, q9
// CHECK: vrhadd.s16	q8, q8, q9      @ encoding: [0xe2,0x01,0x50,0xf2]
	vrhadd.s16	q8, q8, q9
// CHECK: vrhadd.s32	q8, q8, q9      @ encoding: [0xe2,0x01,0x60,0xf2]
	vrhadd.s32	q8, q8, q9
// CHECK: vrhadd.u8	q8, q8, q9      @ encoding: [0xe2,0x01,0x40,0xf3]
	vrhadd.u8	q8, q8, q9
// CHECK: vrhadd.u16	q8, q8, q9      @ encoding: [0xe2,0x01,0x50,0xf3]
	vrhadd.u16	q8, q8, q9
// CHECK: vrhadd.u32	q8, q8, q9      @ encoding: [0xe2,0x01,0x60,0xf3]
	vrhadd.u32	q8, q8, q9

// CHECK: vqadd.s8	d16, d16, d17   @ encoding: [0xb1,0x00,0x40,0xf2]
	vqadd.s8	d16, d16, d17
// CHECK: vqadd.s16	d16, d16, d17   @ encoding: [0xb1,0x00,0x50,0xf2]
	vqadd.s16	d16, d16, d17
// CHECK: vqadd.s32	d16, d16, d17   @ encoding: [0xb1,0x00,0x60,0xf2]
	vqadd.s32	d16, d16, d17
// CHECK: vqadd.s64	d16, d16, d17   @ encoding: [0xb1,0x00,0x70,0xf2]
	vqadd.s64	d16, d16, d17
// CHECK: vqadd.u8	d16, d16, d17   @ encoding: [0xb1,0x00,0x40,0xf3]
	vqadd.u8	d16, d16, d17
// CHECK: vqadd.u16	d16, d16, d17   @ encoding: [0xb1,0x00,0x50,0xf3]
	vqadd.u16	d16, d16, d17
// CHECK: vqadd.u32	d16, d16, d17   @ encoding: [0xb1,0x00,0x60,0xf3]
	vqadd.u32	d16, d16, d17
// CHECK: vqadd.u64	d16, d16, d17   @ encoding: [0xb1,0x00,0x70,0xf3]
	vqadd.u64	d16, d16, d17
// CHECK: vqadd.s8	q8, q8, q9      @ encoding: [0xf2,0x00,0x40,0xf2]
	vqadd.s8	q8, q8, q9
// CHECK: vqadd.s16	q8, q8, q9      @ encoding: [0xf2,0x00,0x50,0xf2]
	vqadd.s16	q8, q8, q9
// CHECK: vqadd.s32	q8, q8, q9      @ encoding: [0xf2,0x00,0x60,0xf2]
	vqadd.s32	q8, q8, q9
// CHECK: vqadd.s64	q8, q8, q9      @ encoding: [0xf2,0x00,0x70,0xf2]
	vqadd.s64	q8, q8, q9
// CHECK: vqadd.u8	q8, q8, q9      @ encoding: [0xf2,0x00,0x40,0xf3]
	vqadd.u8	q8, q8, q9
// CHECK: vqadd.u16	q8, q8, q9      @ encoding: [0xf2,0x00,0x50,0xf3]
	vqadd.u16	q8, q8, q9
// CHECK: vqadd.u32	q8, q8, q9      @ encoding: [0xf2,0x00,0x60,0xf3]
	vqadd.u32	q8, q8, q9
// CHECK: vqadd.u64	q8, q8, q9      @ encoding: [0xf2,0x00,0x70,0xf3]
	vqadd.u64	q8, q8, q9

// CHECK: vaddhn.i16	d16, q8, q9     @ encoding: [0xa2,0x04,0xc0,0xf2]
	vaddhn.i16	d16, q8, q9
// CHECK: vaddhn.i32	d16, q8, q9     @ encoding: [0xa2,0x04,0xd0,0xf2]
	vaddhn.i32	d16, q8, q9
// CHECK: vaddhn.i64	d16, q8, q9     @ encoding: [0xa2,0x04,0xe0,0xf2]
	vaddhn.i64	d16, q8, q9
// CHECK: vraddhn.i16	d16, q8, q9     @ encoding: [0xa2,0x04,0xc0,0xf3]
	vraddhn.i16	d16, q8, q9
// CHECK: vraddhn.i32	d16, q8, q9     @ encoding: [0xa2,0x04,0xd0,0xf3]
	vraddhn.i32	d16, q8, q9
// CHECK: vraddhn.i64	d16, q8, q9     @ encoding: [0xa2,0x04,0xe0,0xf3]
	vraddhn.i64	d16, q8, q9
