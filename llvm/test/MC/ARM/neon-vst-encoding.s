@ RUN: llvm-mc -mcpu=cortex-a8 -triple armv7-apple-darwin -show-encoding < %s | FileCheck %s

	vst1.8	{d16}, [r0, :64]
	vst1.16	{d16}, [r0]
	vst1.32	{d16}, [r0]
	vst1.64	{d16}, [r0]
	vst1.8	{d16, d17}, [r0, :64]
	vst1.16	{d16, d17}, [r0, :128]
	vst1.32	{d16, d17}, [r0]
	vst1.64	{d16, d17}, [r0]
        vst1.8  {d16, d17, d18}, [r0, :64]
        vst1.8  {d16, d17, d18}, [r0, :64]!
        vst1.8  {d16, d17, d18}, [r0], r3
        vst1.8  {d16, d17, d18, d19}, [r0, :64]
        vst1.16  {d16, d17, d18, d19}, [r1, :64]!
        vst1.64  {d16, d17, d18, d19}, [r3], r2

@ CHECK: vst1.8	{d16}, [r0, :64]        @ encoding: [0x1f,0x07,0x40,0xf4]
@ CHECK: vst1.16 {d16}, [r0]            @ encoding: [0x4f,0x07,0x40,0xf4]
@ CHECK: vst1.32 {d16}, [r0]            @ encoding: [0x8f,0x07,0x40,0xf4]
@ CHECK: vst1.64 {d16}, [r0]            @ encoding: [0xcf,0x07,0x40,0xf4]
@ CHECK: vst1.8	{d16, d17}, [r0, :64]   @ encoding: [0x1f,0x0a,0x40,0xf4]
@ CHECK: vst1.16 {d16, d17}, [r0, :128] @ encoding: [0x6f,0x0a,0x40,0xf4]
@ CHECK: vst1.32 {d16, d17}, [r0]       @ encoding: [0x8f,0x0a,0x40,0xf4]
@ CHECK: vst1.64 {d16, d17}, [r0]       @ encoding: [0xcf,0x0a,0x40,0xf4]
@ CHECK: vst1.8	{d16, d17, d18}, [r0, :64] @ encoding: [0x1f,0x06,0x40,0xf4]
@ CHECK: vst1.8	{d16, d17, d18}, [r0, :64]! @ encoding: [0x1d,0x06,0x40,0xf4]
@ CHECK: vst1.8	{d16, d17, d18}, [r0], r3 @ encoding: [0x03,0x06,0x40,0xf4]
@ CHECK: vst1.8	{d16, d17, d18, d19}, [r0, :64] @ encoding: [0x1f,0x02,0x40,0xf4]
@ CHECK: vst1.16 {d16, d17, d18, d19}, [r1, :64]! @ encoding: [0x5d,0x02,0x41,0xf4]
@ CHECK: vst1.64 {d16, d17, d18, d19}, [r3], r2 @ encoding: [0xc2,0x02,0x43,0xf4]


	vst2.8	{d16, d17}, [r0, :64]
	vst2.16	{d16, d17}, [r0, :128]
	vst2.32	{d16, d17}, [r0]
	vst2.8	{d16, d17, d18, d19}, [r0, :64]
	vst2.16	{d16, d17, d18, d19}, [r0, :128]
	vst2.32	{d16, d17, d18, d19}, [r0, :256]
	vst2.8	{d16, d17}, [r0, :64]!
	vst2.16	{q15}, [r0, :128]!
	vst2.32	{d14, d15}, [r0]!
	vst2.8	{d16, d17, d18, d19}, [r0, :64]!
	vst2.16	{d18-d21}, [r0, :128]!
	vst2.32	{q4, q5}, [r0, :256]!

@ CHECK: vst2.8	{d16, d17}, [r0, :64]   @ encoding: [0x1f,0x08,0x40,0xf4]
@ CHECK: vst2.16 {d16, d17}, [r0, :128] @ encoding: [0x6f,0x08,0x40,0xf4]
@ CHECK: vst2.32 {d16, d17}, [r0]       @ encoding: [0x8f,0x08,0x40,0xf4]
@ CHECK: vst2.8	{d16, d17, d18, d19}, [r0, :64] @ encoding: [0x1f,0x03,0x40,0xf4]
@ CHECK: vst2.16 {d16, d17, d18, d19}, [r0, :128] @ encoding: [0x6f,0x03,0x40,0xf4]
@ CHECK: vst2.32 {d16, d17, d18, d19}, [r0, :256] @ encoding: [0xbf,0x03,0x40,0xf4]
@ CHECK: vst2.8	{d16, d17}, [r0, :64]!  @ encoding: [0x1d,0x08,0x40,0xf4]
@ CHECK: vst2.16	{d30, d31}, [r0, :128]! @ encoding: [0x6d,0xe8,0x40,0xf4]
@ CHECK: vst2.32	{d14, d15}, [r0]!       @ encoding: [0x8d,0xe8,0x00,0xf4]
@ CHECK: vst2.8	{d16, d17, d18, d19}, [r0, :64]! @ encoding: [0x1d,0x03,0x40,0xf4]
@ CHECK: vst2.16	{d18, d19, d20, d21}, [r0, :128]! @ encoding: [0x6d,0x23,0x40,0xf4]
@ CHECK: vst2.32	{d8, d9, d10, d11}, [r0, :256]! @ encoding: [0xbd,0x83,0x00,0xf4]


@	vst3.8	{d16, d17, d18}, [r0, :64]
@	vst3.16	{d16, d17, d18}, [r0]
@	vst3.32	{d16, d17, d18}, [r0]
@	vst3.8	{d16, d18, d20}, [r0, :64]!
@	vst3.8	{d17, d19, d21}, [r0, :64]!
@	vst3.16	{d16, d18, d20}, [r0]!
@	vst3.16	{d17, d19, d21}, [r0]!
@	vst3.32	{d16, d18, d20}, [r0]!
@	vst3.32	{d17, d19, d21}, [r0]!

@ FIXME: vst3.8	{d16, d17, d18}, [r0, :64] @ encoding: [0x1f,0x04,0x40,0xf4]
@ FIXME: vst3.16 {d16, d17, d18}, [r0]  @ encoding: [0x4f,0x04,0x40,0xf4]
@ FIXME: vst3.32 {d16, d17, d18}, [r0]  @ encoding: [0x8f,0x04,0x40,0xf4]
@ FIXME: vst3.8	{d16, d18, d20}, [r0, :64]! @ encoding: [0x1d,0x05,0x40,0xf4]
@ FIXME: vst3.8	{d17, d19, d21}, [r0, :64]! @ encoding: [0x1d,0x15,0x40,0xf4]
@ FIXME: vst3.16 {d16, d18, d20}, [r0]! @ encoding: [0x4d,0x05,0x40,0xf4]
@ FIXME: vst3.16 {d17, d19, d21}, [r0]! @ encoding: [0x4d,0x15,0x40,0xf4]
@ FIXME: vst3.32 {d16, d18, d20}, [r0]! @ encoding: [0x8d,0x05,0x40,0xf4]
@ FIXME: vst3.32 {d17, d19, d21}, [r0]! @ encoding: [0x8d,0x15,0x40,0xf4]


@	vst4.8	{d16, d17, d18, d19}, [r0, :64]
@	vst4.16	{d16, d17, d18, d19}, [r0, :128]
@	vst4.8	{d16, d18, d20, d22}, [r0, :256]!
@	vst4.8	{d17, d19, d21, d23}, [r0, :256]!
@	vst4.16	{d16, d18, d20, d22}, [r0]!
@	vst4.16	{d17, d19, d21, d23}, [r0]!
@	vst4.32	{d16, d18, d20, d22}, [r0]!
@	vst4.32	{d17, d19, d21, d23}, [r0]!

@ FIXME: vst4.8	{d16, d17, d18, d19}, [r0, :64] @ encoding: [0x1f,0x00,0x40,0xf4]
@ FIXME: vst4.16 {d16, d17, d18, d19}, [r0, :128] @ encoding: [0x6f,0x00,0x40,0xf4]
@ FIXME: vst4.8	{d16, d18, d20, d22}, [r0, :256]! @ encoding: [0x3d,0x01,0x40,0xf4]
@ FIXME: vst4.8	{d17, d19, d21, d23}, [r0, :256]! @ encoding: [0x3d,0x11,0x40,0xf4]
@ FIXME: vst4.16 {d16, d18, d20, d22}, [r0]! @ encoding: [0x4d,0x01,0x40,0xf4]
@ FIXME: vst4.16 {d17, d19, d21, d23}, [r0]! @ encoding: [0x4d,0x11,0x40,0xf4]
@ FIXME: vst4.32 {d16, d18, d20, d22}, [r0]! @ encoding: [0x8d,0x01,0x40,0xf4]
@ FIXME: vst4.32 {d17, d19, d21, d23}, [r0]! @ encoding: [0x8d,0x11,0x40,0xf4]


	vst2.8	{d16[1], d17[1]}, [r0, :16]
	vst2.p16	{d16[1], d17[1]}, [r0, :32]
	vst2.i32	{d16[1], d17[1]}, [r0]
	vst2.u16	{d17[1], d19[1]}, [r0]
	vst2.f32	{d17[0], d19[0]}, [r0, :64]

        vst2.8 {d2[4], d3[4]}, [r2], r3
        vst2.u8 {d2[4], d3[4]}, [r2]!
        vst2.p8 {d2[4], d3[4]}, [r2]

        vst2.16 {d17[1], d19[1]}, [r0]
        vst2.32 {d17[0], d19[0]}, [r0, :64]
        vst2.i16 {d7[1], d9[1]}, [r1]!
        vst2.32 {d6[0], d8[0]}, [r2, :64]!
        vst2.16 {d2[1], d4[1]}, [r3], r5
        vst2.u32 {d5[0], d7[0]}, [r4, :64], r7

@ CHECK: vst2.8	{d16[1], d17[1]}, [r0, :16] @ encoding: [0x3f,0x01,0xc0,0xf4]
@ CHECK: vst2.16 {d16[1], d17[1]}, [r0, :32] @ encoding: [0x5f,0x05,0xc0,0xf4]
@ CHECK: vst2.32 {d16[1], d17[1]}, [r0]  @ encoding: [0x8f,0x09,0xc0,0xf4]
@ CHECK: vst2.16 {d17[1], d19[1]}, [r0]  @ encoding: [0x6f,0x15,0xc0,0xf4]
@ CHECK: vst2.32 {d17[0], d19[0]}, [r0, :64] @ encoding: [0x5f,0x19,0xc0,0xf4]

@ CHECK: vst2.8	{d2[4], d3[4]}, [r2], r3 @ encoding: [0x83,0x21,0x82,0xf4]
@ CHECK: vst2.8	{d2[4], d3[4]}, [r2]!   @ encoding: [0x8d,0x21,0x82,0xf4]
@ CHECK: vst2.8	{d2[4], d3[4]}, [r2]    @ encoding: [0x8f,0x21,0x82,0xf4]

@ CHECK: vst2.16 {d17[1], d19[1]}, [r0]  @ encoding: [0x6f,0x15,0xc0,0xf4]
@ CHECK: vst2.32 {d17[0], d19[0]}, [r0, :64] @ encoding: [0x5f,0x19,0xc0,0xf4]
@ CHECK: vst2.16 {d7[1], d9[1]}, [r1]!   @ encoding: [0x6d,0x75,0x81,0xf4]
@ CHECK: vst2.32 {d6[0], d8[0]}, [r2, :64]! @ encoding: [0x5d,0x69,0x82,0xf4]
@ CHECK: vst2.16 {d2[1], d4[1]}, [r3], r5 @ encoding: [0x65,0x25,0x83,0xf4]
@ CHECK: vst2.32 {d5[0], d7[0]}, [r4, :64], r7 @ encoding: [0x57,0x59,0x84,0xf4]


	vld3.8 {d16[1], d17[1], d18[1]}, [r1]
	vld3.16 {d6[1], d7[1], d8[1]}, [r2]
	vld3.32 {d1[1], d2[1], d3[1]}, [r3]
	vld3.u16 {d27[2], d29[2], d31[2]}, [r4]
	vld3.i32 {d6[0], d8[0], d10[0]}, [r5]

	vld3.i8 {d12[3], d13[3], d14[3]}, [r6], r1
	vld3.i16 {d11[2], d12[2], d13[2]}, [r7], r2
	vld3.u32 {d2[1], d3[1], d4[1]}, [r8], r3
	vld3.u16 {d14[2], d16[2], d18[2]}, [r9], r4
	vld3.i32 {d16[0], d18[0], d20[0]}, [r10], r5

	vld3.p8 {d6[6], d7[6], d8[6]}, [r8]!
	vld3.16 {d9[2], d10[2], d11[2]}, [r7]!
	vld3.f32 {d1[1], d2[1], d3[1]}, [r6]!
	vld3.p16 {d20[2], d22[2], d24[2]}, [r5]!
	vld3.32 {d5[0], d7[0], d9[0]}, [r4]!

@ CHECK: vld3.8	{d16[1], d17[1], d17[1]}, [r1] @ encoding: [0x2f,0x02,0xe1,0xf4]
@ CHECK: vld3.16 {d6[1], d7[1], d7[1]}, [r2] @ encoding: [0x4f,0x66,0xa2,0xf4]
@ CHECK: vld3.32 {d1[1], d2[1], d2[1]}, [r3] @ encoding: [0x8f,0x1a,0xa3,0xf4]
@ CHECK: vld3.16 {d27[2], d29[2], d29[2]}, [r4] @ encoding: [0xaf,0xb6,0xe4,0xf4]
@ CHECK: vld3.32 {d6[0], d8[0], d8[0]}, [r5] @ encoding: [0x4f,0x6a,0xa5,0xf4]
@ CHECK: vld3.8	{d12[3], d13[3], d13[3]}, [r6], r1 @ encoding: [0x61,0xc2,0xa6,0xf4]
@ CHECK: vld3.16 {d11[2], d12[2], d12[2]}, [r7], r2 @ encoding: [0x82,0xb6,0xa7,0xf4]
@ CHECK: vld3.32 {d2[1], d3[1], d3[1]}, [r8], r3 @ encoding: [0x83,0x2a,0xa8,0xf4]
@ CHECK: vld3.16 {d14[2], d16[2], d16[2]}, [r9], r4 @ encoding: [0xa4,0xe6,0xa9,0xf4]
@ CHECK: vld3.32 {d16[0], d18[0], d18[0]}, [r10], r5 @ encoding: [0x45,0x0a,0xea,0xf4]
@ CHECK: vld3.8	{d6[6], d7[6], d7[6]}, [r8]! @ encoding: [0xcd,0x62,0xa8,0xf4]
@ CHECK: vld3.16 {d9[2], d10[2], d10[2]}, [r7]! @ encoding: [0x8d,0x96,0xa7,0xf4]
@ CHECK: vld3.32 {d1[1], d2[1], d2[1]}, [r6]! @ encoding: [0x8d,0x1a,0xa6,0xf4]
@ CHECK: vld3.16 {d20[2], d21[2], d21[2]}, [r5]! @ encoding: [0xad,0x46,0xe5,0xf4]
@ CHECK: vld3.32 {d5[0], d7[0], d7[0]}, [r4]! @ encoding: [0x4d,0x5a,0xa4,0xf4]


@	vst4.8	{d16[1], d17[1], d18[1], d19[1]}, [r0, :32]
@	vst4.16	{d16[1], d17[1], d18[1], d19[1]}, [r0]
@	vst4.32	{d16[1], d17[1], d18[1], d19[1]}, [r0, :128]
@	vst4.16	{d17[3], d19[3], d21[3], d23[3]}, [r0, :64]
@	vst4.32	{d17[0], d19[0], d21[0], d23[0]}, [r0]

@ FIXME: vst4.8	{d16[1], d17[1], d18[1], d19[1]}, [r0, :32] @ encoding: [0x3f,0x03,0xc0,0xf4]
@ FIXME: vst4.16 {d16[1], d17[1], d18[1], d19[1]}, [r0] @ encoding: [0x4f,0x07,0xc0,0xf4]
@ FIXME: vst4.32 {d16[1], d17[1], d18[1], d19[1]}, [r0, :128] @ encoding: [0xaf,0x0b,0xc0,0xf4]
@ FIXME: vst4.16 {d17[3], d19[3], d21[3], d23[3]}, [r0, :64] @ encoding: [0xff,0x17,0xc0,0xf4]
@ FIXME: vst4.32 {d17[0], d19[0], d21[0], d23[0]}, [r0] @ encoding: [0x4f,0x1b,0xc0,0xf4]


@ Spot-check additional size-suffix aliases.

        vst1.8 {d2}, [r2]
        vst1.p8 {d2}, [r2]
        vst1.u8 {d2}, [r2]

        vst1.8 {q2}, [r2]
        vst1.p8 {q2}, [r2]
        vst1.u8 {q2}, [r2]
        vst1.f32 {q2}, [r2]

@ CHECK: vst1.8	{d2}, [r2]              @ encoding: [0x0f,0x27,0x02,0xf4]
@ CHECK: vst1.8	{d2}, [r2]              @ encoding: [0x0f,0x27,0x02,0xf4]
@ CHECK: vst1.8	{d2}, [r2]              @ encoding: [0x0f,0x27,0x02,0xf4]

@ CHECK: vst1.8	{d4, d5}, [r2]          @ encoding: [0x0f,0x4a,0x02,0xf4]
@ CHECK: vst1.8	{d4, d5}, [r2]          @ encoding: [0x0f,0x4a,0x02,0xf4]
@ CHECK: vst1.8	{d4, d5}, [r2]          @ encoding: [0x0f,0x4a,0x02,0xf4]
@ CHECK: vst1.32 {d4, d5}, [r2]         @ encoding: [0x8f,0x4a,0x02,0xf4]
