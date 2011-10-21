@ RUN: llvm-mc -mcpu=cortex-a8 -triple armv7-apple-darwin -show-encoding < %s | FileCheck %s

	vld1.8	{d16}, [r0, :64]
	vld1.16	{d16}, [r0]
	vld1.32	{d16}, [r0]
	vld1.64	{d16}, [r0]
	vld1.8	{d16, d17}, [r0, :64]
	vld1.16	{d16, d17}, [r0, :128]
	vld1.32	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	vld1.8 {d1, d2, d3}, [r3]
	vld1.16 {d4, d5, d6}, [r3, :64]
	vld1.32 {d5, d6, d7}, [r3]
	vld1.64 {d6, d7, d8}, [r3, :64]
	vld1.8 {d1, d2, d3, d4}, [r3]
	vld1.16 {d4, d5, d6, d7}, [r3, :64]
	vld1.32 {d5, d6, d7, d8}, [r3]
	vld1.64 {d6, d7, d8, d9}, [r3, :64]

@ CHECK: vld1.8	{d16}, [r0, :64]        @ encoding: [0x1f,0x07,0x60,0xf4]
@ CHECK: vld1.16	{d16}, [r0]     @ encoding: [0x4f,0x07,0x60,0xf4]
@ CHECK: vld1.32	{d16}, [r0]     @ encoding: [0x8f,0x07,0x60,0xf4]
@ CHECK: vld1.64	{d16}, [r0]     @ encoding: [0xcf,0x07,0x60,0xf4]
@ CHECK: vld1.8	{d16, d17}, [r0, :64]   @ encoding: [0x1f,0x0a,0x60,0xf4]
@ CHECK: vld1.16	{d16, d17}, [r0, :128] @ encoding: [0x6f,0x0a,0x60,0xf4]
@ CHECK: vld1.32	{d16, d17}, [r0] @ encoding: [0x8f,0x0a,0x60,0xf4]
@ CHECK: vld1.64	{d16, d17}, [r0] @ encoding: [0xcf,0x0a,0x60,0xf4]
@ CHECK: vld1.8	{d1, d2, d3}, [r3]      @ encoding: [0x0f,0x16,0x23,0xf4]
@ CHECK: vld1.16 {d4, d5, d6}, [r3, :64] @ encoding: [0x5f,0x46,0x23,0xf4]
@ CHECK: vld1.32 {d5, d6, d7}, [r3]      @ encoding: [0x8f,0x56,0x23,0xf4]
@ CHECK: vld1.64 {d6, d7, d8}, [r3, :64] @ encoding: [0xdf,0x66,0x23,0xf4]
@ CHECK: vld1.8	{d1, d2, d3, d4}, [r3]  @ encoding: [0x0f,0x12,0x23,0xf4]
@ CHECK: vld1.16 {d4, d5, d6, d7}, [r3, :64] @ encoding: [0x5f,0x42,0x23,0xf4]
@ CHECK: vld1.32 {d5, d6, d7, d8}, [r3]  @ encoding: [0x8f,0x52,0x23,0xf4]
@ CHECK: vld1.64 {d6, d7, d8, d9}, [r3, :64] @ encoding: [0xdf,0x62,0x23,0xf4]


	vld2.8	{d16, d17}, [r0, :64]
	vld2.16	{d16, d17}, [r0, :128]
	vld2.32	{d16, d17}, [r0]
@	vld2.8	{d16, d17, d18, d19}, [r0, :64]
@	vld2.16	{d16, d17, d18, d19}, [r0, :128]
@	vld2.32	{d16, d17, d18, d19}, [r0, :256]

@ CHECK: vld2.8	{d16, d17}, [r0, :64]   @ encoding: [0x1f,0x08,0x60,0xf4]
@ CHECK: vld2.16 {d16, d17}, [r0, :128] @ encoding: [0x6f,0x08,0x60,0xf4]
@ CHECK: vld2.32 {d16, d17}, [r0] @ encoding: [0x8f,0x08,0x60,0xf4]
@ FIXME: vld2.8	{d16, d17, d18, d19}, [r0, :64] @ encoding: [0x1f,0x03,0x60,0xf4]
@ FIXME: vld2.16 {d16, d17, d18, d19}, [r0, :128] @ encoding: [0x6f,0x03,0x60,0xf4]
@ FIXME: vld2.32 {d16, d17, d18, d19}, [r0, :256] @ encoding: [0xbf,0x03,0x60,0xf4]


@	vld3.8	{d16, d17, d18}, [r0, :64]
@	vld3.16	{d16, d17, d18}, [r0]
@	vld3.32	{d16, d17, d18}, [r0]
@	vld3.8	{d16, d18, d20}, [r0, :64]!
@	vld3.8	{d17, d19, d21}, [r0, :64]!
@	vld3.16	{d16, d18, d20}, [r0]!
@	vld3.16	{d17, d19, d21}, [r0]!
@	vld3.32	{d16, d18, d20}, [r0]!
@	vld3.32	{d17, d19, d21}, [r0]!

@ FIXME: vld3.8	{d16, d17, d18}, [r0, :64] @ encoding: [0x1f,0x04,0x60,0xf4]
@ FIXME: vld3.16 {d16, d17, d18}, [r0]  @ encoding: [0x4f,0x04,0x60,0xf4]
@ FIXME: vld3.32 {d16, d17, d18}, [r0]  @ encoding: [0x8f,0x04,0x60,0xf4]
@ FIXME: vld3.8	{d16, d18, d20}, [r0, :64]! @ encoding: [0x1d,0x05,0x60,0xf4]
@ FIXME: vld3.8	{d17, d19, d21}, [r0, :64]! @ encoding: [0x1d,0x15,0x60,0xf4]
@ FIXME: vld3.16 {d16, d18, d20}, [r0]! @ encoding: [0x4d,0x05,0x60,0xf4]
@ FIXME: vld3.16 {d17, d19, d21}, [r0]! @ encoding: [0x4d,0x15,0x60,0xf4]
@ FIXME: vld3.32 {d16, d18, d20}, [r0]! @ encoding: [0x8d,0x05,0x60,0xf4]
@ FIXME: vld3.32 {d17, d19, d21}, [r0]! @ encoding: [0x8d,0x15,0x60,0xf4]


@	vld4.8	{d16, d17, d18, d19}, [r0, :64]
@	vld4.16	{d16, d17, d18, d19}, [r0, :128]
@	vld4.32	{d16, d17, d18, d19}, [r0, :256]
@	vld4.8	{d16, d18, d20, d22}, [r0, :256]!
@	vld4.8	{d17, d19, d21, d23}, [r0, :256]!
@	vld4.16	{d16, d18, d20, d22}, [r0]!
@	vld4.16	{d17, d19, d21, d23}, [r0]!
@	vld4.32	{d16, d18, d20, d22}, [r0]!
@	vld4.32	{d17, d19, d21, d23}, [r0]!

@ FIXME: vld4.8	{d16, d17, d18, d19}, [r0, :64]@ encoding: [0x1f,0x00,0x60,0xf4]
@ FIXME: vld4.16 {d16, d17, d18, d19}, [r0,:128]@ encoding:[0x6f,0x00,0x60,0xf4]
@ FIXME: vld4.32 {d16, d17, d18, d19}, [r0,:256]@ encoding:[0xbf,0x00,0x60,0xf4]
@ FIXME: vld4.8	{d16, d18, d20, d22}, [r0,:256]!@ encoding:[0x3d,0x01,0x60,0xf4]
@ FIXME: vld4.8	{d17, d19, d21, d23}, [r0,:256]!@ encoding:[0x3d,0x11,0x60,0xf4]
@ FIXME: vld4.16 {d16, d18, d20, d22}, [r0]! @ encoding: [0x4d,0x01,0x60,0xf4]
@ FIXME: vld4.16 {d17, d19, d21, d23}, [r0]! @ encoding: [0x4d,0x11,0x60,0xf4]
@ FIXME: vld4.32 {d16, d18, d20, d22}, [r0]! @ encoding: [0x8d,0x01,0x60,0xf4]
@ FIXME: vld4.32 {d17, d19, d21, d23}, [r0]! @ encoding: [0x8d,0x11,0x60,0xf4]


@	vld1.8	{d16[3]}, [r0]
@	vld1.16	{d16[2]}, [r0, :16]
@	vld1.32	{d16[1]}, [r0, :32]

@ FIXME: vld1.8	{d16[3]}, [r0]          @ encoding: [0x6f,0x00,0xe0,0xf4]
@ FIXME: vld1.16 {d16[2]}, [r0, :16]    @ encoding: [0x9f,0x04,0xe0,0xf4]
@ FIXME: vld1.32 {d16[1]}, [r0, :32]    @ encoding: [0xbf,0x08,0xe0,0xf4]


@	vld2.8	{d16[1], d17[1]}, [r0, :16]
@	vld2.16	{d16[1], d17[1]}, [r0, :32]
@	vld2.32	{d16[1], d17[1]}, [r0]
@	vld2.16	{d17[1], d19[1]}, [r0]
@	vld2.32	{d17[0], d19[0]}, [r0, :64]

@ FIXME: vld2.8	{d16[1], d17[1]}, [r0, :16] @ encoding: [0x3f,0x01,0xe0,0xf4]
@ FIXME: vld2.16 {d16[1], d17[1]}, [r0, :32] @ encoding: [0x5f,0x05,0xe0,0xf4]
@ FIXME: vld2.32 {d16[1], d17[1]}, [r0]  @ encoding: [0x8f,0x09,0xe0,0xf4]
@ FIXME: vld2.16 {d17[1], d19[1]}, [r0]  @ encoding: [0x6f,0x15,0xe0,0xf4]
@ FIXME: vld2.32 {d17[0], d19[0]}, [r0, :64] @ encoding: [0x5f,0x19,0xe0,0xf4]


@	vld3.8	{d16[1], d17[1], d18[1]}, [r0]
@	vld3.16	{d16[1], d17[1], d18[1]}, [r0]
@	vld3.32	{d16[1], d17[1], d18[1]}, [r0]
@	vld3.16	{d16[1], d18[1], d20[1]}, [r0]
@	vld3.32	{d17[1], d19[1], d21[1]}, [r0]

@ FIXME: vld3.8	{d16[1], d17[1], d18[1]}, [r0] @ encoding: [0x2f,0x02,0xe0,0xf4]
@ FIXME: vld3.16 {d16[1], d17[1], d18[1]}, [r0]@ encoding: [0x4f,0x06,0xe0,0xf4]
@ FIXME: vld3.32 {d16[1], d17[1], d18[1]}, [r0]@ encoding: [0x8f,0x0a,0xe0,0xf4]
@ FIXME: vld3.16 {d16[1], d18[1], d20[1]}, [r0]@ encoding: [0x6f,0x06,0xe0,0xf4]
@ FIXME: vld3.32 {d17[1], d19[1], d21[1]}, [r0]@ encoding: [0xcf,0x1a,0xe0,0xf4]


@	vld4.8	{d16[1], d17[1], d18[1], d19[1]}, [r0, :32]
@	vld4.16	{d16[1], d17[1], d18[1], d19[1]}, [r0]
@	vld4.32	{d16[1], d17[1], d18[1], d19[1]}, [r0, :128]
@	vld4.16	{d16[1], d18[1], d20[1], d22[1]}, [r0, :64]
@	vld4.32	{d17[0], d19[0], d21[0], d23[0]}, [r0]

@ FIXME: vld4.8	{d16[1], d17[1], d18[1], d19[1]}, [r0, :32] @ encoding: [0x3f,0x03,0xe0,0xf4]
@ FIXME: vld4.16 {d16[1], d17[1], d18[1], d19[1]}, [r0] @ encoding: [0x4f,0x07,0xe0,0xf4]
@ FIXME: vld4.32 {d16[1], d17[1], d18[1], d19[1]}, [r0, :128] @ encoding: [0xaf,0x0b,0xe0,0xf4]
@ FIXME: vld4.16 {d16[1], d18[1], d20[1], d22[1]}, [r0, :64] @ encoding: [0x7f,0x07,0xe0,0xf4]
@ FIXME: vld4.32 {d17[0], d19[0], d21[0], d23[0]}, [r0] @ encoding: [0x4f,0x1b,0xe0,0xf4]
