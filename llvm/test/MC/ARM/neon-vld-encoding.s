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

	vld1.8	{d16}, [r0, :64]!
	vld1.16	{d16}, [r0]!
	vld1.32	{d16}, [r0]!
	vld1.64	{d16}, [r0]!
	vld1.8	{d16, d17}, [r0, :64]!
	vld1.16	{d16, d17}, [r0, :128]!
	vld1.32	{d16, d17}, [r0]!
	vld1.64	{d16, d17}, [r0]!

	vld1.8	{d16}, [r0, :64], r5
	vld1.16	{d16}, [r0], r5
	vld1.32	{d16}, [r0], r5
	vld1.64	{d16}, [r0], r5
	vld1.8	{d16, d17}, [r0, :64], r5
	vld1.16	{d16, d17}, [r0, :128], r5
	vld1.32	{d16, d17}, [r0], r5
	vld1.64	{d16, d17}, [r0], r5

	vld1.8 {d1, d2, d3}, [r3]!
	vld1.16 {d4, d5, d6}, [r3, :64]!
	vld1.32 {d5, d6, d7}, [r3]!
	vld1.64 {d6, d7, d8}, [r3, :64]!

	vld1.8 {d1, d2, d3}, [r3], r6
	vld1.16 {d4, d5, d6}, [r3, :64], r6
	vld1.32 {d5, d6, d7}, [r3], r6
	vld1.64 {d6, d7, d8}, [r3, :64], r6

	vld1.8 {d1, d2, d3, d4}, [r3]!
	vld1.16 {d4, d5, d6, d7}, [r3, :64]!
	vld1.32 {d5, d6, d7, d8}, [r3]!
	vld1.64 {d6, d7, d8, d9}, [r3, :64]!

	vld1.8 {d1, d2, d3, d4}, [r3], r8
	vld1.16 {d4, d5, d6, d7}, [r3, :64], r8
	vld1.32 {d5, d6, d7, d8}, [r3], r8
	vld1.64 {d6, d7, d8, d9}, [r3, :64], r8

@ CHECK: vld1.8 {d16}, [r0, :64]        @ encoding: [0x1f,0x07,0x60,0xf4]
@ CHECK: vld1.16 {d16}, [r0]            @ encoding: [0x4f,0x07,0x60,0xf4]
@ CHECK: vld1.32 {d16}, [r0]            @ encoding: [0x8f,0x07,0x60,0xf4]
@ CHECK: vld1.64 {d16}, [r0]            @ encoding: [0xcf,0x07,0x60,0xf4]
@ CHECK: vld1.8 {d16, d17}, [r0, :64]   @ encoding: [0x1f,0x0a,0x60,0xf4]
@ CHECK: vld1.16 {d16, d17}, [r0, :128] @ encoding: [0x6f,0x0a,0x60,0xf4]
@ CHECK: vld1.32 {d16, d17}, [r0]       @ encoding: [0x8f,0x0a,0x60,0xf4]
@ CHECK: vld1.64 {d16, d17}, [r0]       @ encoding: [0xcf,0x0a,0x60,0xf4]
@ CHECK: vld1.8 {d1, d2, d3}, [r3]      @ encoding: [0x0f,0x16,0x23,0xf4]
@ CHECK: vld1.16 {d4, d5, d6}, [r3, :64] @ encoding: [0x5f,0x46,0x23,0xf4]
@ CHECK: vld1.32 {d5, d6, d7}, [r3]     @ encoding: [0x8f,0x56,0x23,0xf4]
@ CHECK: vld1.64 {d6, d7, d8}, [r3, :64] @ encoding: [0xdf,0x66,0x23,0xf4]
@ CHECK: vld1.8 {d1, d2, d3, d4}, [r3]  @ encoding: [0x0f,0x12,0x23,0xf4]
@ CHECK: vld1.16 {d4, d5, d6, d7}, [r3, :64] @ encoding: [0x5f,0x42,0x23,0xf4]
@ CHECK: vld1.32 {d5, d6, d7, d8}, [r3]  @ encoding: [0x8f,0x52,0x23,0xf4]
@ CHECK: vld1.64 {d6, d7, d8, d9}, [r3, :64] @ encoding: [0xdf,0x62,0x23,0xf4]
@ CHECK: vld1.8	{d16}, [r0, :64]!       @ encoding: [0x1d,0x07,0x60,0xf4]

@ CHECK: vld1.16 {d16}, [r0]!           @ encoding: [0x4d,0x07,0x60,0xf4]
@ CHECK: vld1.32 {d16}, [r0]!           @ encoding: [0x8d,0x07,0x60,0xf4]
@ CHECK: vld1.64 {d16}, [r0]!           @ encoding: [0xcd,0x07,0x60,0xf4]
@ CHECK: vld1.8 {d16, d17}, [r0, :64]!  @ encoding: [0x1d,0x0a,0x60,0xf4]
@ CHECK: vld1.16 {d16, d17}, [r0, :128]! @ encoding: [0x6d,0x0a,0x60,0xf4]
@ CHECK: vld1.32 {d16, d17}, [r0]!      @ encoding: [0x8d,0x0a,0x60,0xf4]
@ CHECK: vld1.64 {d16, d17}, [r0]!      @ encoding: [0xcd,0x0a,0x60,0xf4]

@ CHECK: vld1.8 {d16}, [r0, :64], r5    @ encoding: [0x15,0x07,0x60,0xf4]
@ CHECK: vld1.16 {d16}, [r0], r5        @ encoding: [0x45,0x07,0x60,0xf4]
@ CHECK: vld1.32 {d16}, [r0], r5        @ encoding: [0x85,0x07,0x60,0xf4]
@ CHECK: vld1.64 {d16}, [r0], r5        @ encoding: [0xc5,0x07,0x60,0xf4]
@ CHECK: vld1.8 {d16, d17}, [r0, :64], r5 @ encoding: [0x15,0x0a,0x60,0xf4]
@ CHECK: vld1.16 {d16, d17}, [r0, :128], r5 @ encoding: [0x65,0x0a,0x60,0xf4]
@ CHECK: vld1.32 {d16, d17}, [r0], r5   @ encoding: [0x85,0x0a,0x60,0xf4]
@ CHECK: vld1.64 {d16, d17}, [r0], r5   @ encoding: [0xc5,0x0a,0x60,0xf4]

@ CHECK: vld1.8	{d1, d2, d3}, [r3]!     @ encoding: [0x0d,0x16,0x23,0xf4]
@ CHECK: vld1.16 {d4, d5, d6}, [r3, :64]! @ encoding: [0x5d,0x46,0x23,0xf4]
@ CHECK: vld1.32 {d5, d6, d7}, [r3]!     @ encoding: [0x8d,0x56,0x23,0xf4]
@ CHECK: vld1.64 {d6, d7, d8}, [r3, :64]! @ encoding: [0xdd,0x66,0x23,0xf4]

@ CHECK: vld1.8	{d1, d2, d3}, [r3], r6  @ encoding: [0x06,0x16,0x23,0xf4]
@ CHECK: vld1.16 {d4, d5, d6}, [r3, :64], r6 @ encoding: [0x56,0x46,0x23,0xf4]
@ CHECK: vld1.32 {d5, d6, d7}, [r3], r6  @ encoding: [0x86,0x56,0x23,0xf4]
@ CHECK: vld1.64 {d6, d7, d8}, [r3, :64], r6 @ encoding: [0xd6,0x66,0x23,0xf4]

@ CHECK: vld1.8	{d1, d2, d3, d4}, [r3]! @ encoding: [0x0d,0x12,0x23,0xf4]
@ CHECK: vld1.16 {d4, d5, d6, d7}, [r3, :64]! @ encoding: [0x5d,0x42,0x23,0xf4]
@ CHECK: vld1.32 {d5, d6, d7, d8}, [r3]! @ encoding: [0x8d,0x52,0x23,0xf4]
@ CHECK: vld1.64 {d6, d7, d8, d9}, [r3, :64]! @ encoding: [0xdd,0x62,0x23,0xf4]

@ CHECK: vld1.8	{d1, d2, d3, d4}, [r3], r8 @ encoding: [0x08,0x12,0x23,0xf4]
@ CHECK: vld1.16 {d4, d5, d6, d7}, [r3, :64], r8 @ encoding: [0x58,0x42,0x23,0xf4]
@ CHECK: vld1.32 {d5, d6, d7, d8}, [r3], r8 @ encoding: [0x88,0x52,0x23,0xf4]
@ CHECK: vld1.64 {d6, d7, d8, d9}, [r3, :64], r8 @ encoding: [0xd8,0x62,0x23,0xf4]


	vld2.8	{d16, d17}, [r0, :64]
	vld2.16	{d16, d17}, [r0, :128]
	vld2.32	{d16, d17}, [r0]
	vld2.8	{d16, d17, d18, d19}, [r0, :64]
	vld2.16	{d16, d17, d18, d19}, [r0, :128]
	vld2.32	{d16, d17, d18, d19}, [r0, :256]

@ CHECK: vld2.8	{d16, d17}, [r0, :64]   @ encoding: [0x1f,0x08,0x60,0xf4]
@ CHECK: vld2.16 {d16, d17}, [r0, :128] @ encoding: [0x6f,0x08,0x60,0xf4]
@ CHECK: vld2.32 {d16, d17}, [r0] @ encoding: [0x8f,0x08,0x60,0xf4]
@ CHECK: vld2.8	{d16, d17, d18, d19}, [r0, :64] @ encoding: [0x1f,0x03,0x60,0xf4]
@ CHECK: vld2.16 {d16, d17, d18, d19}, [r0, :128] @ encoding: [0x6f,0x03,0x60,0xf4]
@ CHECK: vld2.32 {d16, d17, d18, d19}, [r0, :256] @ encoding: [0xbf,0x03,0x60,0xf4]


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


	vld1.8 {d4[]}, [r1]
	vld1.8 {d4[]}, [r1]!
	vld1.8 {d4[]}, [r1], r3
	vld1.8 {d4[], d5[]}, [r1]
	vld1.8 {d4[], d5[]}, [r1]!
	vld1.8 {d4[], d5[]}, [r1], r3

@ CHECK: vld1.8	{d4[]}, [r1]            @ encoding: [0x0f,0x4c,0xa1,0xf4]
@ CHECK: vld1.8	{d4[]}, [r1]!           @ encoding: [0x0d,0x4c,0xa1,0xf4]
@ CHECK: vld1.8	{d4[]}, [r1], r3        @ encoding: [0x03,0x4c,0xa1,0xf4]
@ CHECK: vld1.8	{d4[], d5[]}, [r1]      @ encoding: [0x2f,0x4c,0xa1,0xf4]
@ CHECK: vld1.8	{d4[], d5[]}, [r1]!     @ encoding: [0x2d,0x4c,0xa1,0xf4]
@ CHECK: vld1.8	{d4[], d5[]}, [r1], r3  @ encoding: [0x23,0x4c,0xa1,0xf4]

	vld1.8	{d16[3]}, [r0]
@	vld1.16	{d16[2]}, [r0, :16]
@	vld1.32	{d16[1]}, [r0, :32]

@ CHECK: vld1.8	{d16[3]}, [r0]          @ encoding: [0x6f,0x00,0xe0,0xf4]
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


@ Handle 'Q' registers in register lists as if the sub-reg D regs were
@ specified instead.
	vld1.8 {q3}, [r9]
	vld1.8 {q3, q4}, [r9]

@ CHECK: vld1.8	{d6, d7}, [r9]          @ encoding: [0x0f,0x6a,0x29,0xf4]
@ CHECK: vld1.8	{d6, d7, d8, d9}, [r9]  @ encoding: [0x0f,0x62,0x29,0xf4]


@ Spot-check additional size-suffix aliases.
        vld1.8 {d2}, [r2]
        vld1.p8 {d2}, [r2]
        vld1.u8 {d2}, [r2]

        vld1.8 {q2}, [r2]
        vld1.p8 {q2}, [r2]
        vld1.u8 {q2}, [r2]
        vld1.f32 {q2}, [r2]

        vld1.u8 {d2, d3, d4}, [r2]
        vld1.i32 {d2, d3, d4}, [r2]
        vld1.f64 {d2, d3, d4}, [r2]

@ CHECK: vld1.8	{d2}, [r2]              @ encoding: [0x0f,0x27,0x22,0xf4]
@ CHECK: vld1.8	{d2}, [r2]              @ encoding: [0x0f,0x27,0x22,0xf4]
@ CHECK: vld1.8	{d2}, [r2]              @ encoding: [0x0f,0x27,0x22,0xf4]

@ CHECK: vld1.8	{d4, d5}, [r2]          @ encoding: [0x0f,0x4a,0x22,0xf4]
@ CHECK: vld1.8	{d4, d5}, [r2]          @ encoding: [0x0f,0x4a,0x22,0xf4]
@ CHECK: vld1.8	{d4, d5}, [r2]          @ encoding: [0x0f,0x4a,0x22,0xf4]
@ CHECK: vld1.32 {d4, d5}, [r2]         @ encoding: [0x8f,0x4a,0x22,0xf4]

@ CHECK: vld1.8	{d2, d3, d4}, [r2]      @ encoding: [0x0f,0x26,0x22,0xf4]
@ CHECK: vld1.32 {d2, d3, d4}, [r2]     @ encoding: [0x8f,0x26,0x22,0xf4]
@ CHECK: vld1.64 {d2, d3, d4}, [r2]     @ encoding: [0xcf,0x26,0x22,0xf4]


@ Register lists can use the range syntax, just like VLDM
	vld1.f64 {d2-d5}, [r2,:128]!
	vld1.f64 {d2,d3,d4,d5}, [r2,:128]!

@ CHECK: vld1.64 {d2, d3, d4, d5}, [r2, :128]! @ encoding: [0xed,0x22,0x22,0xf4]
@ CHECK: vld1.64 {d2, d3, d4, d5}, [r2, :128]! @ encoding: [0xed,0x22,0x22,0xf4]
