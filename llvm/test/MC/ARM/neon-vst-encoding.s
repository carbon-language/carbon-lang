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


@	vst2.8	{d16, d17}, [r0, :64]
@	vst2.16	{d16, d17}, [r0, :128]
@	vst2.32	{d16, d17}, [r0]
@	vst2.8	{d16, d17, d18, d19}, [r0, :64]
@	vst2.16	{d16, d17, d18, d19}, [r0, :128]
@	vst2.32	{d16, d17, d18, d19}, [r0, :256]

@ FIXME: vst2.8	{d16, d17}, [r0, :64]   @ encoding: [0x1f,0x08,0x40,0xf4]
@ FIXME: vst2.16 {d16, d17}, [r0, :128] @ encoding: [0x6f,0x08,0x40,0xf4]
@ FIXME: vst2.32 {d16, d17}, [r0]       @ encoding: [0x8f,0x08,0x40,0xf4]
@ FIXME: vst2.8	{d16, d17, d18, d19}, [r0, :64] @ encoding: [0x1f,0x03,0x40,0xf4]
@ FIXME: vst2.16 {d16, d17, d18, d19}, [r0, :128] @ encoding: [0x6f,0x03,0x40,0xf4]
@ FIXME: vst2.32 {d16, d17, d18, d19}, [r0, :256] @ encoding: [0xbf,0x03,0x40,0xf4]


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


@	vst2.8	{d16[1], d17[1]}, [r0, :16]
@	vst2.16	{d16[1], d17[1]}, [r0, :32]
@	vst2.32	{d16[1], d17[1]}, [r0]
@	vst2.16	{d17[1], d19[1]}, [r0]
@	vst2.32	{d17[0], d19[0]}, [r0, :64]

@ FIXME: vst2.8	{d16[1], d17[1]}, [r0, :16] @ encoding: [0x3f,0x01,0xc0,0xf4]
@ FIXME: vst2.16 {d16[1], d17[1]}, [r0, :32] @ encoding: [0x5f,0x05,0xc0,0xf4]
@ FIXME: vst2.32 {d16[1], d17[1]}, [r0]  @ encoding: [0x8f,0x09,0xc0,0xf4]
@ FIXME: vst2.16 {d17[1], d19[1]}, [r0]  @ encoding: [0x6f,0x15,0xc0,0xf4]
@ FIXME: vst2.32 {d17[0], d19[0]}, [r0, :64] @ encoding: [0x5f,0x19,0xc0,0xf4]


@	vst3.8	{d16[1], d17[1], d18[1]}, [r0]
@	vst3.16	{d16[1], d17[1], d18[1]}, [r0]
@	vst3.32	{d16[1], d17[1], d18[1]}, [r0]
@	vst3.16	{d17[2], d19[2], d21[2]}, [r0]
@	vst3.32	{d16[0], d18[0], d20[0]}, [r0]

@ FIXME: vst3.8	{d16[1], d17[1], d18[1]}, [r0] @ encoding: [0x2f,0x02,0xc0,0xf4]
@ FIXME: vst3.16 {d16[1], d17[1], d18[1]}, [r0]@ encoding: [0x4f,0x06,0xc0,0xf4]
@ FIXME: vst3.32 {d16[1], d17[1], d18[1]}, [r0]@ encoding: [0x8f,0x0a,0xc0,0xf4]
@ FIXME: vst3.16 {d17[2], d19[2], d21[2]}, [r0]@ encoding: [0xaf,0x16,0xc0,0xf4]
@ FIXME: vst3.32 {d16[0], d18[0], d20[0]}, [r0]@ encoding: [0x4f,0x0a,0xc0,0xf4]


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
