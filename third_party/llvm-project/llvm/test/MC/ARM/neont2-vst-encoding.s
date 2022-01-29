@ RUN: llvm-mc -mcpu=cortex-a8 -triple thumb-unknown-unknown -show-encoding < %s | FileCheck %s

.code 16

@ CHECK: vst1.8	{d16}, [r0:64]        @ encoding: [0x40,0xf9,0x1f,0x07]
  vst1.8	{d16}, [r0:64]
@ CHECK: vst1.16	{d16}, [r0]             @ encoding: [0x40,0xf9,0x4f,0x07]
  vst1.16	{d16}, [r0]
@ CHECK: vst1.32	{d16}, [r0]             @ encoding: [0x40,0xf9,0x8f,0x07]
  vst1.32	{d16}, [r0]
@ CHECK: vst1.64	{d16}, [r0]             @ encoding: [0x40,0xf9,0xcf,0x07]
  vst1.64	{d16}, [r0]
@ CHECK: vst1.8	{d16, d17}, [r0:64]   @ encoding: [0x40,0xf9,0x1f,0x0a]
  vst1.8	{d16, d17}, [r0:64]
@ CHECK: vst1.16	{d16, d17}, [r0:128]  @ encoding: [0x40,0xf9,0x6f,0x0a]
  vst1.16	{d16, d17}, [r0:128]
@ CHECK: vst1.32	{d16, d17}, [r0]        @ encoding: [0x40,0xf9,0x8f,0x0a]
  vst1.32	{d16, d17}, [r0]
@ CHECK: vst1.64	{d16, d17}, [r0]        @ encoding: [0x40,0xf9,0xcf,0x0a]
  vst1.64	{d16, d17}, [r0]

@ CHECK: vst2.8	{d16, d17}, [r0:64]   @ encoding: [0x40,0xf9,0x1f,0x08]
  vst2.8	{d16, d17}, [r0:64]
@ CHECK: vst2.16	{d16, d17}, [r0:128]  @ encoding: [0x40,0xf9,0x6f,0x08]
  vst2.16	{d16, d17}, [r0:128]
@ CHECK: vst2.32	{d16, d17}, [r0]        @ encoding: [0x40,0xf9,0x8f,0x08]
  vst2.32	{d16, d17}, [r0]
@ CHECK: vst2.8	{d16, d17, d18, d19}, [r0:64] @ encoding: [0x40,0xf9,0x1f,0x03]
  vst2.8	{d16, d17, d18, d19}, [r0:64]
@ CHECK: vst2.16	{d16, d17, d18, d19}, [r0:128] @ encoding: [0x40,0xf9,0x6f,0x03]
  vst2.16	{d16, d17, d18, d19}, [r0:128]
@ CHECK: vst2.32	{d16, d17, d18, d19}, [r0:256] @ encoding: [0x40,0xf9,0xbf,0x03]
  vst2.32	{d16, d17, d18, d19}, [r0:256]

@ CHECK: vst3.8	{d16, d17, d18}, [r0:64] @ encoding: [0x40,0xf9,0x1f,0x04]
  vst3.8	{d16, d17, d18}, [r0:64]
@ CHECK: vst3.16	{d16, d17, d18}, [r0]   @ encoding: [0x40,0xf9,0x4f,0x04]
  vst3.16	{d16, d17, d18}, [r0]
@ CHECK: vst3.32	{d16, d17, d18}, [r0]   @ encoding: [0x40,0xf9,0x8f,0x04]
  vst3.32	{d16, d17, d18}, [r0]
@ CHECK: vst3.8	{d16, d18, d20}, [r0:64]! @ encoding: [0x40,0xf9,0x1d,0x05]
  vst3.8	{d16, d18, d20}, [r0:64]!
@ CHECK: vst3.8	{d17, d19, d21}, [r0:64]! @ encoding: [0x40,0xf9,0x1d,0x15]
  vst3.8	{d17, d19, d21}, [r0:64]!
@ CHECK: vst3.16	{d16, d18, d20}, [r0]!  @ encoding: [0x40,0xf9,0x4d,0x05]
  vst3.16	{d16, d18, d20}, [r0]!
@ CHECK: vst3.16	{d17, d19, d21}, [r0]!  @ encoding: [0x40,0xf9,0x4d,0x15]
  vst3.16	{d17, d19, d21}, [r0]!
@ CHECK: vst3.32	{d16, d18, d20}, [r0]!  @ encoding: [0x40,0xf9,0x8d,0x05]
  vst3.32	{d16, d18, d20}, [r0]!
@ CHECK: vst3.32	{d17, d19, d21}, [r0]!  @ encoding: [0x40,0xf9,0x8d,0x15]
  vst3.32	{d17, d19, d21}, [r0]!

@ CHECK: vst4.8	{d16, d17, d18, d19}, [r0:64] @ encoding: [0x40,0xf9,0x1f,0x00]
  vst4.8	{d16, d17, d18, d19}, [r0:64]
@ CHECK: vst4.16	{d16, d17, d18, d19}, [r0:128] @ encoding: [0x40,0xf9,0x6f,0x00]
  vst4.16	{d16, d17, d18, d19}, [r0:128]
@ CHECK: vst4.8	{d16, d18, d20, d22}, [r0:256]! @ encoding: [0x40,0xf9,0x3d,0x01]
  vst4.8	{d16, d18, d20, d22}, [r0:256]!
@ CHECK: vst4.8	{d17, d19, d21, d23}, [r0:256]! @ encoding: [0x40,0xf9,0x3d,0x11]
  vst4.8	{d17, d19, d21, d23}, [r0:256]!
@ CHECK: vst4.16	{d16, d18, d20, d22}, [r0]! @ encoding: [0x40,0xf9,0x4d,0x01]
  vst4.16	{d16, d18, d20, d22}, [r0]!
@ CHECK: vst4.16	{d17, d19, d21, d23}, [r0]! @ encoding: [0x40,0xf9,0x4d,0x11]
  vst4.16	{d17, d19, d21, d23}, [r0]!
@ CHECK: vst4.32	{d16, d18, d20, d22}, [r0]! @ encoding: [0x40,0xf9,0x8d,0x01]
  vst4.32	{d16, d18, d20, d22}, [r0]!
@ CHECK: vst4.32	{d17, d19, d21, d23}, [r0]! @ encoding: [0x40,0xf9,0x8d,0x11]
  vst4.32	{d17, d19, d21, d23}, [r0]!

@ CHECK: vst2.8	{d16[1], d17[1]}, [r0:16] @ encoding: [0xc0,0xf9,0x3f,0x01]
  vst2.8	{d16[1], d17[1]}, [r0:16]
@ CHECK: vst2.16	{d16[1], d17[1]}, [r0:32] @ encoding: [0xc0,0xf9,0x5f,0x05]
  vst2.16	{d16[1], d17[1]}, [r0:32]
@ CHECK: vst2.32	{d16[1], d17[1]}, [r0]  @ encoding: [0xc0,0xf9,0x8f,0x09]
  vst2.32	{d16[1], d17[1]}, [r0]
@ CHECK: vst2.16	{d17[1], d19[1]}, [r0]  @ encoding: [0xc0,0xf9,0x6f,0x15]
  vst2.16	{d17[1], d19[1]}, [r0]
@ CHECK: vst2.32	{d17[0], d19[0]}, [r0:64] @ encoding: [0xc0,0xf9,0x5f,0x19]
  vst2.32	{d17[0], d19[0]}, [r0:64]

@ CHECK: vst3.8	{d16[1], d17[1], d18[1]}, [r0] @ encoding: [0xc0,0xf9,0x2f,0x02]
  vst3.8	{d16[1], d17[1], d18[1]}, [r0]
@ CHECK: vst3.16	{d16[1], d17[1], d18[1]}, [r0] @ encoding: [0xc0,0xf9,0x4f,0x06]
  vst3.16	{d16[1], d17[1], d18[1]}, [r0]
@ CHECK: vst3.32	{d16[1], d17[1], d18[1]}, [r0] @ encoding: [0xc0,0xf9,0x8f,0x0a]
  vst3.32	{d16[1], d17[1], d18[1]}, [r0]
@ CHECK: vst3.16	{d17[2], d19[2], d21[2]}, [r0] @ encoding: [0xc0,0xf9,0xaf,0x16]
  vst3.16	{d17[2], d19[2], d21[2]}, [r0]
@ CHECK: vst3.32	{d16[0], d18[0], d20[0]}, [r0] @ encoding: [0xc0,0xf9,0x4f,0x0a]
  vst3.32	{d16[0], d18[0], d20[0]}, [r0]

@ CHECK: vst4.8	{d16[1], d17[1], d18[1], d19[1]}, [r0:32] @ encoding: [0xc0,0xf9,0x3f,0x03]
  vst4.8	{d16[1], d17[1], d18[1], d19[1]}, [r0:32]
@ CHECK: vst4.16	{d16[1], d17[1], d18[1], d19[1]}, [r0] @ encoding: [0xc0,0xf9,0x4f,0x07]
  vst4.16	{d16[1], d17[1], d18[1], d19[1]}, [r0]
@ CHECK: vst4.32	{d16[1], d17[1], d18[1], d19[1]}, [r0:128] @ encoding: [0xc0,0xf9,0xaf,0x0b]
  vst4.32	{d16[1], d17[1], d18[1], d19[1]}, [r0:128]
@ CHECK: vst4.16	{d17[3], d19[3], d21[3], d23[3]}, [r0:64] @ encoding: [0xc0,0xf9,0xff,0x17]
  vst4.16	{d17[3], d19[3], d21[3], d23[3]}, [r0:64]
@ CHECK: vst4.32	{d17[0], d19[0], d21[0], d23[0]}, [r0] @ encoding: [0xc0,0xf9,0x4f,0x1b]
  vst4.32	{d17[0], d19[0], d21[0], d23[0]}, [r0]

@ rdar://11082188
        vst2.8 {d8, d10}, [r4]
@ CHECK: vst2.8	{d8, d10}, [r4]         @ encoding: [0x04,0xf9,0x0f,0x89]
