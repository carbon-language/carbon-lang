@ RUN: llvm-mc -mcpu=cortex-a8 -triple armv7-apple-darwin -show-encoding < %s | FileCheck %s
@ XFAIL: *

@ CHECK: vst1.8	{d16}, [r0, :64]        @ encoding: [0x1f,0x07,0x40,0xf4]
  vst1.8	{d16}, [r0, :64]
@ CHECK: vst1.16	{d16}, [r0]             @ encoding: [0x4f,0x07,0x40,0xf4]
  vst1.16	{d16}, [r0]
@ CHECK: vst1.32	{d16}, [r0]             @ encoding: [0x8f,0x07,0x40,0xf4]
  vst1.32	{d16}, [r0]
@ CHECK: vst1.64	{d16}, [r0]             @ encoding: [0xcf,0x07,0x40,0xf4]
  vst1.64	{d16}, [r0]
@ CHECK: vst1.8	{d16, d17}, [r0, :64]   @ encoding: [0x1f,0x0a,0x40,0xf4]
  vst1.8	{d16, d17}, [r0, :64]
@ CHECK: vst1.16	{d16, d17}, [r0, :128]  @ encoding: [0x6f,0x0a,0x40,0xf4]
  vst1.16	{d16, d17}, [r0, :128]
@ CHECK: vst1.32	{d16, d17}, [r0]        @ encoding: [0x8f,0x0a,0x40,0xf4]
  vst1.32	{d16, d17}, [r0]
@ CHECK: vst1.64	{d16, d17}, [r0]        @ encoding: [0xcf,0x0a,0x40,0xf4]
  vst1.64	{d16, d17}, [r0]

@ CHECK: vst2.8	{d16, d17}, [r0, :64]   @ encoding: [0x1f,0x08,0x40,0xf4]
  vst2.8	{d16, d17}, [r0, :64]
@ CHECK: vst2.16	{d16, d17}, [r0, :128]  @ encoding: [0x6f,0x08,0x40,0xf4]
  vst2.16	{d16, d17}, [r0, :128]
@ CHECK: vst2.32	{d16, d17}, [r0]        @ encoding: [0x8f,0x08,0x40,0xf4]
  vst2.32	{d16, d17}, [r0]
@ CHECK: vst2.8	{d16, d17, d18, d19}, [r0, :64] @ encoding: [0x1f,0x03,0x40,0xf4]
  vst2.8	{d16, d17, d18, d19}, [r0, :64]
@ CHECK: vst2.16	{d16, d17, d18, d19}, [r0, :128] @ encoding: [0x6f,0x03,0x40,0xf4]
  vst2.16	{d16, d17, d18, d19}, [r0, :128]
@ CHECK: vst2.32	{d16, d17, d18, d19}, [r0, :256] @ encoding: [0xbf,0x03,0x40,0xf4]
  vst2.32	{d16, d17, d18, d19}, [r0, :256]

@ CHECK: vst3.8	{d16, d17, d18}, [r0, :64] @ encoding: [0x1f,0x04,0x40,0xf4]
  vst3.8	{d16, d17, d18}, [r0, :64]
@ CHECK: vst3.16	{d16, d17, d18}, [r0]   @ encoding: [0x4f,0x04,0x40,0xf4]
  vst3.16	{d16, d17, d18}, [r0]
@ CHECK: vst3.32	{d16, d17, d18}, [r0]   @ encoding: [0x8f,0x04,0x40,0xf4]
  vst3.32	{d16, d17, d18}, [r0]
@ CHECK: vst3.8	{d16, d18, d20}, [r0, :64]! @ encoding: [0x1d,0x05,0x40,0xf4]
  vst3.8	{d16, d18, d20}, [r0, :64]!
@ CHECK: vst3.8	{d17, d19, d21}, [r0, :64]! @ encoding: [0x1d,0x15,0x40,0xf4]
  vst3.8	{d17, d19, d21}, [r0, :64]!
@ CHECK: vst3.16	{d16, d18, d20}, [r0]!  @ encoding: [0x4d,0x05,0x40,0xf4]
  vst3.16	{d16, d18, d20}, [r0]!
@ CHECK: vst3.16	{d17, d19, d21}, [r0]!  @ encoding: [0x4d,0x15,0x40,0xf4]
  vst3.16	{d17, d19, d21}, [r0]!
@ CHECK: vst3.32	{d16, d18, d20}, [r0]!  @ encoding: [0x8d,0x05,0x40,0xf4]
  vst3.32	{d16, d18, d20}, [r0]!
@ CHECK: vst3.32	{d17, d19, d21}, [r0]!  @ encoding: [0x8d,0x15,0x40,0xf4]
  vst3.32	{d17, d19, d21}, [r0]!

@ CHECK: vst4.8	{d16, d17, d18, d19}, [r0, :64] @ encoding: [0x1f,0x00,0x40,0xf4]
  vst4.8	{d16, d17, d18, d19}, [r0, :64]
@ CHECK: vst4.16	{d16, d17, d18, d19}, [r0, :128] @ encoding: [0x6f,0x00,0x40,0xf4]
  vst4.16	{d16, d17, d18, d19}, [r0, :128]
@ CHECK: vst4.8	{d16, d18, d20, d22}, [r0, :256]! @ encoding: [0x3d,0x01,0x40,0xf4]
  vst4.8	{d16, d18, d20, d22}, [r0, :256]!
@ CHECK: vst4.8	{d17, d19, d21, d23}, [r0, :256]! @ encoding: [0x3d,0x11,0x40,0xf4]
  vst4.8	{d17, d19, d21, d23}, [r0, :256]!
@ CHECK: vst4.16	{d16, d18, d20, d22}, [r0]! @ encoding: [0x4d,0x01,0x40,0xf4]
  vst4.16	{d16, d18, d20, d22}, [r0]!
@ CHECK: vst4.16	{d17, d19, d21, d23}, [r0]! @ encoding: [0x4d,0x11,0x40,0xf4]
  vst4.16	{d17, d19, d21, d23}, [r0]!
@ CHECK: vst4.32	{d16, d18, d20, d22}, [r0]! @ encoding: [0x8d,0x01,0x40,0xf4]
  vst4.32	{d16, d18, d20, d22}, [r0]!
@ CHECK: vst4.32	{d17, d19, d21, d23}, [r0]! @ encoding: [0x8d,0x11,0x40,0xf4]
  vst4.32	{d17, d19, d21, d23}, [r0]!
