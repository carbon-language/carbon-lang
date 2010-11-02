@ RUN: llvm-mc -mcpu=cortex-a8 -triple armv7-apple-darwin -show-encoding < %s | FileCheck %s
@ XFAIL: *

@ CHECK: vld1.8	{d16}, [r0, :64]        @ encoding: [0x1f,0x07,0x60,0xf4]
	vld1.8	{d16}, [r0, :64]
@ CHECK: vld1.16	{d16}, [r0]             @ encoding: [0x4f,0x07,0x60,0xf4]
  vld1.16	{d16}, [r0]
@ CHECK: vld1.32	{d16}, [r0]             @ encoding: [0x8f,0x07,0x60,0xf4]
  vld1.32	{d16}, [r0]
@ CHECK: vld1.64	{d16}, [r0]             @ encoding: [0xcf,0x07,0x60,0xf4]
  vld1.64	{d16}, [r0]
@ CHECK: vld1.8	{d16, d17}, [r0, :64]   @ encoding: [0x1f,0x0a,0x60,0xf4]
  vld1.8	{d16, d17}, [r0, :64]
@ CHECK: vld1.16	{d16, d17}, [r0, :128]  @ encoding: [0x6f,0x0a,0x60,0xf4]
  vld1.16	{d16, d17}, [r0, :128]
@ CHECK: vld1.32	{d16, d17}, [r0]        @ encoding: [0x8f,0x0a,0x60,0xf4]
  vld1.32	{d16, d17}, [r0]
@ CHECK: vld1.64	{d16, d17}, [r0]        @ encoding: [0xcf,0x0a,0x60,0xf4]
  vld1.64	{d16, d17}, [r0]

@ CHECK: vld2.8	{d16, d17}, [r0, :64]   @ encoding: [0x1f,0x08,0x60,0xf4]
  vld2.8	{d16, d17}, [r0, :64]
@ CHECK: vld2.16	{d16, d17}, [r0, :128]  @ encoding: [0x6f,0x08,0x60,0xf4]
  vld2.16	{d16, d17}, [r0, :128]
@ CHECK: vld2.32	{d16, d17}, [r0]        @ encoding: [0x8f,0x08,0x60,0xf4]
  vld2.32	{d16, d17}, [r0]
@ CHECK: vld2.8	{d16, d17, d18, d19}, [r0, :64] @ encoding: [0x1f,0x03,0x60,0xf4]
  vld2.8	{d16, d17, d18, d19}, [r0, :64]
@ CHECK: vld2.16	{d16, d17, d18, d19}, [r0, :128] @ encoding: [0x6f,0x03,0x60,0xf4]
  vld2.16	{d16, d17, d18, d19}, [r0, :128]
@ CHECK: vld2.32	{d16, d17, d18, d19}, [r0, :256] @ encoding: [0xbf,0x03,0x60,0xf4]
  vld2.32	{d16, d17, d18, d19}, [r0, :256]

@ CHECK: vld3.8	{d16, d17, d18}, [r0, :64] @ encoding: [0x1f,0x04,0x60,0xf4]
  vld3.8	{d16, d17, d18}, [r0, :64]
@ CHECK: vld3.16	{d16, d17, d18}, [r0]   @ encoding: [0x4f,0x04,0x60,0xf4]
  vld3.16	{d16, d17, d18}, [r0]
@ CHECK: vld3.32	{d16, d17, d18}, [r0]   @ encoding: [0x8f,0x04,0x60,0xf4]
  vld3.32	{d16, d17, d18}, [r0]
@ CHECK: vld3.8	{d16, d18, d20}, [r0, :64]! @ encoding: [0x1d,0x05,0x60,0xf4]
  vld3.8	{d16, d18, d20}, [r0, :64]!
@ CHECK: vld3.8	{d17, d19, d21}, [r0, :64]! @ encoding: [0x1d,0x15,0x60,0xf4]
  vld3.8	{d17, d19, d21}, [r0, :64]!
@ CHECK: vld3.16	{d16, d18, d20}, [r0]!  @ encoding: [0x4d,0x05,0x60,0xf4] 
  vld3.16	{d16, d18, d20}, [r0]!
@ CHECK: vld3.16	{d17, d19, d21}, [r0]!  @ encoding: [0x4d,0x15,0x60,0xf4]
  vld3.16	{d17, d19, d21}, [r0]!
@ CHECK: vld3.32	{d16, d18, d20}, [r0]!  @ encoding: [0x8d,0x05,0x60,0xf4]
  vld3.32	{d16, d18, d20}, [r0]!
@ CHECK: vld3.32	{d17, d19, d21}, [r0]!  @ encoding: [0x8d,0x15,0x60,0xf4]
  vld3.32	{d17, d19, d21}, [r0]!

@ CHECK: vld4.8	{d16, d17, d18, d19}, [r0, :64] @ encoding: [0x1f,0x00,0x60,0xf4]
  vld4.8	{d16, d17, d18, d19}, [r0, :64]
@ CHECK: vld4.16	{d16, d17, d18, d19}, [r0, :128] @ encoding: [0x6f,0x00,0x60,0xf4]
  vld4.16	{d16, d17, d18, d19}, [r0, :128]
@ CHECK: vld4.32	{d16, d17, d18, d19}, [r0, :256] @ encoding: [0xbf,0x00,0x60,0xf4]
  vld4.32	{d16, d17, d18, d19}, [r0, :256]
@ CHECK: vld4.8	{d16, d18, d20, d22}, [r0, :256]! @ encoding: [0x3d,0x01,0x60,0xf4]
  vld4.8	{d16, d18, d20, d22}, [r0, :256]!
@ CHECK: vld4.8	{d17, d19, d21, d23}, [r0, :256]! @ encoding: [0x3d,0x11,0x60,0xf4]
  vld4.8	{d17, d19, d21, d23}, [r0, :256]!
@ CHECK: vld4.16	{d16, d18, d20, d22}, [r0]! @ encoding: [0x4d,0x01,0x60,0xf4]
  vld4.16	{d16, d18, d20, d22}, [r0]!
@ CHECK: vld4.16	{d17, d19, d21, d23}, [r0]! @ encoding: [0x4d,0x11,0x60,0xf4]
  vld4.16	{d17, d19, d21, d23}, [r0]!
@ CHECK: vld4.32	{d16, d18, d20, d22}, [r0]! @ encoding: [0x8d,0x01,0x60,0xf4]
  vld4.32	{d16, d18, d20, d22}, [r0]!
@ CHECK: vld4.32	{d17, d19, d21, d23}, [r0]! @ encoding: [0x8d,0x11,0x60,0xf4]
  vld4.32	{d17, d19, d21, d23}, [r0]!

  
  