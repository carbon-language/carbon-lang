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
