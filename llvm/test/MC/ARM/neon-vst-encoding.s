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

