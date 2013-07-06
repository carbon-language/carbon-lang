@ RUN: llvm-mc -triple armv8 -mattr=+v8fp -show-encoding < %s | FileCheck %s

@ VCVT{B,T}

  vcvtt.f64.f16 d3, s1
@ CHECK: vcvtt.f64.f16 d3, s1      @ encoding: [0xe0,0x3b,0xb2,0xee]
  vcvtt.f16.f64 s5, d12
@ CHECK: vcvtt.f16.f64 s5, d12     @ encoding: [0xcc,0x2b,0xf3,0xee]

  vcvtb.f64.f16 d3, s1
@ CHECK: vcvtb.f64.f16 d3, s1     @ encoding: [0x60,0x3b,0xb2,0xee]
  vcvtb.f16.f64 s4, d1
@ CHECK: vcvtb.f16.f64 s4, d1     @ encoding: [0x41,0x2b,0xb3,0xee]

  vcvttge.f64.f16 d3, s1
@ CHECK: vcvttge.f64.f16 d3, s1      @ encoding: [0xe0,0x3b,0xb2,0xae]
  vcvttgt.f16.f64 s5, d12
@ CHECK: vcvttgt.f16.f64 s5, d12     @ encoding: [0xcc,0x2b,0xf3,0xce]

  vcvtbeq.f64.f16 d3, s1
@ CHECK: vcvtbeq.f64.f16 d3, s1     @ encoding: [0x60,0x3b,0xb2,0x0e]
  vcvtblt.f16.f64 s4, d1
@ CHECK: vcvtblt.f16.f64 s4, d1     @ encoding: [0x41,0x2b,0xb3,0xbe]

@ VSEL
  vselge.f32 s4, s1, s23
@ CHECK: vselge.f32 s4, s1, s23    @ encoding: [0xab,0x2a,0x20,0xfe]
  vselge.f64 d30, d31, d23
@ CHECK: vselge.f64 d30, d31, d23  @ encoding: [0xa7,0xeb,0x6f,0xfe]
  vselgt.f32 s0, s1, s0
@ CHECK: vselgt.f32 s0, s1, s0    @ encoding: [0x80,0x0a,0x30,0xfe]
  vselgt.f64 d5, d10, d20
@ CHECK: vselgt.f64 d5, d10, d20  @ encoding: [0x24,0x5b,0x3a,0xfe]
  vseleq.f32 s30, s28, s23
@ CHECK: vseleq.f32 s30, s28, s23 @ encoding: [0x2b,0xfa,0x0e,0xfe]
  vseleq.f64 d2, d4, d8
@ CHECK: vseleq.f64 d2, d4, d8    @ encoding: [0x08,0x2b,0x04,0xfe]
  vselvs.f32 s21, s16, s14
@ CHECK: vselvs.f32 s21, s16, s14 @ encoding: [0x07,0xaa,0x58,0xfe]
  vselvs.f64 d0, d1, d31
@ CHECK: vselvs.f64 d0, d1, d31   @ encoding: [0x2f,0x0b,0x11,0xfe]


@ VMAXNM / VMINNM
  vmaxnm.f32 s5, s12, s0
@ CHECK: vmaxnm.f32 s5, s12, s0    @ encoding: [0x00,0x2a,0xc6,0xfe]
  vmaxnm.f64 d5, d22, d30
@ CHECK: vmaxnm.f64 d5, d22, d30   @ encoding: [0xae,0x5b,0x86,0xfe]
  vminnm.f32 s0, s0, s12
@ CHECK: vminnm.f32 s0, s0, s12    @ encoding: [0x46,0x0a,0x80,0xfe]
  vminnm.f64 d4, d6, d9
@ CHECK: vminnm.f64 d4, d6, d9     @ encoding: [0x49,0x4b,0x86,0xfe]
