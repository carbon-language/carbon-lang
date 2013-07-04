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
