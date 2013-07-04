@ RUN: llvm-mc -triple armv7 -show-encoding < %s | FileCheck %s

@ VCVT{B,T}

  vcvtt.f64.f16 d3, s1
@ CHECK-NOT: vcvtt.f64.f16 d3, s1      @ encoding: [0xe0,0x3b,0xb2,0xee]
  vcvtt.f16.f64 s5, d12
@ CHECK-NOT: vcvtt.f16.f64 s5, d12     @ encoding: [0xcc,0x2b,0xf3,0xee]


