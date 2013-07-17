@ RUN: llvm-mc -triple thumbv8 -mattr=+neon -show-encoding < %s | FileCheck %s

vmaxnm.f32 d4, d5, d1
@ CHECK: vmaxnm.f32 d4, d5, d1 @ encoding: [0x05,0xff,0x11,0x4f]
vmaxnm.f32 q2, q4, q6
@ CHECK: vmaxnm.f32 q2, q4, q6 @ encoding: [0x08,0xff,0x5c,0x4f]
vminnm.f32 d5, d4, d30
@ CHECK: vminnm.f32 d5, d4, d30 @ encoding: [0x24,0xff,0x3e,0x5f]
vminnm.f32 q0, q13, q2
@ CHECK: vminnm.f32 q0, q13, q2 @ encoding: [0x2a,0xff,0xd4,0x0f]
