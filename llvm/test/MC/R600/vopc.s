// RUN: llvm-mc -arch=amdgcn -show-encoding %s | FileCheck %s
// RUN: llvm-mc -arch=amdgcn -mcpu=SI -show-encoding %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Generic Checks
//===----------------------------------------------------------------------===//

// src0 sgpr
v_cmp_lt_f32 vcc, s2, v4
// CHECK: v_cmp_lt_f32_e32 vcc, s2, v4 ; encoding: [0x02,0x08,0x02,0x7c]

// src0 inline immediate
v_cmp_lt_f32 vcc, 0, v4
// CHECK: v_cmp_lt_f32_e32 vcc, 0, v4 ; encoding: [0x80,0x08,0x02,0x7c]

// src0 literal
v_cmp_lt_f32 vcc, 10.0, v4
// CHECK: v_cmp_lt_f32_e32 vcc, 0x41200000, v4 ; encoding: [0xff,0x08,0x02,0x7c,0x00,0x00,0x20,0x41]

// src0, src1 max vgpr
v_cmp_lt_f32 vcc, v255, v255
// CHECK: v_cmp_lt_f32_e32 vcc, v255, v255 ; encoding: [0xff,0xff,0x03,0x7c]

// force 32-bit encoding
v_cmp_lt_f32_e32 vcc, v2, v4
// CHECK: v_cmp_lt_f32_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x02,0x7c]


//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

v_cmp_f_f32 vcc, v2, v4
// CHECK: v_cmp_f_f32_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x00,0x7c]

v_cmp_lt_f32 vcc, v2, v4
// CHECK: v_cmp_lt_f32_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x02,0x7c]

// TODO: Add tests for the rest of the instructions.

