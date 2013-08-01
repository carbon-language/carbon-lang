// RUN: llvm-mc -triple aarch64-none-linux-gnu -mattr=+neon -show-encoding < %s | FileCheck %s

// Check that the assembler can handle the documented syntax for AArch64


//------------------------------------------------------------------------------
// Vector Integer Saturating Rounding Shift Lef (Signed)
//------------------------------------------------------------------------------
         sqrshl v0.8b, v1.8b, v2.8b
         sqrshl v0.16b, v1.16b, v2.16b
         sqrshl v0.4h, v1.4h, v2.4h
         sqrshl v0.8h, v1.8h, v2.8h
         sqrshl v0.2s, v1.2s, v2.2s
         sqrshl v0.4s, v1.4s, v2.4s
         sqrshl v0.2d, v1.2d, v2.2d

// CHECK: sqrshl v0.8b, v1.8b, v2.8b        // encoding: [0x20,0x5c,0x22,0x0e]
// CHECK: sqrshl v0.16b, v1.16b, v2.16b     // encoding: [0x20,0x5c,0x22,0x4e]
// CHECK: sqrshl v0.4h, v1.4h, v2.4h        // encoding: [0x20,0x5c,0x62,0x0e]
// CHECK: sqrshl v0.8h, v1.8h, v2.8h        // encoding: [0x20,0x5c,0x62,0x4e]
// CHECK: sqrshl v0.2s, v1.2s, v2.2s        // encoding: [0x20,0x5c,0xa2,0x0e]
// CHECK: sqrshl v0.4s, v1.4s, v2.4s        // encoding: [0x20,0x5c,0xa2,0x4e]
// CHECK: sqrshl v0.2d, v1.2d, v2.2d        // encoding: [0x20,0x5c,0xe2,0x4e]

//------------------------------------------------------------------------------
// Vector Integer Saturating Rounding Shift Lef (Unsigned)
//------------------------------------------------------------------------------
         uqrshl v0.8b, v1.8b, v2.8b
         uqrshl v0.16b, v1.16b, v2.16b
         uqrshl v0.4h, v1.4h, v2.4h
         uqrshl v0.8h, v1.8h, v2.8h
         uqrshl v0.2s, v1.2s, v2.2s
         uqrshl v0.4s, v1.4s, v2.4s
         uqrshl v0.2d, v1.2d, v2.2d

// CHECK: uqrshl v0.8b, v1.8b, v2.8b        // encoding: [0x20,0x5c,0x22,0x2e]
// CHECK: uqrshl v0.16b, v1.16b, v2.16b     // encoding: [0x20,0x5c,0x22,0x6e]
// CHECK: uqrshl v0.4h, v1.4h, v2.4h        // encoding: [0x20,0x5c,0x62,0x2e]
// CHECK: uqrshl v0.8h, v1.8h, v2.8h        // encoding: [0x20,0x5c,0x62,0x6e]
// CHECK: uqrshl v0.2s, v1.2s, v2.2s        // encoding: [0x20,0x5c,0xa2,0x2e]
// CHECK: uqrshl v0.4s, v1.4s, v2.4s        // encoding: [0x20,0x5c,0xa2,0x6e]
// CHECK: uqrshl v0.2d, v1.2d, v2.2d        // encoding: [0x20,0x5c,0xe2,0x6e]

//------------------------------------------------------------------------------
// Scalar Integer Saturating Rounding Shift Lef (Signed)
//------------------------------------------------------------------------------
         sqrshl b0, b1, b2
         sqrshl h10, h11, h12
         sqrshl s20, s21, s2
         sqrshl d17, d31, d8

// CHECK: sqrshl b0, b1, b2        // encoding: [0x20,0x5c,0x22,0x5e]
// CHECK: sqrshl h10, h11, h12     // encoding: [0x6a,0x5d,0x6c,0x5e]
// CHECK: sqrshl s20, s21, s2      // encoding: [0xb4,0x5e,0xa2,0x5e]
// CHECK: sqrshl d17, d31, d8      // encoding: [0xf1,0x5f,0xe8,0x5e]

//------------------------------------------------------------------------------
// Scalar Integer Saturating Rounding Shift Lef (Unsigned)
//------------------------------------------------------------------------------
         uqrshl b0, b1, b2
         uqrshl h10, h11, h12
         uqrshl s20, s21, s2
         uqrshl d17, d31, d8

// CHECK: uqrshl b0, b1, b2        // encoding: [0x20,0x5c,0x22,0x7e]
// CHECK: uqrshl h10, h11, h12     // encoding: [0x6a,0x5d,0x6c,0x7e]
// CHECK: uqrshl s20, s21, s2      // encoding: [0xb4,0x5e,0xa2,0x7e]
// CHECK: uqrshl d17, d31, d8      // encoding: [0xf1,0x5f,0xe8,0x7e]


