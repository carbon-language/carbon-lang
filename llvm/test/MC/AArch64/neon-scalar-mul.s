// RUN: llvm-mc -triple aarch64-none-linux-gnu -mattr=+neon -show-encoding < %s | FileCheck %s

// Check that the assembler can handle the documented syntax for AArch64

//----------------------------------------------------------------------
// Scalar Integer Saturating Doubling Multiply Half High
//----------------------------------------------------------------------

    sqdmulh h10, h11, h12
    sqdmulh s20, s21, s2
        
// CHECK: sqdmulh h10, h11, h12     // encoding: [0x6a,0xb5,0x6c,0x5e]
// CHECK: sqdmulh s20, s21, s2      // encoding: [0xb4,0xb6,0xa2,0x5e]

//----------------------------------------------------------------------
// Scalar Integer Saturating Rounding Doubling Multiply Half High
//----------------------------------------------------------------------

    sqrdmulh h10, h11, h12
    sqrdmulh s20, s21, s2
        
// CHECK: sqrdmulh h10, h11, h12     // encoding: [0x6a,0xb5,0x6c,0x7e]
// CHECK: sqrdmulh s20, s21, s2      // encoding: [0xb4,0xb6,0xa2,0x7e]

//----------------------------------------------------------------------
// Floating-point Multiply Extended
//----------------------------------------------------------------------

    fmulx s20, s22, s15
    fmulx d23, d11, d1

// CHECK: fmulx s20, s22, s15   // encoding: [0xd4,0xde,0x2f,0x5e]
// CHECK: fmulx d23, d11, d1    // encoding: [0x77,0xdd,0x61,0x5e]
