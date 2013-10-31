// RUN: llvm-mc -triple aarch64-none-linux-gnu -mattr=+neon -show-encoding < %s | FileCheck %s

// Check that the assembler can handle the documented syntax for AArch64

//----------------------------------------------------------------------
// Scalar Signed Integer Convert To Floating-point
//----------------------------------------------------------------------

    scvtf s22, s13
    scvtf d21, d12

// CHECK: scvtf s22, s13    // encoding: [0xb6,0xd9,0x21,0x5e]
// CHECK: scvtf d21, d12    // encoding: [0x95,0xd9,0x61,0x5e]

//----------------------------------------------------------------------
// Scalar Unsigned Integer Convert To Floating-point
//----------------------------------------------------------------------

    ucvtf s22, s13
    ucvtf d21, d14

// CHECK: ucvtf s22, s13    // encoding: [0xb6,0xd9,0x21,0x7e]
// CHECK: ucvtf d21, d14    // encoding: [0xd5,0xd9,0x61,0x7e]

//----------------------------------------------------------------------
// Scalar Signed Fixed-point Convert To Floating-Point (Immediate)
//----------------------------------------------------------------------

    scvtf s22, s13, #32
    scvtf d21, d12, #64

// CHECK: scvtf s22, s13, #32  // encoding: [0xb6,0xe5,0x20,0x5f]
// CHECK: scvtf d21, d12, #64  // encoding: [0x95,0xe5,0x40,0x5f]    

//----------------------------------------------------------------------
// Scalar Unsigned Fixed-point Convert To Floating-Point (Immediate)
//----------------------------------------------------------------------

    ucvtf s22, s13, #32
    ucvtf d21, d14, #64

// CHECK: ucvtf s22, s13, #32  // encoding: [0xb6,0xe5,0x20,0x7f]
// CHECK: ucvtf d21, d14, #64  // encoding: [0xd5,0xe5,0x40,0x7f]
