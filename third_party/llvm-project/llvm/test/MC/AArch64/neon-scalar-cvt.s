// RUN: llvm-mc -triple aarch64-none-linux-gnu -mattr=+neon,+fullfp16 -show-encoding < %s | FileCheck %s

// Check that the assembler can handle the documented syntax for AArch64

//----------------------------------------------------------------------
// Scalar Signed Integer Convert To Floating-point
//----------------------------------------------------------------------

    scvtf h23, h14
    scvtf s22, s13
    scvtf d21, d12

// CHECK: scvtf   h23, h14                // encoding: [0xd7,0xd9,0x79,0x5e]
// CHECK: scvtf s22, s13    // encoding: [0xb6,0xd9,0x21,0x5e]
// CHECK: scvtf d21, d12    // encoding: [0x95,0xd9,0x61,0x5e]

//----------------------------------------------------------------------
// Scalar Unsigned Integer Convert To Floating-point
//----------------------------------------------------------------------

    ucvtf h20, h12
    ucvtf s22, s13
    ucvtf d21, d14

// CHECK: ucvtf   h20, h12                // encoding: [0x94,0xd9,0x79,0x7e]
// CHECK: ucvtf s22, s13    // encoding: [0xb6,0xd9,0x21,0x7e]
// CHECK: ucvtf d21, d14    // encoding: [0xd5,0xd9,0x61,0x7e]

//----------------------------------------------------------------------
// Scalar Signed Fixed-point Convert To Floating-Point (Immediate)
//----------------------------------------------------------------------

    scvtf h22, h13, #16
    scvtf s22, s13, #32
    scvtf d21, d12, #64

// CHECK: scvtf   h22, h13, #16           // encoding: [0xb6,0xe5,0x10,0x5f]
// CHECK: scvtf s22, s13, #32  // encoding: [0xb6,0xe5,0x20,0x5f]
// CHECK: scvtf d21, d12, #64  // encoding: [0x95,0xe5,0x40,0x5f]    

//----------------------------------------------------------------------
// Scalar Unsigned Fixed-point Convert To Floating-Point (Immediate)
//----------------------------------------------------------------------

    ucvtf h22, h13, #16
    ucvtf s22, s13, #32
    ucvtf d21, d14, #64

// CHECK: ucvtf   h22, h13, #16           // encoding: [0xb6,0xe5,0x10,0x7f]
// CHECK: ucvtf s22, s13, #32  // encoding: [0xb6,0xe5,0x20,0x7f]
// CHECK: ucvtf d21, d14, #64  // encoding: [0xd5,0xe5,0x40,0x7f]

//----------------------------------------------------------------------
// Scalar Floating-point Convert To Signed Fixed-point (Immediate)
//----------------------------------------------------------------------

    fcvtzs h21, h12, #1
    fcvtzs s21, s12, #1
    fcvtzs d21, d12, #1

// CHECK: fcvtzs  h21, h12, #1            // encoding: [0x95,0xfd,0x1f,0x5f]
// CHECK: fcvtzs s21, s12, #1  // encoding: [0x95,0xfd,0x3f,0x5f]
// CHECK: fcvtzs d21, d12, #1  // encoding: [0x95,0xfd,0x7f,0x5f]
        
//----------------------------------------------------------------------
// Scalar Floating-point Convert To Unsigned Fixed-point (Immediate)
//----------------------------------------------------------------------

    fcvtzu h21, h12, #1
    fcvtzu s21, s12, #1
    fcvtzu d21, d12, #1

// CHECK: fcvtzu  h21, h12, #1            // encoding: [0x95,0xfd,0x1f,0x7f]
// CHECK: fcvtzu s21, s12, #1  // encoding: [0x95,0xfd,0x3f,0x7f]
// CHECK: fcvtzu d21, d12, #1  // encoding: [0x95,0xfd,0x7f,0x7f]

//----------------------------------------------------------------------
// Scalar Floating-point Convert To Lower Precision Narrow, Rounding To
// Odd
//----------------------------------------------------------------------

    fcvtxn s22, d13

// CHECK: fcvtxn s22, d13    // encoding: [0xb6,0x69,0x61,0x7e]

//----------------------------------------------------------------------
// Scalar Floating-point Convert To Signed Integer, Rounding To Nearest
// With Ties To Away
//----------------------------------------------------------------------

    fcvtas h12, h13
    fcvtas s12, s13
    fcvtas d21, d14

// CHECK: fcvtas  h12, h13                // encoding: [0xac,0xc9,0x79,0x5e]
// CHECK: fcvtas s12, s13    // encoding: [0xac,0xc9,0x21,0x5e]
// CHECK: fcvtas d21, d14    // encoding: [0xd5,0xc9,0x61,0x5e]

//----------------------------------------------------------------------
// Scalar Floating-point Convert To Unsigned Integer, Rounding To
// Nearest With Ties To Away
//----------------------------------------------------------------------

    fcvtau h12, h13
    fcvtau s12, s13
    fcvtau d21, d14

// CHECK: fcvtau  h12, h13                // encoding: [0xac,0xc9,0x79,0x7e]
// CHECK: fcvtau s12, s13    // encoding: [0xac,0xc9,0x21,0x7e]
// CHECK: fcvtau d21, d14    // encoding: [0xd5,0xc9,0x61,0x7e]

//----------------------------------------------------------------------
// Scalar Floating-point Convert To Signed Integer, Rounding Toward
// Minus Infinity
//----------------------------------------------------------------------

    fcvtms h22, h13
    fcvtms s22, s13
    fcvtms d21, d14

// CHECK: fcvtms  h22, h13                // encoding: [0xb6,0xb9,0x79,0x5e]
// CHECK: fcvtms s22, s13    // encoding: [0xb6,0xb9,0x21,0x5e]
// CHECK: fcvtms d21, d14    // encoding: [0xd5,0xb9,0x61,0x5e]

//----------------------------------------------------------------------
// Scalar Floating-point Convert To Unsigned Integer, Rounding Toward
// Minus Infinity
//----------------------------------------------------------------------

    fcvtmu h12, h13
    fcvtmu s12, s13
    fcvtmu d21, d14

// CHECK: fcvtmu  h12, h13                // encoding: [0xac,0xb9,0x79,0x7e]
// CHECK: fcvtmu s12, s13    // encoding: [0xac,0xb9,0x21,0x7e]
// CHECK: fcvtmu d21, d14    // encoding: [0xd5,0xb9,0x61,0x7e]

//----------------------------------------------------------------------
// Scalar Floating-point Convert To Signed Integer, Rounding To Nearest
// With Ties To Even
//----------------------------------------------------------------------

    fcvtns h22, h13
    fcvtns s22, s13
    fcvtns d21, d14

// CHECK: fcvtns  h22, h13                // encoding: [0xb6,0xa9,0x79,0x5e]
// CHECK: fcvtns s22, s13    // encoding: [0xb6,0xa9,0x21,0x5e]
// CHECK: fcvtns d21, d14    // encoding: [0xd5,0xa9,0x61,0x5e]

//----------------------------------------------------------------------
// Scalar Floating-point Convert To Unsigned Integer, Rounding To
// Nearest With Ties To Even
//----------------------------------------------------------------------

    fcvtnu h12, h13
    fcvtnu s12, s13
    fcvtnu d21, d14

// CHECK: fcvtnu  h12, h13                // encoding: [0xac,0xa9,0x79,0x7e]
// CHECK: fcvtnu s12, s13    // encoding: [0xac,0xa9,0x21,0x7e]
// CHECK: fcvtnu d21, d14    // encoding: [0xd5,0xa9,0x61,0x7e]
        
//----------------------------------------------------------------------
// Scalar Floating-point Convert To Signed Integer, Rounding Toward
// Positive Infinity
//----------------------------------------------------------------------

    fcvtps h22, h13
    fcvtps s22, s13
    fcvtps d21, d14

// CHECK: fcvtps  h22, h13                // encoding: [0xb6,0xa9,0xf9,0x5e]
// CHECK: fcvtps s22, s13    // encoding: [0xb6,0xa9,0xa1,0x5e]
// CHECK: fcvtps d21, d14    // encoding: [0xd5,0xa9,0xe1,0x5e]
        
//----------------------------------------------------------------------
// Scalar Floating-point Convert To Unsigned Integer, Rounding Toward
// Positive Infinity
//----------------------------------------------------------------------

    fcvtpu h12, h13
    fcvtpu s12, s13
    fcvtpu d21, d14

// CHECK: fcvtpu  h12, h13                // encoding: [0xac,0xa9,0xf9,0x7e]
// CHECK: fcvtpu s12, s13    // encoding: [0xac,0xa9,0xa1,0x7e]
// CHECK: fcvtpu d21, d14    // encoding: [0xd5,0xa9,0xe1,0x7e]

//----------------------------------------------------------------------
// Scalar Floating-point Convert To Signed Integer, Rounding Toward Zero
//----------------------------------------------------------------------

    fcvtzs h12, h13
    fcvtzs s12, s13
    fcvtzs d21, d14

// CHECK: fcvtzs  h12, h13                // encoding: [0xac,0xb9,0xf9,0x5e]
// CHECK: fcvtzs s12, s13    // encoding: [0xac,0xb9,0xa1,0x5e]
// CHECK: fcvtzs d21, d14    // encoding: [0xd5,0xb9,0xe1,0x5e]
        
//----------------------------------------------------------------------
// Scalar Floating-point Convert To Unsigned Integer, Rounding Toward 
// Zero
//----------------------------------------------------------------------

    fcvtzu h12, h13
    fcvtzu s12, s13
    fcvtzu d21, d14

// CHECK: fcvtzu  h12, h13                // encoding: [0xac,0xb9,0xf9,0x7e]
// CHECK: fcvtzu s12, s13    // encoding: [0xac,0xb9,0xa1,0x7e]
// CHECK: fcvtzu d21, d14    // encoding: [0xd5,0xb9,0xe1,0x7e]
