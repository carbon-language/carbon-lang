// RUN: llvm-mc -triple aarch64-none-linux-gnu -mattr=+neon -show-encoding < %s | FileCheck %s

//------------------------------------------------------------------------------
// Scalar Integer Saturating Add (Signed)
//------------------------------------------------------------------------------
         sqadd b0, b1, b2
         sqadd h10, h11, h12
         sqadd s20, s21, s2
         sqadd d17, d31, d8

// CHECK: sqadd b0, b1, b2        // encoding: [0x20,0x0c,0x22,0x5e]
// CHECK: sqadd h10, h11, h12     // encoding: [0x6a,0x0d,0x6c,0x5e]
// CHECK: sqadd s20, s21, s2      // encoding: [0xb4,0x0e,0xa2,0x5e]
// CHECK: sqadd d17, d31, d8      // encoding: [0xf1,0x0f,0xe8,0x5e]

//------------------------------------------------------------------------------
// Scalar Integer Saturating Add (Unsigned)
//------------------------------------------------------------------------------
         uqadd b0, b1, b2
         uqadd h10, h11, h12
         uqadd s20, s21, s2
         uqadd d17, d31, d8

// CHECK: uqadd b0, b1, b2        // encoding: [0x20,0x0c,0x22,0x7e]
// CHECK: uqadd h10, h11, h12     // encoding: [0x6a,0x0d,0x6c,0x7e]
// CHECK: uqadd s20, s21, s2      // encoding: [0xb4,0x0e,0xa2,0x7e]
// CHECK: uqadd d17, d31, d8      // encoding: [0xf1,0x0f,0xe8,0x7e]

//------------------------------------------------------------------------------
// Scalar Integer Saturating Sub (Signed)
//------------------------------------------------------------------------------
         sqsub b0, b1, b2
         sqsub h10, h11, h12
         sqsub s20, s21, s2
         sqsub d17, d31, d8

// CHECK: sqsub b0, b1, b2        // encoding: [0x20,0x2c,0x22,0x5e]
// CHECK: sqsub h10, h11, h12     // encoding: [0x6a,0x2d,0x6c,0x5e]
// CHECK: sqsub s20, s21, s2      // encoding: [0xb4,0x2e,0xa2,0x5e]
// CHECK: sqsub d17, d31, d8      // encoding: [0xf1,0x2f,0xe8,0x5e]

//------------------------------------------------------------------------------
// Scalar Integer Saturating Sub (Unsigned)
//------------------------------------------------------------------------------
         uqsub b0, b1, b2
         uqsub h10, h11, h12
         uqsub s20, s21, s2
         uqsub d17, d31, d8

// CHECK: uqsub b0, b1, b2        // encoding: [0x20,0x2c,0x22,0x7e]
// CHECK: uqsub h10, h11, h12     // encoding: [0x6a,0x2d,0x6c,0x7e]
// CHECK: uqsub s20, s21, s2      // encoding: [0xb4,0x2e,0xa2,0x7e]
// CHECK: uqsub d17, d31, d8      // encoding: [0xf1,0x2f,0xe8,0x7e]

