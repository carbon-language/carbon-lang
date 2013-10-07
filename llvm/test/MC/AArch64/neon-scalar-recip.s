// RUN: llvm-mc -triple aarch64-none-linux-gnu -mattr=+neon -show-encoding < %s | FileCheck %s

// Check that the assembler can handle the documented syntax for AArch64

//----------------------------------------------------------------------
// Floating-point Reciprocal Step
//----------------------------------------------------------------------

    frecps s21, s16, s13
    frecps d22, d30, d21

// CHECK: frecps s21, s16, s13   // encoding: [0x15,0xfe,0x2d,0x5e]
// CHECK: frecps d22, d30, d21   // encoding: [0xd6,0xff,0x75,0x5e]

//----------------------------------------------------------------------
// Floating-point Reciprocal Square Root Step
//----------------------------------------------------------------------

    frsqrts s21, s5, s12
    frsqrts d8, d22, d18

// CHECK: frsqrts s21, s5, s12   // encoding: [0xb5,0xfc,0xac,0x5e]
// CHECK: frsqrts d8, d22, d18   // encoding: [0xc8,0xfe,0xf2,0x5e]
