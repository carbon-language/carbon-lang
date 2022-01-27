// RUN: llvm-mc -triple aarch64-none-linux-gnu -mattr=+neon,+fullfp16 -show-encoding < %s | FileCheck %s

// Check that the assembler can handle the documented syntax for AArch64

//----------------------------------------------------------------------
// Vector Reciprocal Square Root Step (Floating Point)
//----------------------------------------------------------------------
         frsqrts v0.4h, v31.4h, v16.4h
         frsqrts v4.8h, v7.8h, v15.8h
         frsqrts v0.2s, v31.2s, v16.2s
         frsqrts v4.4s, v7.4s, v15.4s
         frsqrts v29.2d, v2.2d, v5.2d

// CHECK: frsqrts v0.4h, v31.4h, v16.4h   // encoding: [0xe0,0x3f,0xd0,0x0e]
// CHECK: frsqrts v4.8h, v7.8h, v15.8h    // encoding: [0xe4,0x3c,0xcf,0x4e]
// CHECK: frsqrts v0.2s, v31.2s, v16.2s // encoding: [0xe0,0xff,0xb0,0x0e]
// CHECK: frsqrts v4.4s, v7.4s, v15.4s  // encoding: [0xe4,0xfc,0xaf,0x4e]
// CHECK: frsqrts v29.2d, v2.2d, v5.2d  // encoding: [0x5d,0xfc,0xe5,0x4e]

//----------------------------------------------------------------------
// Vector Reciprocal Step (Floating Point)
//----------------------------------------------------------------------
         frecps v3.4h, v8.4h, v12.4h
         frecps v31.8h, v29.8h, v28.8h
         frecps v31.4s, v29.4s, v28.4s
         frecps v3.2s, v8.2s, v12.2s
         frecps v17.2d, v15.2d, v13.2d

// CHECK: frecps  v3.4h, v8.4h, v12.4h    // encoding: [0x03,0x3d,0x4c,0x0e]
// CHECK: frecps  v31.8h, v29.8h, v28.8h  // encoding: [0xbf,0x3f,0x5c,0x4e]
// CHECK: frecps v31.4s, v29.4s, v28.4s  // encoding: [0xbf,0xff,0x3c,0x4e]
// CHECK: frecps v3.2s, v8.2s, v12.2s    // encoding: [0x03,0xfd,0x2c,0x0e]
// CHECK: frecps v17.2d, v15.2d, v13.2d  // encoding: [0xf1,0xfd,0x6d,0x4e]


