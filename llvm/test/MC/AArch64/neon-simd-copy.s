// RUN: llvm-mc -triple=aarch64 -mattr=+neon -show-encoding < %s | FileCheck %s

// Check that the assembler can handle the documented syntax for AArch64


//------------------------------------------------------------------------------
// Insert element (vector, from main)
//------------------------------------------------------------------------------
         ins v2.b[2], w1
         ins v7.h[7], w14
         ins v20.s[0], w30
         ins v1.d[1], x7
         
// CHECK: ins	v2.b[2], w1           // encoding: [0x22,0x1c,0x05,0x4e]
// CHECK: ins	v7.h[7], w14          // encoding: [0xc7,0x1d,0x1e,0x4e]
// CHECK: ins	v20.s[0], w30         // encoding: [0xd4,0x1f,0x04,0x4e]
// CHECK: ins	v1.d[1], x7           // encoding: [0xe1,0x1c,0x18,0x4e]


//------------------------------------------------------------------------------
// Signed integer move (main, from element)
//------------------------------------------------------------------------------
         smov w1, v0.b[15]
         smov w14, v6.h[4]
         smov x1, v0.b[15]
         smov x14, v6.h[4]
         smov x20, v9.s[2]

// CHECK: smov	w1, v0.b[15]          // encoding: [0x01,0x2c,0x1f,0x0e]
// CHECK: smov	w14, v6.h[4]          // encoding: [0xce,0x2c,0x12,0x0e]
// CHECK: smov	x1, v0.b[15]          // encoding: [0x01,0x2c,0x1f,0x4e]
// CHECK: smov	x14, v6.h[4]          // encoding: [0xce,0x2c,0x12,0x4e]
// CHECK: smov	x20, v9.s[2]          // encoding: [0x34,0x2d,0x14,0x4e]         


//------------------------------------------------------------------------------
// Unsigned integer move (main, from element)
//------------------------------------------------------------------------------
         umov w1, v0.b[15]
         umov w14, v6.h[4]
         umov w20, v9.s[2]
         umov x7, v18.d[1]

// CHECK: umov	w1, v0.b[15]          // encoding: [0x01,0x3c,0x1f,0x0e]
// CHECK: umov	w14, v6.h[4]          // encoding: [0xce,0x3c,0x12,0x0e]
// CHECK: umov	w20, v9.s[2]          // encoding: [0x34,0x3d,0x14,0x0e]
// CHECK: umov	x7, v18.d[1]          // encoding: [0x47,0x3e,0x18,0x4e]

//------------------------------------------------------------------------------
// Insert element (vector, from element)
//------------------------------------------------------------------------------

         Ins v1.b[14], v3.b[6]
         Ins v6.h[7], v7.h[5]
         Ins v15.s[3], v22.s[2]
         Ins v0.d[0], v4.d[1]

// CHECK: ins	v1.b[14], v3.b[6]       // encoding: [0x61,0x34,0x1d,0x6e]
// CHECK: ins	v6.h[7], v7.h[5]        // encoding: [0xe6,0x54,0x1e,0x6e]
// CHECK: ins	v15.s[3], v22.s[2]      // encoding: [0xcf,0x5e,0x1c,0x6e]
// CHECK: ins	v0.d[0], v4.d[1]        // encoding: [0x80,0x44,0x08,0x6e]










