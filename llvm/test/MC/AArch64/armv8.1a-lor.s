// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.1a < %s | FileCheck %s


//------------------------------------------------------------------------------
// Load acquire / store release
//------------------------------------------------------------------------------
        ldlarb w0,[x1]
        ldlarh w0,[x1]
        ldlar  w0,[x1]
        ldlar  x0,[x1]
// CHECK:   ldlarb w0, [x1]   // encoding: [0x20,0x7c,0xdf,0x08]
// CHECK:   ldlarh w0, [x1]   // encoding: [0x20,0x7c,0xdf,0x48]
// CHECK:   ldlar  w0, [x1]   // encoding: [0x20,0x7c,0xdf,0x88]
// CHECK:   ldlar  x0, [x1]   // encoding: [0x20,0x7c,0xdf,0xc8]
        stllrb w0,[x1]
        stllrh w0,[x1]
        stllr  w0,[x1]
        stllr  x0,[x1]
// CHECK:   stllrb w0, [x1]   // encoding: [0x20,0x7c,0x9f,0x08]
// CHECK:   stllrh w0, [x1]   // encoding: [0x20,0x7c,0x9f,0x48]
// CHECK:   stllr  w0, [x1]   // encoding: [0x20,0x7c,0x9f,0x88]
// CHECK:   stllr  x0, [x1]   // encoding: [0x20,0x7c,0x9f,0xc8]

        msr    LORSA_EL1, x0
        msr    LOREA_EL1, x0
        msr    LORN_EL1, x0
        msr    LORC_EL1, x0
        mrs    x0, LORID_EL1
// CHECK:   msr    LORSA_EL1, x0 // encoding: [0x00,0xa4,0x18,0xd5]
// CHECK:   msr    LOREA_EL1, x0 // encoding: [0x20,0xa4,0x18,0xd5]
// CHECK:   msr    LORN_EL1, x0  // encoding: [0x40,0xa4,0x18,0xd5]
// CHECK:   msr    LORC_EL1, x0  // encoding: [0x60,0xa4,0x18,0xd5]
// CHECK:   mrs    x0, LORID_EL1 // encoding: [0xe0,0xa4,0x38,0xd5]
