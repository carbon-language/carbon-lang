// PR19320
// RUN: llvm-mc -triple=armv7-linux-gnueabi -show-encoding < %s | FileCheck %s
.text

// CHECK: ldrd	r12, sp, [r0, #32]      @ encoding: [0xd0,0xc2,0xc0,0xe1]
        ldrd    r12, [r0, #32]

// CHECK: strd	r12, sp, [r0, #32]      @ encoding: [0xf0,0xc2,0xc0,0xe1]
        strd    r12, [r0, #32]
