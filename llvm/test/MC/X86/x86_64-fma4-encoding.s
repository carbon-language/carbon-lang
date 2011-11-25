// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: vfmaddsd  (%rcx), %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0xf9,0x6b,0x01,0x10]
          vfmaddsd  (%rcx), %xmm1, %xmm0, %xmm0

// CHECK: vfmaddsd   %xmm1, (%rcx), %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0x79,0x6b,0x01,0x10]
          vfmaddsd   %xmm1, (%rcx),%xmm0, %xmm0

// CHECK: vfmaddsd   %xmm2, %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0xf9,0x6b,0xc2,0x10]
          vfmaddsd   %xmm2, %xmm1, %xmm0, %xmm0
