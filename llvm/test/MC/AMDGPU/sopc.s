// RUN: llvm-mc -arch=amdgcn -show-encoding %s | FileCheck %s
// RUN: llvm-mc -arch=amdgcn -show-encoding %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

s_cmp_eq_i32 s1, s2
// CHECK: s_cmp_eq_i32 s1, s2 ; encoding: [0x01,0x02,0x00,0xbf]
