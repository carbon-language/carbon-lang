// RUN: llvm-mc -arch=amdgcn -show-encoding %s | FileCheck %s
// RUN: llvm-mc -arch=amdgcn -mcpu=SI -show-encoding %s | FileCheck %s

s_load_dword s1, s[2:3], 1
// CHECK: s_load_dword s1, s[2:3], 0x1 ; encoding: [0x01,0x83,0x00,0xc0]

s_load_dword s1, s[2:3], s4
// CHECK: s_load_dword s1, s[2:3], s4 ; encoding: [0x04,0x82,0x00,0xc0]

s_load_dwordx2 s[2:3], s[2:3], 1
// CHECK: s_load_dwordx2 s[2:3], s[2:3], 0x1 ; encoding: [0x01,0x03,0x41,0xc0]

s_load_dwordx2 s[2:3], s[2:3], s4
// CHECK: s_load_dwordx2 s[2:3], s[2:3], s4 ; encoding: [0x04,0x02,0x41,0xc0]

s_load_dwordx4 s[4:7], s[2:3], 1
// CHECK: s_load_dwordx4 s[4:7], s[2:3], 0x1 ; encoding: [0x01,0x03,0x82,0xc0]

s_load_dwordx4 s[4:7], s[2:3], s4
// CHECK: s_load_dwordx4 s[4:7], s[2:3], s4 ; encoding: [0x04,0x02,0x82,0xc0]

s_load_dwordx8 s[8:15], s[2:3], 1
// CHECK: s_load_dwordx8 s[8:15], s[2:3], 0x1 ; encoding: [0x01,0x03,0xc4,0xc0]

s_load_dwordx8 s[8:15], s[2:3], s4
// CHECK: s_load_dwordx8 s[8:15], s[2:3], s4 ; encoding: [0x04,0x02,0xc4,0xc0]

s_load_dwordx16 s[16:31], s[2:3], 1
// CHECK: s_load_dwordx16 s[16:31], s[2:3], 0x1 ; encoding: [0x01,0x03,0x08,0xc1]

s_load_dwordx16 s[16:31], s[2:3], s4
// CHECK: s_load_dwordx16 s[16:31], s[2:3], s4 ; encoding: [0x04,0x02,0x08,0xc1]
