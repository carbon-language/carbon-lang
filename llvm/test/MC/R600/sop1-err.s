// RUN: not llvm-mc -arch=amdgcn %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=SI %s 2>&1 | FileCheck %s

s_mov_b32 v1, s2
// CHECK: error: invalid operand for instruction

s_mov_b32 s1, v0
// CHECK: error: invalid operand for instruction

s_mov_b32 s[1:2], s0
// CHECK: error: invalid operand for instruction

s_mov_b32 s0, s[1:2]
// CHECK: error: invalid operand for instruction

s_mov_b32 s220, s0
// CHECK: error: invalid operand for instruction

s_mov_b32 s0, s220
// CHECK: error: invalid operand for instruction

s_mov_b64 s1, s[0:1]
// CHECK: error: invalid operand for instruction

s_mov_b64 s[0:1], s1
// CHECK: error: invalid operand for instruction

// Immediate greater than 32-bits
s_mov_b32 s1, 0xfffffffff
// CHECK: error: invalid immediate: only 32-bit values are legal

// Immediate greater than 32-bits
s_mov_b64 s[0:1], 0xfffffffff
// CHECK: error: invalid immediate: only 32-bit values are legal

// Out of range register
s_mov_b32 s
