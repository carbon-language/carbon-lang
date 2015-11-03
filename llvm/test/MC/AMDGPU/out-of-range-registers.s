// RUN: not llvm-mc -arch=amdgcn -mcpu=tahiti %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga %s 2>&1 | FileCheck %s

s_add_i32 s104, s0, s1
// CHECK: error: invalid operand for instruction

s_add_i32 s105, s0, s1
// CHECK: error: invalid operand for instruction

v_add_i32 v256, v0, v1
// CHECK: error: invalid operand for instruction

v_add_i32 v257, v0, v1
// CHECK: error: invalid operand for instruction
