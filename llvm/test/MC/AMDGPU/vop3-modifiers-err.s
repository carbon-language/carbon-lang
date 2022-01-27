// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga %s 2>&1 | FileCheck --implicit-check-not=error: %s

//---------------------------------------------------------------------------//
// VOP3 Modifiers
//---------------------------------------------------------------------------//

// 'neg(1)' cannot be encoded as 32-bit literal while preserving e64 semantics
v_ceil_f64_e32 v[0:1], neg(1)
// CHECK: error: invalid operand for instruction

v_ceil_f32 v0, --1
// CHECK: error: invalid syntax, expected 'neg' modifier

v_ceil_f16 v0, abs(neg(1))
// CHECK: error: failed parsing operand

v_cvt_f16_u16_e64 v5, s1 noXXXclamp
// CHECK: error: invalid operand for instruction
