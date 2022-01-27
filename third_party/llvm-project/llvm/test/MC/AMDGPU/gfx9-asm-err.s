// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx900 %s 2>&1 | FileCheck -check-prefix=GFX9ERR --implicit-check-not=error: %s

v_cvt_f16_u16_e64 v5, 0.5
// GFX9ERR: error: literal operands are not supported

v_cvt_f16_u16_e64 v5, -4.0
// GFX9ERR: error: literal operands are not supported

v_add_u16_e64 v5, v1, 0.5
// GFX9ERR: error: literal operands are not supported

v_add_u16_e64 v5, v1, -4.0
// GFX9ERR: error: literal operands are not supported

v_cvt_f16_i16_e64 v5, 0.5
// GFX9ERR: error: literal operands are not supported

v_cvt_f16_i16_e64 v5, -4.0
// GFX9ERR: error: literal operands are not supported

v_add_u16_e64 v5, 0.5, v2
// GFX9ERR: error: literal operands are not supported

v_add_u16_e64 v5, -4.0, v2
// GFX9ERR: error: literal operands are not supported

v_subrev_u16_e64 v5, v1, 0.5
// GFX9ERR: error: literal operands are not supported

v_subrev_u16_e64 v5, v1, -4.0
// GFX9ERR: error: literal operands are not supported

v_cvt_u32_f64 v5, v[0:1] quad_perm:[0,2,1,1] row_mask:0xf bank_mask:0xf
// GFX9ERR: error: not a valid operand.
