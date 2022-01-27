// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx900 -show-encoding %s | FileCheck %s --check-prefix=GFX9
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx900 %s 2>&1 | FileCheck %s --check-prefix=NOGFX9 --implicit-check-not=error:

//===----------------------------------------------------------------------===//
// Relocatable expressions cannot be used with SDWA modifiers.
//===----------------------------------------------------------------------===//

v_mov_b32_sdwa v1, sext(u)
// NOGFX9: error: expected an absolute expression

//===----------------------------------------------------------------------===//
// Instructions can use only one literal.
// Relocatable expressions are counted as literals.
//===----------------------------------------------------------------------===//

s_sub_u32 s0, 123, u
// NOGFX9: error: only one literal operand is allowed

s_sub_u32 s0, u, u
// NOGFX9: error: only one literal operand is allowed

s_sub_u32 s0, u, u1
// NOGFX9: error: only one literal operand is allowed

v_bfe_u32 v0, v2, v3, u
// NOGFX9: error: literal operands are not supported

//===----------------------------------------------------------------------===//
// Constant expressions may be used with 'sext' modifier
//===----------------------------------------------------------------------===//

i1=1

v_mov_b32_sdwa v1, sext(i1-2)
// GFX9: v_mov_b32_sdwa v1, sext(-1) dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD ; encoding: [0xf9,0x02,0x02,0x7e,0xc1,0x16,0x8e,0x00]

v_mov_b32_sdwa v1, sext(-2+i1)
// GFX9: v_mov_b32_sdwa v1, sext(-1) dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD ; encoding: [0xf9,0x02,0x02,0x7e,0xc1,0x16,0x8e,0x00]

//===----------------------------------------------------------------------===//
// Constant expressions may be used with op_sel* and neg_* modifiers.
//===----------------------------------------------------------------------===//

v_pk_add_u16 v1, v2, v3 op_sel:[2-i1,i1-1]
// GFX9: v_pk_add_u16 v1, v2, v3 op_sel:[1,0] ; encoding: [0x01,0x48,0x8a,0xd3,0x02,0x07,0x02,0x18]

v_pk_add_u16 v1, v2, v3 neg_lo:[2-i1,i1-1]
// GFX9: v_pk_add_u16 v1, v2, v3 neg_lo:[1,0] ; encoding: [0x01,0x40,0x8a,0xd3,0x02,0x07,0x02,0x38]
