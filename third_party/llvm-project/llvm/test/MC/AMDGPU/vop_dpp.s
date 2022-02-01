// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga -show-encoding %s | FileCheck %s --check-prefixes=VI,VI9
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx900 -show-encoding %s | FileCheck %s --check-prefixes=GFX9,VI9

// RUN: not llvm-mc -arch=amdgcn %s 2>&1 | FileCheck %s --check-prefixes=NOSI,NOSICI --implicit-check-not=error:
// RUN: not llvm-mc -arch=amdgcn -mcpu=tahiti %s 2>&1 | FileCheck %s --check-prefixes=NOSI,NOSICI --implicit-check-not=error:
// RUN: not llvm-mc -arch=amdgcn -mcpu=bonaire %s 2>&1 | FileCheck %s --check-prefixes=NOSICI,NOCI --implicit-check-not=error:
// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga %s 2>&1 | FileCheck %s --check-prefix=NOVI --implicit-check-not=error:
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx900 %s 2>&1 | FileCheck %s --check-prefix=NOGFX9 --implicit-check-not=error:

//===----------------------------------------------------------------------===//
// Check dpp_ctrl values
//===----------------------------------------------------------------------===//

// NOSICI: error: not a valid operand.
// VI9: v_mov_b32_dpp v0, v0 quad_perm:[0,2,1,1] row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x02,0x00,0x7e,0x00,0x58,0x00,0xff]
v_mov_b32 v0, v0 quad_perm:[0,2,1,1]

// NOSICI: error: not a valid operand.
// VI9: v_mov_b32_dpp v0, v0 row_shl:1 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x02,0x00,0x7e,0x00,0x01,0x01,0xff]
v_mov_b32 v0, v0 row_shl:1

// NOSICI: error: not a valid operand.
// VI9: v_mov_b32_dpp v0, v0 row_shr:15 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x02,0x00,0x7e,0x00,0x1f,0x01,0xff]
v_mov_b32 v0, v0 row_shr:0xf

// NOSICI: error: not a valid operand.
// VI9: v_mov_b32_dpp v0, v0 row_ror:12 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x02,0x00,0x7e,0x00,0x2c,0x01,0xff]
v_mov_b32 v0, v0 row_ror:0xc

// NOSICI: error: not a valid operand.
// VI9: v_mov_b32_dpp v0, v0 wave_shl:1 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x02,0x00,0x7e,0x00,0x30,0x01,0xff]
v_mov_b32 v0, v0 wave_shl:1

// NOSICI: error: not a valid operand.
// VI9: v_mov_b32_dpp v0, v0 wave_rol:1 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x02,0x00,0x7e,0x00,0x34,0x01,0xff]
v_mov_b32 v0, v0 wave_rol:1

// NOSICI: error: not a valid operand.
// VI9: v_mov_b32_dpp v0, v0 wave_shr:1 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x02,0x00,0x7e,0x00,0x38,0x01,0xff]
v_mov_b32 v0, v0 wave_shr:1

// NOSICI: error: not a valid operand.
// VI9: v_mov_b32_dpp v0, v0 wave_ror:1 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x02,0x00,0x7e,0x00,0x3c,0x01,0xff]
v_mov_b32 v0, v0 wave_ror:1

// NOSICI: error: invalid operand for instruction
// VI9: v_mov_b32_dpp v0, v0 row_mirror row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x02,0x00,0x7e,0x00,0x40,0x01,0xff]
v_mov_b32 v0, v0 row_mirror

// NOSICI: error: invalid operand for instruction
// VI9: v_mov_b32_dpp v0, v0 row_half_mirror row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x02,0x00,0x7e,0x00,0x41,0x01,0xff]
v_mov_b32 v0, v0 row_half_mirror

// NOSICI: error: not a valid operand.
// VI9: v_mov_b32_dpp v0, v0 row_bcast:15 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x02,0x00,0x7e,0x00,0x42,0x01,0xff]
v_mov_b32 v0, v0 row_bcast:15

// NOSICI: error: not a valid operand.
// VI9: v_mov_b32_dpp v0, v0 row_bcast:31 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x02,0x00,0x7e,0x00,0x43,0x01,0xff]
v_mov_b32 v0, v0 row_bcast:31

//===----------------------------------------------------------------------===//
// Check bound_control modifier.
// Both bound_ctrl:0 and bound_ctrl:1 are legal and encoded as 1.
// See bug 35397 for details.
//===----------------------------------------------------------------------===//

// NOSICI: error: not a valid operand.
// VI9: v_mov_b32_dpp v0, v0 quad_perm:[1,3,0,1] row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x02,0x00,0x7e,0x00,0x4d,0x08,0xa1]
v_mov_b32 v0, v0 quad_perm:[1,3,0,1] row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_mov_b32_dpp v0, v0 quad_perm:[1,3,0,1] row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x02,0x00,0x7e,0x00,0x4d,0x08,0xa1]
v_mov_b32 v0, v0 quad_perm:[1,3,0,1] row_mask:0xa bank_mask:0x1 bound_ctrl:1

// NOSICI: error: not a valid operand.
// NOGFX9: error: invalid bound_ctrl value.
// NOVI: error: invalid bound_ctrl value.
v_mov_b32 v0, v0 quad_perm:[1,3,0,1] row_mask:0xa bank_mask:0x1 bound_ctrl:-1

// NOSICI: error: not a valid operand.
// NOGFX9: error: invalid bound_ctrl value.
// NOVI: error: invalid bound_ctrl value.
v_mov_b32 v0, v0 quad_perm:[1,3,0,1] row_mask:0xa bank_mask:0x1 bound_ctrl:2

//===----------------------------------------------------------------------===//
// Check optional fields
//===----------------------------------------------------------------------===//

// NOSICI: error: not a valid operand.
// VI9: v_mov_b32_dpp v0, v0 quad_perm:[1,3,0,1] row_mask:0xa bank_mask:0xf ; encoding: [0xfa,0x02,0x00,0x7e,0x00,0x4d,0x00,0xaf]
v_mov_b32 v0, v0 quad_perm:[1,3,0,1] row_mask:0xa

// NOSICI: error: not a valid operand.
// VI9: v_mov_b32_dpp v0, v0 quad_perm:[1,3,0,1] row_mask:0xf bank_mask:0x1 ; encoding: [0xfa,0x02,0x00,0x7e,0x00,0x4d,0x00,0xf1]
v_mov_b32 v0, v0 quad_perm:[1,3,0,1] bank_mask:0x1

// NOSICI: error: not a valid operand.
// VI9: v_mov_b32_dpp v0, v0 quad_perm:[1,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1 ; encoding: [0xfa,0x02,0x00,0x7e,0x00,0x4d,0x08,0xff]
v_mov_b32 v0, v0 quad_perm:[1,3,0,1] bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_mov_b32_dpp v0, v0 quad_perm:[1,3,0,1] row_mask:0xa bank_mask:0x1 ; encoding: [0xfa,0x02,0x00,0x7e,0x00,0x4d,0x00,0xa1]
v_mov_b32 v0, v0 quad_perm:[1,3,0,1] row_mask:0xa bank_mask:0x1

// NOSICI: error: not a valid operand.
// VI9: v_mov_b32_dpp v0, v0 quad_perm:[1,3,0,1] row_mask:0xa bank_mask:0xf bound_ctrl:1 ; encoding: [0xfa,0x02,0x00,0x7e,0x00,0x4d,0x08,0xaf]
v_mov_b32 v0, v0 quad_perm:[1,3,0,1] row_mask:0xa bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_mov_b32_dpp v0, v0 quad_perm:[1,3,0,1] row_mask:0xf bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x02,0x00,0x7e,0x00,0x4d,0x08,0xf1]
v_mov_b32 v0, v0 quad_perm:[1,3,0,1] bank_mask:0x1 bound_ctrl:0

//===----------------------------------------------------------------------===//
// Check modifiers
//===----------------------------------------------------------------------===//

// NOSICI: error: not a valid operand.
// VI9: v_add_f32_dpp v0, -v0, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x00,0x00,0x02,0x00,0x01,0x19,0xa1]
v_add_f32 v0, -v0, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_add_f32_dpp v0, v0, |v0| row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x00,0x00,0x02,0x00,0x01,0x89,0xa1]
v_add_f32 v0, v0, |v0| row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_add_f32_dpp v0, -v0, |v0| row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x00,0x00,0x02,0x00,0x01,0x99,0xa1]
v_add_f32 v0, -v0, |v0| row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_add_f32_dpp v0, |v0|, -v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x00,0x00,0x02,0x00,0x01,0x69,0xa1]
v_add_f32 v0, |v0|, -v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

//===----------------------------------------------------------------------===//
// Check VOP1 opcodes
//===----------------------------------------------------------------------===//

// NOSICI: error: not a valid operand.
// GCN: v_nop row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0 ; encoding: [0xfa,0x00,0x00,0x7e,0x00,0x01,0x09,0xa1]
v_nop row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_cvt_u32_f32_dpp v0, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x0e,0x00,0x7e,0x00,0x01,0x09,0xa1]
v_cvt_u32_f32 v0, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_fract_f32_dpp v0, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x36,0x00,0x7e,0x00,0x01,0x09,0xa1]
v_fract_f32 v0, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_sin_f32_dpp v0, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x52,0x00,0x7e,0x00,0x01,0x09,0xa1]
v_sin_f32 v0, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_mov_b32_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x02,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_mov_b32 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_cvt_f32_i32_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x0a,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_cvt_f32_i32 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_cvt_f32_u32_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x0c,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_cvt_f32_u32 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_cvt_i32_f32_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x10,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_cvt_i32_f32 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_cvt_f16_f32_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x14,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_cvt_f16_f32 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_cvt_f32_f16_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x16,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_cvt_f32_f16 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_cvt_rpi_i32_f32_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x18,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_cvt_rpi_i32_f32 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_cvt_flr_i32_f32_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x1a,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_cvt_flr_i32_f32 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_cvt_off_f32_i4_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x1c,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_cvt_off_f32_i4 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_cvt_f32_ubyte0_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x22,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_cvt_f32_ubyte0 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_cvt_f32_ubyte1_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x24,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_cvt_f32_ubyte1 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_cvt_f32_ubyte2_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x26,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_cvt_f32_ubyte2 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_cvt_f32_ubyte3_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x28,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_cvt_f32_ubyte3 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_trunc_f32_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x38,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_trunc_f32 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_ceil_f32_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x3a,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_ceil_f32 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_rndne_f32_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x3c,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_rndne_f32 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_floor_f32_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x3e,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_floor_f32 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_exp_f32_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x40,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_exp_f32 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_log_f32_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x42,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_log_f32 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_rcp_f32_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x44,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_rcp_f32 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_rcp_iflag_f32_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x46,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_rcp_iflag_f32 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_rsq_f32_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x48,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_rsq_f32 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_sqrt_f32_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x4e,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_sqrt_f32 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_cos_f32_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x54,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_cos_f32 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_not_b32_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x56,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_not_b32 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_bfrev_b32_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x58,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_bfrev_b32 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_ffbh_u32_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x5a,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_ffbh_u32 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_ffbl_b32_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x5c,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_ffbl_b32 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_ffbh_i32_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x5e,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_ffbh_i32 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_frexp_exp_i32_f32_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x66,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_frexp_exp_i32_f32 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_frexp_mant_f32_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x68,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_frexp_mant_f32 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// VI9: v_log_legacy_f32_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x98,0x02,0x7e,0x00,0x01,0x09,0xa1]
// NOSI: error: instruction not supported on this GPU
// NOCI: error: not a valid operand.
v_log_legacy_f32 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// VI9: v_exp_legacy_f32_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x96,0x02,0x7e,0x00,0x01,0x09,0xa1]
// NOSI: error: instruction not supported on this GPU
// NOCI: error: not a valid operand.
v_exp_legacy_f32 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: instruction not supported on this GPU
// VI9: v_cvt_f16_u16_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x72,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_cvt_f16_u16 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: instruction not supported on this GPU
// VI9: v_cvt_f16_i16_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x74,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_cvt_f16_i16 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: instruction not supported on this GPU
// VI9: v_cvt_u16_f16_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x76,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_cvt_u16_f16 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: instruction not supported on this GPU
// VI9: v_cvt_i16_f16_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x78,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_cvt_i16_f16 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: instruction not supported on this GPU
// VI9: v_rcp_f16_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x7a,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_rcp_f16 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: instruction not supported on this GPU
// VI9: v_sqrt_f16_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x7c,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_sqrt_f16 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: instruction not supported on this GPU
// VI9: v_rsq_f16_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x7e,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_rsq_f16 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: instruction not supported on this GPU
// VI9: v_log_f16_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x80,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_log_f16 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: instruction not supported on this GPU
// VI9: v_exp_f16_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x82,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_exp_f16 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: instruction not supported on this GPU
// VI9: v_frexp_mant_f16_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x84,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_frexp_mant_f16 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: instruction not supported on this GPU
// VI9: v_frexp_exp_i16_f16_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x86,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_frexp_exp_i16_f16 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: instruction not supported on this GPU
// VI9: v_floor_f16_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x88,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_floor_f16 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: instruction not supported on this GPU
// VI9: v_ceil_f16_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x8a,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_ceil_f16 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: instruction not supported on this GPU
// VI9: v_trunc_f16_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x8c,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_trunc_f16 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: instruction not supported on this GPU
// VI9: v_rndne_f16_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x8e,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_rndne_f16 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: instruction not supported on this GPU
// VI9: v_fract_f16_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x90,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_fract_f16 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: instruction not supported on this GPU
// VI9: v_sin_f16_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x92,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_sin_f16 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: instruction not supported on this GPU
// VI9: v_cos_f16_dpp v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x94,0x02,0x7e,0x00,0x01,0x09,0xa1]
v_cos_f16 v1, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// GFX9:   v_cvt_norm_i16_f16_dpp v5, |v1| quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0 ; encoding: [0xfa,0x9a,0x0a,0x7e,0x01,0xe4,0x20,0x00]
// NOSICI: error: instruction not supported on this GPU
// NOVI:   error: instruction not supported on this GPU
v_cvt_norm_i16_f16_dpp v5, |v1| quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0

// GFX9:   v_cvt_norm_u16_f16_dpp v5, v1 quad_perm:[3,2,1,0] row_mask:0x0 bank_mask:0x0 ; encoding: [0xfa,0x9c,0x0a,0x7e,0x01,0x1b,0x00,0x00]
// NOSICI: error: instruction not supported on this GPU
// NOVI:   error: instruction not supported on this GPU
v_cvt_norm_u16_f16_dpp v5, v1 quad_perm:[3,2,1,0] row_mask:0x0 bank_mask:0x0

// GFX9:   v_sat_pk_u8_i16_dpp v5, v1 row_ror:15 row_mask:0x0 bank_mask:0x0 ; encoding: [0xfa,0x9e,0x0a,0x7e,0x01,0x2f,0x01,0x00]
// NOSICI: error: instruction not supported on this GPU
// NOVI:   error: instruction not supported on this GPU
v_sat_pk_u8_i16_dpp v5, v1 row_ror:15 row_mask:0x0 bank_mask:0x0

// NOSICI: error: instruction not supported on this GPU
// NOVI: error: instruction not supported on this GPU
// GFX9: v_screen_partition_4se_b32_dpp v5, v1 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0 bound_ctrl:1 ; encoding: [0xfa,0x6e,0x0a,0x7e,0x01,0xe4,0x08,0x00]
v_screen_partition_4se_b32_dpp v5, v1 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0 bound_ctrl:0

//===----------------------------------------------------------------------===//
// Check VOP2 opcodes
//===----------------------------------------------------------------------===//
// ToDo: VOP2bInst instructions: v_add_u32, v_sub_u32 ... (vcc and ApplyMnemonic in AsmMatcherEmitter.cpp)

// NOSICI: error: not a valid operand.
// VI9: v_mac_f32_dpp v0, v0, v0  row_shl:1 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x00,0x00,0x2c,0x00,0x01,0x01,0xff]
v_mac_f32 v0, v0, v0 row_shl:1

// NOSICI: error: not a valid operand.
// VI9: v_mac_f32_dpp v0, v0, v0  row_shr:15 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x00,0x00,0x2c,0x00,0x1f,0x01,0xff]
v_mac_f32 v0, v0, v0 row_shr:0xf

// NOSICI: error: not a valid operand.
// VI9: v_mac_f32_dpp v0, v0, v0 quad_perm:[1,3,0,1] row_mask:0xa bank_mask:0xf bound_ctrl:1 ; encoding: [0xfa,0x00,0x00,0x2c,0x00,0x4d,0x08,0xaf]
v_mac_f32 v0, v0, v0 quad_perm:[1,3,0,1] row_mask:0xa bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_add_f32_dpp v0, v0, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x00,0x00,0x02,0x00,0x01,0x09,0xa1]
v_add_f32 v0, v0, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_min_f32_dpp v0, v0, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x00,0x00,0x14,0x00,0x01,0x09,0xa1]
v_min_f32 v0, v0, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_and_b32_dpp v0, v0, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x00,0x00,0x26,0x00,0x01,0x09,0xa1]
v_and_b32 v0, v0, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_mul_i32_i24_dpp v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x0c,0x02,0x01,0x09,0xa1]
v_mul_i32_i24 v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_sub_f32_dpp v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x04,0x02,0x01,0x09,0xa1]
v_sub_f32 v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_subrev_f32_dpp v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x06,0x02,0x01,0x09,0xa1]
v_subrev_f32 v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_mul_f32_dpp v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x0a,0x02,0x01,0x09,0xa1]
v_mul_f32 v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_mul_hi_i32_i24_dpp v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x0e,0x02,0x01,0x09,0xa1]
v_mul_hi_i32_i24 v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_mul_u32_u24_dpp v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x10,0x02,0x01,0x09,0xa1]
v_mul_u32_u24 v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_mul_hi_u32_u24_dpp v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x12,0x02,0x01,0x09,0xa1]
v_mul_hi_u32_u24 v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_max_f32_dpp v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x16,0x02,0x01,0x09,0xa1]
v_max_f32 v1, v2 v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_min_i32_dpp v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x18,0x02,0x01,0x09,0xa1]
v_min_i32 v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_max_i32_dpp v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x1a,0x02,0x01,0x09,0xa1]
v_max_i32 v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_min_u32_dpp v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x1c,0x02,0x01,0x09,0xa1]
v_min_u32 v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_max_u32_dpp v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x1e,0x02,0x01,0x09,0xa1]
v_max_u32 v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_lshrrev_b32_dpp v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x20,0x02,0x01,0x09,0xa1]
v_lshrrev_b32 v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_ashrrev_i32_dpp v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x22,0x02,0x01,0x09,0xa1]
v_ashrrev_i32 v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_lshlrev_b32_dpp v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x24,0x02,0x01,0x09,0xa1]
v_lshlrev_b32 v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_or_b32_dpp v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x28,0x02,0x01,0x09,0xa1]
v_or_b32 v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9: v_xor_b32_dpp v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x2a,0x02,0x01,0x09,0xa1]
v_xor_b32 v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: instruction not supported on this GPU
// VI9: v_add_f16_dpp v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x3e,0x02,0x01,0x09,0xa1]
v_add_f16 v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: instruction not supported on this GPU
// VI9: v_sub_f16_dpp v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x40,0x02,0x01,0x09,0xa1]
v_sub_f16 v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: instruction not supported on this GPU
// VI9: v_subrev_f16_dpp v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x42,0x02,0x01,0x09,0xa1]
v_subrev_f16 v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: instruction not supported on this GPU
// VI9: v_mul_f16_dpp v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x44,0x02,0x01,0x09,0xa1]
v_mul_f16 v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: instruction not supported on this GPU
// VI9: v_mac_f16_dpp v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x46,0x02,0x01,0x09,0xa1]
v_mac_f16 v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: instruction not supported on this GPU
// VI9: v_add_u16_dpp v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x4c,0x02,0x01,0x09,0xa1]
v_add_u16 v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: instruction not supported on this GPU
// VI9: v_sub_u16_dpp v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x4e,0x02,0x01,0x09,0xa1]
v_sub_u16 v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: instruction not supported on this GPU
// VI9: v_subrev_u16_dpp v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x50,0x02,0x01,0x09,0xa1]
v_subrev_u16 v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: instruction not supported on this GPU
// VI9: v_mul_lo_u16_dpp v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x52,0x02,0x01,0x09,0xa1]
v_mul_lo_u16 v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: instruction not supported on this GPU
// VI9: v_lshlrev_b16_dpp v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x54,0x02,0x01,0x09,0xa1]
v_lshlrev_b16 v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: instruction not supported on this GPU
// VI9: v_lshrrev_b16_dpp v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x56,0x02,0x01,0x09,0xa1]
v_lshrrev_b16 v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: instruction not supported on this GPU
// VI9: v_ashrrev_i16_dpp v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x58,0x02,0x01,0x09,0xa1]
v_ashrrev_i16 v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: instruction not supported on this GPU
// VI9: v_max_f16_dpp v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x5a,0x02,0x01,0x09,0xa1]
v_max_f16 v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: instruction not supported on this GPU
// VI9: v_min_f16_dpp v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x5c,0x02,0x01,0x09,0xa1]
v_min_f16 v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: instruction not supported on this GPU
// VI9: v_max_u16_dpp v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x5e,0x02,0x01,0x09,0xa1]
v_max_u16 v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: instruction not supported on this GPU
// VI9: v_max_i16_dpp v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x60,0x02,0x01,0x09,0xa1]
v_max_i16 v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: instruction not supported on this GPU
// VI9: v_min_u16_dpp v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x62,0x02,0x01,0x09,0xa1]
v_min_u16 v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: instruction not supported on this GPU
// VI9: v_min_i16_dpp v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x64,0x02,0x01,0x09,0xa1]
v_min_i16 v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: instruction not supported on this GPU
// VI9: v_ldexp_f16_dpp v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x66,0x02,0x01,0x09,0xa1]
v_ldexp_f16 v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: instruction not supported on this GPU
// NOGFX9: error: not a valid operand.
// VI: v_add_u32_dpp v1, vcc, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x32,0x02,0x01,0x09,0xa1]
v_add_u32 v1, vcc, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: instruction not supported on this GPU
// NOGFX9: error: not a valid operand.
// VI: v_sub_u32_dpp v1, vcc, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x34,0x02,0x01,0x09,0xa1]
v_sub_u32 v1, vcc, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: instruction not supported on this GPU
// NOGFX9: error: not a valid operand.
// VI: v_subrev_u32_dpp v1, vcc, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x36,0x02,0x01,0x09,0xa1]
v_subrev_u32 v1, vcc, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// NOGFX9: error: instruction not supported on this GPU
// VI: v_addc_u32_dpp v1, vcc, v2, v3, vcc row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x38,0x02,0x01,0x09,0xa1]
v_addc_u32 v1, vcc, v2, v3, vcc row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// NOGFX9: error: instruction not supported on this GPU
// VI: v_subb_u32_dpp v1, vcc, v2, v3, vcc row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x3a,0x02,0x01,0x09,0xa1]
v_subb_u32 v1, vcc, v2, v3, vcc row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// NOGFX9: error: instruction not supported on this GPU
// VI: v_subbrev_u32_dpp v1, vcc, v2, v3, vcc row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x3c,0x02,0x01,0x09,0xa1]
v_subbrev_u32 v1, vcc, v2, v3, vcc row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// NOVI: error: instruction not supported on this GPU
// GFX9: v_add_co_u32_dpp v1, vcc, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x32,0x02,0x01,0x09,0xa1]
v_add_co_u32 v1, vcc, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// NOVI: error: instruction not supported on this GPU
// GFX9: v_sub_co_u32_dpp v1, vcc, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x34,0x02,0x01,0x09,0xa1]
v_sub_co_u32 v1, vcc, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// NOVI: error: instruction not supported on this GPU
// GFX9: v_subrev_co_u32_dpp v1, vcc, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x36,0x02,0x01,0x09,0xa1]
v_subrev_co_u32 v1, vcc, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: instruction not supported on this GPU
// NOVI: error: instruction not supported on this GPU
// GFX9: v_addc_co_u32_dpp v1, vcc, v2, v3, vcc row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x38,0x02,0x01,0x09,0xa1]
v_addc_co_u32 v1, vcc, v2, v3, vcc row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: instruction not supported on this GPU
// NOVI: error: instruction not supported on this GPU
// GFX9: v_subb_co_u32_dpp v1, vcc, v2, v3, vcc row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x3a,0x02,0x01,0x09,0xa1]
v_subb_co_u32 v1, vcc, v2, v3, vcc row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: instruction not supported on this GPU
// NOVI: error: instruction not supported on this GPU
// GFX9: v_subbrev_co_u32_dpp v1, vcc, v2, v3, vcc row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:1 ; encoding: [0xfa,0x06,0x02,0x3c,0x02,0x01,0x09,0xa1]
v_subbrev_co_u32 v1, vcc, v2, v3, vcc row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand.
// VI9:    v_cndmask_b32_dpp v5, v1, v2, vcc quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0 ; encoding: [0xfa,0x04,0x0a,0x00,0x01,0xe4,0x00,0x00]
v_cndmask_b32_dpp v5, v1, v2, vcc quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0

// NOSICI: error: not a valid operand.
// VI9:    v_cndmask_b32_dpp v5, v1, v2, vcc row_shl:15 row_mask:0x0 bank_mask:0x0 ; encoding: [0xfa,0x04,0x0a,0x00,0x01,0x0f,0x01,0x00]
v_cndmask_b32_dpp v5, v1, v2, vcc row_shl:15 row_mask:0x0 bank_mask:0x0

//===----------------------------------------------------------------------===//
// Check that immideates and scalar regs are not supported
//===----------------------------------------------------------------------===//

// NOSICI: error: not a valid operand
// NOVI: error: invalid operand for instruction
// NOGFX9: error: invalid operand for instruction
v_mov_b32 v0, 1 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand
// NOVI: error: invalid operand for instruction
// NOGFX9: error: invalid operand for instruction
v_and_b32 v0, 42, v1 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand
// NOVI: error: invalid operand for instruction
// NOGFX9: error: invalid operand for instruction
v_add_f32 v0, v1, 345 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand
// NOVI: error: invalid operand for instruction
// NOGFX9: error: invalid operand for instruction
v_mov_b32 v0, s1 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand
// NOVI: error: invalid operand for instruction
// NOGFX9: error: invalid operand for instruction
v_and_b32 v0, s42, v1 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: not a valid operand
// NOVI: error: invalid operand for instruction
// NOGFX9: error: invalid operand for instruction
v_add_f32 v0, v1, s45 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

//===----------------------------------------------------------------------===//
// Validate register size checks (bug 37943)
//===----------------------------------------------------------------------===//

// NOSICI: error: dpp variant of this instruction is not supported
// NOVI: error: invalid operand for instruction
// NOGFX9: error: invalid operand for instruction
v_add_f32_dpp v5, v[1:2], v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0

// NOSICI: error: dpp variant of this instruction is not supported
// NOVI: error: invalid operand for instruction
// NOGFX9: error: invalid operand for instruction
v_add_f32_dpp v5, v[1:3], v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0

// NOSICI: error: dpp variant of this instruction is not supported
// NOVI: error: invalid operand for instruction
// NOGFX9: error: invalid operand for instruction
v_add_f32_dpp v5, v1, v[1:2] quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0

// NOSICI: error: dpp variant of this instruction is not supported
// NOVI: error: invalid operand for instruction
// NOGFX9: error: invalid operand for instruction
v_add_f32_dpp v5, v1, v[1:4] quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0

// NOSICI: error: instruction not supported on this GPU
// NOVI: error: invalid operand for instruction
// NOGFX9: error: invalid operand for instruction
v_add_f16 v1, v[2:3], v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error: instruction not supported on this GPU
// NOVI: error: invalid operand for instruction
// NOGFX9: error: invalid operand for instruction
v_add_f16 v1, v3, v[2:3] row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0
