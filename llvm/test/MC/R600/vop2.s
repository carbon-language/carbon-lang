// RUN: llvm-mc -arch=amdgcn -show-encoding %s | FileCheck %s
// RUN: llvm-mc -arch=amdgcn -mcpu=SI -show-encoding %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Generic Checks for floating-point instructions (These have modifiers).
//===----------------------------------------------------------------------===//

// TODO: 64-bit encoding of instructions with modifiers

// _e32 suffix
// CHECK: v_add_f32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x06]
v_add_f32_e32 v1, v2, v3

// src0 inline immediate
// CHECK: v_add_f32_e32 v1, 1.0, v3 ; encoding: [0xf2,0x06,0x02,0x06]
v_add_f32 v1, 1.0, v3

// src0 negative inline immediate
// CHECK: v_add_f32_e32 v1, -1.0, v3 ; encoding: [0xf3,0x06,0x02,0x06]
v_add_f32 v1, -1.0, v3

// src0 literal
// CHECK: v_add_f32_e32 v1, 0x42c80000, v3 ; encoding: [0xff,0x06,0x02,0x06,0x00,0x00,0xc8,0x42]
v_add_f32 v1, 100.0, v3

// src0 negative literal
// CHECK: v_add_f32_e32 v1, 0xc2c80000, v3 ; encoding: [0xff,0x06,0x02,0x06,0x00,0x00,0xc8,0xc2]
v_add_f32 v1, -100.0, v3

//===----------------------------------------------------------------------===//
// Generic Checks for integer instructions (These don't have modifiers).
//===----------------------------------------------------------------------===//

// _e32 suffix
// CHECK: v_mul_i32_i24_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x12] 
v_mul_i32_i24_e32 v1, v2, v3

// _e64 suffix
// CHECK: v_mul_i32_i24_e64 v1, v2, v3 ; encoding: [0x01,0x00,0x12,0xd2,0x02,0x07,0x02,0x00]
v_mul_i32_i24_e64 v1, v2, v3

// src0 inline
// CHECK: v_mul_i32_i24_e32 v1, 3, v3 ; encoding: [0x83,0x06,0x02,0x12]
v_mul_i32_i24 v1, 3, v3

// src0 negative inline
// CHECK: v_mul_i32_i24_e32 v1, -3, v3 ; encoding: [0xc3,0x06,0x02,0x12]
v_mul_i32_i24 v1, -3, v3

// src1 inline
// CHECK: v_mul_i32_i24_e64 v1, v2, 3 ; encoding: [0x01,0x00,0x12,0xd2,0x02,0x07,0x01,0x00]
v_mul_i32_i24 v1, v2, 3

// src1 negative inline
// CHECK: v_mul_i32_i24_e64 v1, v2, -3 ; encoding: [0x01,0x00,0x12,0xd2,0x02,0x87,0x01,0x00]
v_mul_i32_i24 v1, v2, -3

// src0 literal
// CHECK: v_mul_i32_i24_e32 v1, 0x64, v3 ; encoding: [0xff,0x06,0x02,0x12,0x64,0x00,0x00,0x00]
v_mul_i32_i24 v1, 100, v3

// src1 negative literal
// CHECK: v_mul_i32_i24_e32 v1, 0xffffff9c, v3 ; encoding: [0xff,0x06,0x02,0x12,0x9c,0xff,0xff,0xff]
v_mul_i32_i24 v1, -100, v3

//===----------------------------------------------------------------------===//
// Checks for legal operands
//===----------------------------------------------------------------------===//

// src0 sgpr
// CHECK: v_mul_i32_i24_e32 v1, s2, v3 ; encoding: [0x02,0x06,0x02,0x12]
v_mul_i32_i24 v1, s2, v3

// src1 sgpr
// CHECK: v_mul_i32_i24_e64 v1, v2, s3 ; encoding: [0x01,0x00,0x12,0xd2,0x02,0x07,0x00,0x00]
v_mul_i32_i24 v1, v2, s3

// src0, src1 same sgpr
// CHECK: v_mul_i32_i24_e64 v1, s2, s2 ; encoding: [0x01,0x00,0x12,0xd2,0x02,0x04,0x00,0x00]
v_mul_i32_i24 v1, s2, s2

// src0 sgpr, src1 inline
// CHECK: v_mul_i32_i24_e64 v1, s2, 3 ; encoding: [0x01,0x00,0x12,0xd2,0x02,0x06,0x01,0x00]
v_mul_i32_i24 v1, s2, 3

// src0 inline src1 sgpr
// CHECK: v_mul_i32_i24_e64 v1, 3, s3 ; encoding: [0x01,0x00,0x12,0xd2,0x83,0x06,0x00,0x00]
v_mul_i32_i24 v1, 3, s3

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

// CHECK: v_cndmask_b32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x00]
v_cndmask_b32 v1, v2, v3

// CHECK: v_readlane_b32 s1, v2, s3 ; encoding: [0x02,0x07,0x02,0x02]
v_readlane_b32 s1, v2, s3

// CHECK: v_writelane_b32 v1, s2, s3 ; encoding: [0x02,0x06,0x02,0x04]
v_writelane_b32 v1, s2, s3

// CHECK: v_add_f32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x06]
v_add_f32 v1, v2, v3

// CHECK: v_sub_f32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x08]
v_sub_f32 v1, v2, v3

// CHECK: v_subrev_f32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x0a]
v_subrev_f32 v1, v2, v3

// CHECK: v_mac_legacy_f32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x0c]
v_mac_legacy_f32 v1, v2, v3

// CHECK: v_mul_legacy_f32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x0e]
v_mul_legacy_f32_e32 v1, v2, v3

// CHECK: v_mul_f32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x10]
v_mul_f32 v1, v2, v3

// CHECK: v_mul_i32_i24_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x12]
v_mul_i32_i24 v1, v2, v3

// CHECK: v_mul_hi_i32_i24_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x14]
v_mul_hi_i32_i24 v1, v2, v3

// CHECK: v_mul_u32_u24_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x16]
v_mul_u32_u24 v1, v2, v3

// CHECK: v_mul_hi_u32_u24_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x18]
v_mul_hi_u32_u24 v1, v2, v3

// CHECK: v_min_legacy_f32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x1a]
v_min_legacy_f32_e32 v1, v2, v3

// CHECK: v_max_legacy_f32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x1c]
v_max_legacy_f32 v1, v2, v3

// CHECK: v_min_f32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x1e]
v_min_f32_e32 v1, v2, v3

// CHECK: v_max_f32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x20]
v_max_f32 v1, v2 v3

// CHECK: v_min_i32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x22]
v_min_i32 v1, v2, v3

// CHECK: v_max_i32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x24]
v_max_i32 v1, v2, v3

// CHECK: v_min_u32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x26]
v_min_u32 v1, v2, v3

// CHECK: v_max_u32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x28]
v_max_u32 v1, v2, v3

// CHECK: v_lshr_b32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x2a]
v_lshr_b32 v1, v2, v3

// CHECK: v_lshrrev_b32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x2c]
v_lshrrev_b32 v1, v2, v3

// CHECK: v_ashr_i32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x2e]
v_ashr_i32 v1, v2, v3

// CHECK: v_ashrrev_i32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x30]
v_ashrrev_i32 v1, v2, v3

// CHECK: v_lshl_b32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x32]
v_lshl_b32_e32 v1, v2, v3

// CHECK: v_lshlrev_b32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x34]
v_lshlrev_b32 v1, v2, v3

// CHECK: v_and_b32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x36]
v_and_b32 v1, v2, v3

// CHECK: v_or_b32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x38]
v_or_b32 v1, v2, v3

// CHECK: v_xor_b32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x3a]
v_xor_b32 v1, v2, v3

// CHECK: v_bfm_b32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x3c]
v_bfm_b32 v1, v2, v3

// CHECK: v_mac_f32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x3e]
v_mac_f32 v1, v2, v3

// CHECK: v_madmk_f32_e32 v1, v2, v3, 0x42800000 ; encoding: [0x02,0x07,0x02,0x40,0x00,0x00,0x80,0x42]
v_madmk_f32 v1, v2, v3, 64.0

// CHECK: v_madak_f32_e32 v1, v2, v3, 0x42800000 ; encoding: [0x02,0x07,0x02,0x42,0x00,0x00,0x80,0x42]
v_madak_f32 v1, v2, v3, 64.0

// CHECK: v_bcnt_u32_b32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x44]
v_bcnt_u32_b32 v1, v2, v3

// CHECK: v_mbcnt_lo_u32_b32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x46]
v_mbcnt_lo_u32_b32 v1, v2, v3

// CHECK: v_mbcnt_hi_u32_b32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x48]
v_mbcnt_hi_u32_b32_e32 v1, v2, v3

// CHECK: v_add_i32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x4a]
v_add_i32 v1, v2, v3

// CHECK: v_sub_i32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x4c]
v_sub_i32_e32 v1, v2, v3

// CHECK: v_subrev_i32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x4e]
v_subrev_i32 v1, v2, v3

// CHECK : v_addc_u32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x50]
v_addc_u32 v1, v2, v3

// CHECK: v_subb_u32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x52]
v_subb_u32 v1, v2, v3

// CHECK: v_subbrev_u32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x54]
v_subbrev_u32 v1, v2, v3

// CHECK: v_ldexp_f32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x56]
v_ldexp_f32 v1, v2, v3

// CHECK: v_cvt_pkaccum_u8_f32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x58]
v_cvt_pkaccum_u8_f32 v1, v2, v3

// CHECK: v_cvt_pknorm_i16_f32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x5a]
v_cvt_pknorm_i16_f32 v1, v2, v3

// CHECK: v_cvt_pknorm_u16_f32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x5c]
v_cvt_pknorm_u16_f32 v1, v2, v3

// CHECK: v_cvt_pkrtz_f16_f32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x5e]
v_cvt_pkrtz_f16_f32 v1, v2, v3

// CHECK: v_cvt_pk_u16_u32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x60]
v_cvt_pk_u16_u32_e32 v1, v2, v3

// CHECK: v_cvt_pk_i16_i32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x62]
v_cvt_pk_i16_i32 v1, v2, v3
