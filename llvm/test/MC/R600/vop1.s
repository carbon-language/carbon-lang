// RUN: llvm-mc -arch=amdgcn -show-encoding %s | FileCheck %s
// RUN: llvm-mc -arch=amdgcn -mcpu=SI -show-encoding %s | FileCheck %s

// CHECK: v_nop ; encoding: [0x00,0x00,0x00,0x7e]
v_nop

// CHECK: v_mov_b32_e32 v1, v2 ; encoding: [0x02,0x03,0x02,0x7e]
v_mov_b32 v1, v2

// CHECK: v_readfirstlane_b32 s1, v2 ; encoding: [0x02,0x05,0x02,0x7e]
v_readfirstlane_b32 s1, v2

// CHECK: v_cvt_i32_f64_e32 v1, v[2:3] ; encoding: [0x02,0x07,0x02,0x7e]
v_cvt_i32_f64 v1, v[2:3]

// CHECK: v_cvt_f64_i32_e32 v[1:2], v2 ; encoding: [0x02,0x09,0x02,0x7e]
v_cvt_f64_i32 v[1:2], v2

// CHECK: v_cvt_f32_i32_e32 v1, v2 ; encoding: [0x02,0x0b,0x02,0x7e]
v_cvt_f32_i32 v1, v2

// CHECK: v_cvt_f32_u32_e32 v1, v2 ; encoding: [0x02,0x0d,0x02,0x7e]
v_cvt_f32_u32 v1, v2

// CHECK: v_cvt_u32_f32_e32 v1, v2 ; encoding: [0x02,0x0f,0x02,0x7e
v_cvt_u32_f32 v1, v2

// CHECK: v_cvt_i32_f32_e32 v1, v2 ; encoding: [0x02,0x11,0x02,0x7e]
v_cvt_i32_f32 v1, v2

// CHECK: v_mov_fed_b32_e32 v1, v2 ; encoding: [0x02,0x13,0x02,0x7e]
v_mov_fed_b32 v1, v2

// CHECK: v_cvt_f16_f32_e32 v1, v2 ; encoding: [0x02,0x15,0x02,0x7e]
v_cvt_f16_f32 v1, v2

// CHECK: v_cvt_f32_f16_e32 v1, v2 ; encoding: [0x02,0x17,0x02,0x7e]
v_cvt_f32_f16 v1, v2

// CHECK: v_cvt_rpi_i32_f32_e32 v1, v2 ; encoding: [0x02,0x19,0x02,0x7e]
v_cvt_rpi_i32_f32 v1, v2

// CHECK: v_cvt_flr_i32_f32_e32 v1, v2 ; encoding: [0x02,0x1b,0x02,0x7e]
v_cvt_flr_i32_f32 v1, v2

// CHECK: v_cvt_off_f32_i4_e32 v1, v2 ; encoding: [0x02,0x1d,0x02,0x7e]
v_cvt_off_f32_i4_e32 v1, v2

// CHECK: v_cvt_f32_f64_e32 v1, v[2:3] ; encoding: [0x02,0x1f,0x02,0x7e]
v_cvt_f32_f64 v1, v[2:3]

// CHECK: v_cvt_f64_f32_e32 v[1:2], v2 ; encoding: [0x02,0x21,0x02,0x7e]
v_cvt_f64_f32 v[1:2], v2

// CHECK: v_cvt_f32_ubyte0_e32 v1, v2 ; encoding: [0x02,0x23,0x02,0x7e]
v_cvt_f32_ubyte0 v1, v2

// CHECK: v_cvt_f32_ubyte1_e32 v1, v2 ; encoding: [0x02,0x25,0x02,0x7e]
v_cvt_f32_ubyte1_e32 v1, v2

// CHECK: v_cvt_f32_ubyte2_e32 v1, v2 ; encoding: [0x02,0x27,0x02,0x7e]
v_cvt_f32_ubyte2 v1, v2

// CHECK: v_cvt_f32_ubyte3_e32 v1, v2 ; encoding: [0x02,0x29,0x02,0x7e]
v_cvt_f32_ubyte3 v1, v2

// CHECK: v_cvt_u32_f64_e32 v1, v[2:3] ; encoding: [0x02,0x2b,0x02,0x7e]
v_cvt_u32_f64 v1, v[2:3]

// CHECK: v_cvt_f64_u32_e32 v[1:2], v2 ; encoding: [0x02,0x2d,0x02,0x7e]
v_cvt_f64_u32 v[1:2], v2

// CHECK: v_fract_f32_e32 v1, v2 ; encoding: [0x02,0x41,0x02,0x7e]
v_fract_f32 v1, v2

// CHECK: v_trunc_f32_e32 v1, v2 ; encoding: [0x02,0x43,0x02,0x7e]
v_trunc_f32 v1, v2

// CHECK: v_ceil_f32_e32 v1, v2 ; encoding: [0x02,0x45,0x02,0x7e]
v_ceil_f32 v1, v2

// CHECK: v_rndne_f32_e32 v1, v2 ; encoding: [0x02,0x47,0x02,0x7e]
v_rndne_f32 v1, v2

// CHECK: v_floor_f32_e32 v1, v2 ; encoding: [0x02,0x49,0x02,0x7e]
v_floor_f32_e32 v1, v2

// CHECK: v_exp_f32_e32 v1, v2 ; encoding: [0x02,0x4b,0x02,0x7e]
v_exp_f32 v1, v2

// CHECK: v_log_clamp_f32_e32 v1, v2 ; encoding: [0x02,0x4d,0x02,0x7e]
v_log_clamp_f32 v1, v2

// CHECK: v_log_f32_e32 v1, v2 ; encoding: [0x02,0x4f,0x02,0x7e]
v_log_f32 v1, v2

// CHECK: v_rcp_clamp_f32_e32 v1, v2 ; encoding: [0x02,0x51,0x02,0x7e]
v_rcp_clamp_f32 v1, v2

// CHECK: v_rcp_legacy_f32_e32 v1, v2 ; encoding: [0x02,0x53,0x02,0x7e]
v_rcp_legacy_f32 v1, v2

// CHECK: v_rcp_f32_e32 v1, v2 ; encoding: [0x02,0x55,0x02,0x7e]
v_rcp_f32 v1, v2

// CHECK: v_rcp_iflag_f32_e32 v1, v2 ; encoding: [0x02,0x57,0x02,0x7e]
v_rcp_iflag_f32 v1, v2

// CHECK: v_rsq_clamp_f32_e32 v1, v2 ; encoding: [0x02,0x59,0x02,0x7e]
v_rsq_clamp_f32 v1, v2

// CHECK: v_rsq_legacy_f32_e32 v1, v2 ; encoding: [0x02,0x5b,0x02,0x7e]
v_rsq_legacy_f32 v1, v2

// CHECK: v_rsq_f32_e32 v1, v2 ; encoding: [0x02,0x5d,0x02,0x7e]
v_rsq_f32_e32 v1, v2

// CHECK: v_rcp_f64_e32 v[1:2], v[2:3] ; encoding: [0x02,0x5f,0x02,0x7e]
v_rcp_f64 v[1:2], v[2:3]

// CHECK: v_rcp_clamp_f64_e32 v[1:2], v[2:3] ; encoding: [0x02,0x61,0x02,0x7e]
v_rcp_clamp_f64 v[1:2], v[2:3]

// CHECK: v_rsq_f64_e32 v[1:2], v[2:3] ; encoding: [0x02,0x63,0x02,0x7e]
v_rsq_f64 v[1:2], v[2:3]

// CHECK: v_rsq_clamp_f64_e32 v[1:2], v[2:3] ; encoding: [0x02,0x65,0x02,0x7e]
v_rsq_clamp_f64 v[1:2], v[2:3]

// CHECK: v_sqrt_f32_e32 v1, v2 ; encoding: [0x02,0x67,0x02,0x7e]
v_sqrt_f32 v1, v2

// CHECK: v_sqrt_f64_e32 v[1:2], v[2:3] ; encoding: [0x02,0x69,0x02,0x7e]
v_sqrt_f64 v[1:2], v[2:3]

// CHECK: v_sin_f32_e32 v1, v2 ; encoding: [0x02,0x6b,0x02,0x7e]
v_sin_f32 v1, v2

// CHECK: v_cos_f32_e32 v1, v2 ; encoding: [0x02,0x6d,0x02,0x7e]
v_cos_f32 v1, v2

// CHECK: v_not_b32_e32 v1, v2 ; encoding: [0x02,0x6f,0x02,0x7e]
v_not_b32 v1, v2

// CHECK: v_bfrev_b32_e32 v1, v2 ; encoding: [0x02,0x71,0x02,0x7e]
v_bfrev_b32 v1, v2

// CHECK: v_ffbh_u32_e32 v1, v2 ; encoding: [0x02,0x73,0x02,0x7e]
v_ffbh_u32 v1, v2

// CHECK: v_ffbl_b32_e32 v1, v2 ; encoding: [0x02,0x75,0x02,0x7e]
v_ffbl_b32 v1, v2

// CHECK: v_ffbh_i32_e32 v1, v2 ; encoding: [0x02,0x77,0x02,0x7e]
v_ffbh_i32_e32 v1, v2

// CHECK: v_frexp_exp_i32_f64_e32 v1, v[2:3] ; encoding: [0x02,0x79,0x02,0x7e]
v_frexp_exp_i32_f64 v1, v[2:3]

// CHECK: v_frexp_mant_f64_e32 v[1:2], v[2:3] ; encoding: [0x02,0x7b,0x02,0x7e]
v_frexp_mant_f64 v[1:2], v[2:3]

// CHECK: v_fract_f64_e32 v[1:2], v[2:3] ; encoding: [0x02,0x7d,0x02,0x7e]
v_fract_f64 v[1:2], v[2:3]

// CHECK: v_frexp_exp_i32_f32_e32 v1, v2 ; encoding: [0x02,0x7f,0x02,0x7e]
v_frexp_exp_i32_f32 v1, v2

// CHECK: v_frexp_mant_f32_e32 v1, v2 ; encoding: [0x02,0x81,0x02,0x7e]
v_frexp_mant_f32 v1, v2

// CHECK: v_clrexcp ; encoding: [0x00,0x82,0x00,0x7e]
v_clrexcp

// CHECK: v_movreld_b32_e32 v1, v2 ; encoding: [0x02,0x85,0x02,0x7e]
v_movreld_b32 v1, v2

// CHECK: v_movrels_b32_e32 v1, v2 ; encoding: [0x02,0x87,0x02,0x7e]
v_movrels_b32 v1, v2

// CHECK: v_movrelsd_b32_e32 v1, v2 ; encoding: [0x02,0x89,0x02,0x7e]
v_movrelsd_b32 v1, v2
