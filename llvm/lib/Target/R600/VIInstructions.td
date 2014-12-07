//===-- VIInstructions.td - VI Instruction Defintions ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Instruction definitions for VI and newer.
//===----------------------------------------------------------------------===//

let SubtargetPredicate = isVI in {

def V_LDEXP_F32 : VOP3InstVI <0x288, "v_ldexp_f32", VOP_F32_F32_I32,
  AMDGPUldexp
>;
def V_BFM_B32 : VOP3InstVI <0x293, "v_bfm_b32", VOP_I32_I32_I32, AMDGPUbfm>;
def V_BCNT_U32_B32 : VOP3InstVI <0x28b, "v_bcnt_u32_b32", VOP_I32_I32_I32>;
def V_MBCNT_LO_U32_B32 : VOP3InstVI <0x28c, "v_mbcnt_lo_u32_b32",
  VOP_I32_I32_I32
>;
def V_MBCNT_HI_U32_B32 : VOP3InstVI <0x28d, "v_mbcnt_hi_u32_b32",
  VOP_I32_I32_I32
>;

def V_CVT_PKRTZ_F16_F32 : VOP3InstVI <0x296, "v_cvt_pkrtz_f16_f32",
 VOP_I32_F32_F32, int_SI_packf16
>;

defm BUFFER_LOAD_DWORD_VI : MUBUF_Load_Helper_vi <
  0x14, "buffer_load_dword", VReg_32, i32, global_load
>;

defm BUFFER_LOAD_FORMAT_XYZW_VI : MUBUF_Load_Helper_vi <
  0x03, "buffer_load_format_xyzw", VReg_128
>;

} // End SubtargetPredicate = isVI

//===----------------------------------------------------------------------===//
// VOP2 Patterns
//===----------------------------------------------------------------------===//

let Predicates = [isVI] in {

def : Pat <
  (int_SI_tid),
  (V_MBCNT_HI_U32_B32 0xffffffff,
                      (V_MBCNT_LO_U32_B32 0xffffffff, 0))
>;

//===----------------------------------------------------------------------===//
// SMEM Patterns
//===----------------------------------------------------------------------===//

// 1. Offset as 8bit DWORD immediate
def : Pat <
  (SIload_constant v4i32:$sbase, IMM20bit:$offset),
  (S_BUFFER_LOAD_DWORD_IMM $sbase, (as_i32imm $offset))
>;

//===----------------------------------------------------------------------===//
// MUBUF Patterns
//===----------------------------------------------------------------------===//

// Offset in an 32Bit VGPR
def : Pat <
  (SIload_constant v4i32:$sbase, i32:$voff),
  (BUFFER_LOAD_DWORD_VI_OFFEN $sbase, $voff, 0, 0, 0, 0, 0)
>;

// Offset in an 32Bit VGPR
def : Pat <
  (SIload_constant v4i32:$sbase, i32:$voff),
  (BUFFER_LOAD_DWORD_VI_OFFEN $sbase, $voff, 0, 0, 0, 0, 0)
>;

/* int_SI_vs_load_input */
def : Pat<
  (SIload_input v4i32:$tlst, imm:$attr_offset, i32:$buf_idx_vgpr),
  (BUFFER_LOAD_FORMAT_XYZW_VI_IDXEN $tlst, $buf_idx_vgpr, imm:$attr_offset, 0, 0, 0, 0)
>;

defm : MUBUF_Load_Dword <i32, BUFFER_LOAD_DWORD_VI_OFFSET,
                         BUFFER_LOAD_DWORD_VI_OFFEN,
                         BUFFER_LOAD_DWORD_VI_IDXEN,
                         BUFFER_LOAD_DWORD_VI_BOTHEN>;

} // End Predicates = [isVI]
