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

let SIAssemblerPredicate = DisableInst, SubtargetPredicate = isVI in {

//===----------------------------------------------------------------------===//
// VOP1 Instructions
//===----------------------------------------------------------------------===//

defm V_CVT_F16_U16 : VOP1Inst <vop1<0, 0x39>, "v_cvt_f16_u16", VOP_F16_I16>;
defm V_CVT_F16_I16 : VOP1Inst <vop1<0, 0x3a>, "v_cvt_f16_i16", VOP_F16_I16>;
defm V_CVT_U16_F16 : VOP1Inst <vop1<0, 0x3b>, "v_cvt_u16_f16", VOP_I16_F16>;
defm V_CVT_I16_F16 : VOP1Inst <vop1<0, 0x3c>, "v_cvt_i16_f16", VOP_I16_F16>;
defm V_RCP_F16 : VOP1Inst <vop1<0, 0x3d>, "v_rcp_f16", VOP_F16_F16>;
defm V_SQRT_F16 : VOP1Inst <vop1<0, 0x3e>, "v_sqrt_f16", VOP_F16_F16>;
defm V_RSQ_F16 : VOP1Inst <vop1<0, 0x3f>, "v_rsq_f16", VOP_F16_F16>;
defm V_LOG_F16 : VOP1Inst <vop1<0, 0x40>, "v_log_f16", VOP_F16_F16>;
defm V_EXP_F16 : VOP1Inst <vop1<0, 0x41>, "v_exp_f16", VOP_F16_F16>;
defm V_FREXP_MANT_F16 : VOP1Inst <vop1<0, 0x42>, "v_frexp_mant_f16",
  VOP_F16_F16
>;
defm V_FREXP_EXP_I16_F16 : VOP1Inst <vop1<0, 0x43>, "v_frexp_exp_i16_f16",
  VOP_I16_F16
>;
defm V_FLOOR_F16 : VOP1Inst <vop1<0, 0x44>, "v_floor_f16", VOP_F16_F16>;
defm V_CEIL_F16 : VOP1Inst <vop1<0, 0x45>, "v_ceil_f16", VOP_F16_F16>;
defm V_TRUNC_F16 : VOP1Inst <vop1<0, 0x46>, "v_trunc_f16", VOP_F16_F16>;
defm V_RNDNE_F16 : VOP1Inst <vop1<0, 0x47>, "v_rndne_f16", VOP_F16_F16>;
defm V_FRACT_F16 : VOP1Inst <vop1<0, 0x48>, "v_fract_f16", VOP_F16_F16>;
defm V_SIN_F16 : VOP1Inst <vop1<0, 0x49>, "v_sin_f16", VOP_F16_F16>;
defm V_COS_F16 : VOP1Inst <vop1<0, 0x4a>, "v_cos_f16", VOP_F16_F16>;

} // End SIAssemblerPredicate = DisableInst, SubtargetPredicate = isVI

//===----------------------------------------------------------------------===//
// SMEM Patterns
//===----------------------------------------------------------------------===//

let Predicates = [isVI] in {

// 1. Offset as 20bit DWORD immediate
def : Pat <
  (SIload_constant v4i32:$sbase, IMM20bit:$offset),
  (S_BUFFER_LOAD_DWORD_IMM $sbase, (as_i32imm $offset))
>;

} // End Predicates = [isVI]
