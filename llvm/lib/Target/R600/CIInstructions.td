//===-- CIInstructions.td - CI Instruction Defintions ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Instruction definitions for CI and newer.
//===----------------------------------------------------------------------===//


def isCIVI : Predicate <
  "Subtarget->getGeneration() == AMDGPUSubtarget::SEA_ISLANDS || "
  "Subtarget->getGeneration() == AMDGPUSubtarget::VOLCANIC_ISLANDS"
>;

//===----------------------------------------------------------------------===//
// VOP1 Instructions
//===----------------------------------------------------------------------===//

let SubtargetPredicate = isCIVI in {

defm V_TRUNC_F64 : VOP1Inst <vop1<0x17>, "v_trunc_f64",
  VOP_F64_F64, ftrunc
>;
defm V_CEIL_F64 : VOP1Inst <vop1<0x18>, "v_ceil_f64",
  VOP_F64_F64, fceil
>;
defm V_FLOOR_F64 : VOP1Inst <vop1<0x1A>, "v_floor_f64",
  VOP_F64_F64, ffloor
>;
defm V_RNDNE_F64 : VOP1Inst <vop1<0x19>, "v_rndne_f64",
  VOP_F64_F64, frint
>;
defm V_LOG_LEGACY_F32 : VOP1Inst <vop1<0x45, 0x4c>, "v_log_legacy_f32",
  VOP_F32_F32
>;
defm V_EXP_LEGACY_F32 : VOP1Inst <vop1<0x46, 0x4b>, "v_exp_legacy_f32",
  VOP_F32_F32
>;
} // End SubtargetPredicate = isCIVI
