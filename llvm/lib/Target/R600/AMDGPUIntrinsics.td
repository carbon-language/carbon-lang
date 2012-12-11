//===-- AMDGPUIntrinsics.td - Common intrinsics  -*- tablegen -*-----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines intrinsics that are used by all hw codegen targets.
//
//===----------------------------------------------------------------------===//

let TargetPrefix = "AMDGPU", isTarget = 1 in {

  def int_AMDGPU_load_const : Intrinsic<[llvm_float_ty], [llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_load_imm : Intrinsic<[llvm_v4f32_ty], [llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_reserve_reg : Intrinsic<[], [llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_store_output : Intrinsic<[], [llvm_float_ty, llvm_i32_ty], []>;
  def int_AMDGPU_swizzle : Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_i32_ty], [IntrNoMem]>;

  def int_AMDGPU_arl : Intrinsic<[llvm_i32_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_AMDGPU_cndlt : Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty, llvm_float_ty], [IntrNoMem]>;
  def int_AMDGPU_div : Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty], [IntrNoMem]>;
  def int_AMDGPU_dp4 : Intrinsic<[llvm_float_ty], [llvm_v4f32_ty, llvm_v4f32_ty], [IntrNoMem]>;
  def int_AMDGPU_kill : Intrinsic<[], [llvm_float_ty], []>;
  def int_AMDGPU_kilp : Intrinsic<[], [], []>;
  def int_AMDGPU_lrp : Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty, llvm_float_ty], [IntrNoMem]>;
  def int_AMDGPU_mul : Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty], [IntrNoMem]>;
  def int_AMDGPU_pow : Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty], [IntrNoMem]>;
  def int_AMDGPU_rcp : Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_AMDGPU_rsq : Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_AMDGPU_seq : Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty], [IntrNoMem]>;
  def int_AMDGPU_sgt : Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty], [IntrNoMem]>;
  def int_AMDGPU_sge : Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty], [IntrNoMem]>;
  def int_AMDGPU_sle : Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty], [IntrNoMem]>;
  def int_AMDGPU_sne : Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty], [IntrNoMem]>;
  def int_AMDGPU_mullit : Intrinsic<[llvm_v4f32_ty], [llvm_float_ty, llvm_float_ty, llvm_float_ty], [IntrNoMem]>;
  def int_AMDGPU_tex : Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_txb : Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_txf : Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_txq : Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_txd : Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_v4f32_ty, llvm_v4f32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_txl : Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_trunc : Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_AMDGPU_ddx : Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_ddy : Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_imax : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_imin : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_umax : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_umin : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_cube : Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty], [IntrNoMem]>;

  def int_AMDGPU_shader_type : Intrinsic<[], [llvm_i32_ty], []>;
}

let TargetPrefix = "TGSI", isTarget = 1 in {

  def int_TGSI_lit_z : Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty, llvm_float_ty],[IntrNoMem]>;
}

include "SIIntrinsics.td"
