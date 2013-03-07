//===-- SIIntrinsics.td - SI Intrinsic defs ----------------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// SI Intrinsic Definitions
//
//===----------------------------------------------------------------------===//


let TargetPrefix = "SI", isTarget = 1 in {

  def int_SI_packf16 : Intrinsic <[llvm_i32_ty], [llvm_float_ty, llvm_float_ty], [IntrNoMem]>;
  def int_SI_export : Intrinsic <[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty], []>;
  /* XXX: We may need a seperate intrinsic here for loading integer values */
  def int_SI_load_const : Intrinsic <[llvm_float_ty], [llvm_i64_ty, llvm_i32_ty], []>;
  def int_SI_vs_load_input : Intrinsic <[llvm_v4f32_ty], [llvm_v16i8_ty, llvm_i16_ty, llvm_i32_ty], [IntrReadMem]> ;
  def int_SI_wqm : Intrinsic <[], [], []>;

  class Sample : Intrinsic <[llvm_v4f32_ty], [llvm_i32_ty, llvm_anyvector_ty, llvm_v32i8_ty, llvm_v16i8_ty, llvm_i32_ty], [IntrReadMem]>;

  def int_SI_sample : Sample;
  def int_SI_sampleb : Sample;
  def int_SI_samplel : Sample;

  /* Interpolation Intrinsics */

  def int_SI_set_M0 : Intrinsic <[llvm_i32_ty], [llvm_i32_ty]>;
  class Interp : Intrinsic <[llvm_float_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrReadMem]>;

  def int_SI_fs_interp_linear_center : Interp;
  def int_SI_fs_interp_linear_centroid : Interp;
  def int_SI_fs_interp_persp_center : Interp;
  def int_SI_fs_interp_persp_centroid : Interp;
  def int_SI_fs_interp_constant : Interp;

  def int_SI_fs_read_face : Intrinsic <[llvm_float_ty], [], [IntrNoMem]>;
  def int_SI_fs_read_pos : Intrinsic <[llvm_float_ty], [llvm_i32_ty], [IntrNoMem]>;

  /* Control flow Intrinsics */

  def int_SI_if : Intrinsic<[llvm_i64_ty], [llvm_i1_ty, llvm_empty_ty], []>;
  def int_SI_else : Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_empty_ty], []>;
  def int_SI_break : Intrinsic<[llvm_i64_ty], [llvm_i64_ty], []>;
  def int_SI_if_break : Intrinsic<[llvm_i64_ty], [llvm_i1_ty, llvm_i64_ty], []>;
  def int_SI_else_break : Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty], []>;
  def int_SI_loop : Intrinsic<[], [llvm_i64_ty, llvm_empty_ty], []>;
  def int_SI_end_cf : Intrinsic<[], [llvm_i64_ty], []>;
}
