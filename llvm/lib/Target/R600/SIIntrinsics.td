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
  def int_SI_load_const : Intrinsic <[llvm_float_ty], [llvm_v16i8_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_SI_vs_load_input : Intrinsic <[llvm_v4f32_ty], [llvm_v16i8_ty, llvm_i16_ty, llvm_i32_ty], [IntrNoMem]> ;

  class Sample : Intrinsic <[llvm_v4f32_ty], [llvm_anyvector_ty, llvm_v32i8_ty, llvm_v16i8_ty, llvm_i32_ty], [IntrNoMem]>;

  def int_SI_sample : Sample;
  def int_SI_sampleb : Sample;
  def int_SI_sampled : Sample;
  def int_SI_samplel : Sample;

  def int_SI_imageload : Intrinsic <[llvm_v4i32_ty], [llvm_anyvector_ty, llvm_v32i8_ty, llvm_i32_ty], [IntrNoMem]>;

  def int_SI_resinfo : Intrinsic <[llvm_v4i32_ty], [llvm_i32_ty, llvm_v32i8_ty, llvm_i32_ty], [IntrNoMem]>;

  /* Interpolation Intrinsics */

  def int_SI_fs_constant : Intrinsic <[llvm_float_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_SI_fs_interp : Intrinsic <[llvm_float_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

  /* Control flow Intrinsics */

  def int_SI_if : Intrinsic<[llvm_i64_ty], [llvm_i1_ty, llvm_empty_ty], []>;
  def int_SI_else : Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_empty_ty], []>;
  def int_SI_break : Intrinsic<[llvm_i64_ty], [llvm_i64_ty], []>;
  def int_SI_if_break : Intrinsic<[llvm_i64_ty], [llvm_i1_ty, llvm_i64_ty], []>;
  def int_SI_else_break : Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty], []>;
  def int_SI_loop : Intrinsic<[], [llvm_i64_ty, llvm_empty_ty], []>;
  def int_SI_end_cf : Intrinsic<[], [llvm_i64_ty], []>;
}
