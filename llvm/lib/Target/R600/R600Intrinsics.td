//===-- R600Intrinsics.td - R600 Instrinsic defs -------*- tablegen -*-----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// R600 Intrinsic Definitions
//
//===----------------------------------------------------------------------===//

let TargetPrefix = "R600", isTarget = 1 in {
  def int_R600_load_input :
    Intrinsic<[llvm_float_ty], [llvm_i32_ty], [IntrNoMem]>;
  def int_R600_interp_input :
    Intrinsic<[llvm_float_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_R600_load_texbuf :
    Intrinsic<[llvm_v4f32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_R600_store_swizzle :
    Intrinsic<[], [llvm_v4f32_ty, llvm_i32_ty, llvm_i32_ty], []>;
  def int_R600_store_stream_output :
    Intrinsic<[], [llvm_v4f32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], []>;
  def int_R600_store_pixel_depth :
      Intrinsic<[], [llvm_float_ty], []>;
  def int_R600_store_pixel_stencil :
      Intrinsic<[], [llvm_float_ty], []>;
  def int_R600_store_dummy :
      Intrinsic<[], [llvm_i32_ty], []>;
}
