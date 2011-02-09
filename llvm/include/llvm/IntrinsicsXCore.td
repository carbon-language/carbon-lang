//==- IntrinsicsXCore.td - XCore intrinsics                 -*- tablegen -*-==//
// 
// Copyright (C) 2008 XMOS
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the XCore-specific intrinsics.
//
//===----------------------------------------------------------------------===//

let TargetPrefix = "xcore" in {  // All intrinsics start with "llvm.xcore.".
  def int_xcore_bitrev : Intrinsic<[llvm_i32_ty],[llvm_i32_ty],[IntrNoMem]>;
  def int_xcore_getid : Intrinsic<[llvm_i32_ty],[],[IntrNoMem]>;

  // Resource instructions.
  def int_xcore_getr : Intrinsic<[llvm_anyptr_ty],[llvm_i32_ty]>;
  def int_xcore_freer : Intrinsic<[],[llvm_anyptr_ty],
                                   [NoCapture<0>]>;
  def int_xcore_in : Intrinsic<[llvm_i32_ty],[llvm_anyptr_ty],[NoCapture<0>]>;
  def int_xcore_int : Intrinsic<[llvm_i32_ty],[llvm_anyptr_ty],
                                [NoCapture<0>]>;
  def int_xcore_inct : Intrinsic<[llvm_i32_ty],[llvm_anyptr_ty],
                                 [NoCapture<0>]>;
  def int_xcore_out : Intrinsic<[],[llvm_anyptr_ty, llvm_i32_ty],
                                [NoCapture<0>]>;
  def int_xcore_outt : Intrinsic<[],[llvm_anyptr_ty, llvm_i32_ty],
                                 [NoCapture<0>]>;
  def int_xcore_outct : Intrinsic<[],[llvm_anyptr_ty, llvm_i32_ty],
                                  [NoCapture<0>]>;
  def int_xcore_chkct : Intrinsic<[],[llvm_anyptr_ty, llvm_i32_ty],
                                  [NoCapture<0>]>;
  def int_xcore_setd : Intrinsic<[],[llvm_anyptr_ty, llvm_i32_ty],
                                  [NoCapture<0>]>;
  def int_xcore_setc : Intrinsic<[],[llvm_anyptr_ty, llvm_i32_ty],
                                  [NoCapture<0>]>;
}
