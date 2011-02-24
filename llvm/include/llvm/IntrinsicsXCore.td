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
  def int_xcore_inshr : Intrinsic<[llvm_i32_ty],[llvm_anyptr_ty, llvm_i32_ty],
                                  [NoCapture<0>]>;
  def int_xcore_outshr : Intrinsic<[llvm_i32_ty],[llvm_anyptr_ty, llvm_i32_ty],
                                  [NoCapture<0>]>;
  def int_xcore_setpt : Intrinsic<[],[llvm_anyptr_ty, llvm_i32_ty],
                                  [NoCapture<0>]>;
  def int_xcore_getts : Intrinsic<[llvm_i32_ty],[llvm_anyptr_ty],
                                  [NoCapture<0>]>;
  def int_xcore_syncr : Intrinsic<[],[llvm_anyptr_ty],
                                  [NoCapture<0>]>;
  def int_xcore_settw : Intrinsic<[],[llvm_anyptr_ty, llvm_i32_ty],
                                  [NoCapture<0>]>;
  def int_xcore_setv : Intrinsic<[],[llvm_anyptr_ty, llvm_ptr_ty],
                                 [NoCapture<0>]>;
  def int_xcore_eeu : Intrinsic<[],[llvm_anyptr_ty], [NoCapture<0>]>;

  // Intrinsics for events.
  def int_xcore_waitevent : Intrinsic<[llvm_ptr_ty],[], [IntrReadMem]>;
  def int_xcore_clre : Intrinsic<[],[],[]>;
}
