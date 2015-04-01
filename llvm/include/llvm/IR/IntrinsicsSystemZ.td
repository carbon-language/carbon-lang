//===- IntrinsicsSystemZ.td - Defines SystemZ intrinsics ---*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the SystemZ-specific intrinsics.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//
// Transactional-execution intrinsics
//
//===----------------------------------------------------------------------===//

let TargetPrefix = "s390" in {
  def int_s390_tbegin : Intrinsic<[llvm_i32_ty], [llvm_ptr_ty, llvm_i32_ty],
                                  [IntrNoDuplicate]>;

  def int_s390_tbegin_nofloat : Intrinsic<[llvm_i32_ty],
                                          [llvm_ptr_ty, llvm_i32_ty],
                                          [IntrNoDuplicate]>;

  def int_s390_tbeginc : Intrinsic<[], [llvm_ptr_ty, llvm_i32_ty],
                                   [IntrNoDuplicate]>;

  def int_s390_tabort : Intrinsic<[], [llvm_i64_ty],
                                  [IntrNoReturn, Throws]>;

  def int_s390_tend : GCCBuiltin<"__builtin_tend">,
                      Intrinsic<[llvm_i32_ty], []>;

  def int_s390_etnd : GCCBuiltin<"__builtin_tx_nesting_depth">,
                      Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>;

  def int_s390_ntstg : Intrinsic<[], [llvm_i64_ty, llvm_ptr64_ty],
                                 [IntrReadWriteArgMem]>;

  def int_s390_ppa_txassist : GCCBuiltin<"__builtin_tx_assist">,
                              Intrinsic<[], [llvm_i32_ty]>;
}

