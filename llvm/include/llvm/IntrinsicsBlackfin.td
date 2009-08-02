//===- IntrinsicsBlackfin.td - Defines Blackfin intrinsics -*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the blackfin-specific intrinsics.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Core synchronisation etc.
//
// These intrinsics have sideeffects. Each represent a single instruction, but
// workarounds are sometimes required depending on the cpu.

let TargetPrefix = "bfin" in {

  // Execute csync instruction with workarounds
  def int_bfin_csync : GCCBuiltin<"__builtin_bfin_csync">,
          Intrinsic<[llvm_void_ty]>;

  // Execute ssync instruction with workarounds
  def int_bfin_ssync : GCCBuiltin<"__builtin_bfin_ssync">,
          Intrinsic<[llvm_void_ty]>;

  // Execute idle instruction with workarounds
  def int_bfin_idle : GCCBuiltin<"__builtin_bfin_idle">,
          Intrinsic<[llvm_void_ty]>;

}
