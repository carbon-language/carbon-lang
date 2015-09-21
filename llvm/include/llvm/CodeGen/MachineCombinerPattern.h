//===-- llvm/CodeGen/MachineCombinerPattern.h - Instruction pattern supported by
// combiner  ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines instruction pattern supported by combiner
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINECOMBINERPATTERN_H
#define LLVM_CODEGEN_MACHINECOMBINERPATTERN_H

namespace llvm {

/// Enumeration of instruction pattern supported by machine combiner
///
///
namespace MachineCombinerPattern {
// Forward declaration
enum MC_PATTERN : int {
  // These are commutative variants for reassociating a computation chain. See
  // the comments before getMachineCombinerPatterns() in TargetInstrInfo.cpp.
  MC_REASSOC_AX_BY = 0,
  MC_REASSOC_AX_YB = 1,
  MC_REASSOC_XA_BY = 2,
  MC_REASSOC_XA_YB = 3,

  /// Enumeration of instruction pattern supported by AArch64 machine combiner
  MC_NONE,
  MC_MULADDW_OP1,
  MC_MULADDW_OP2,
  MC_MULSUBW_OP1,
  MC_MULSUBW_OP2,
  MC_MULADDWI_OP1,
  MC_MULSUBWI_OP1,
  MC_MULADDX_OP1,
  MC_MULADDX_OP2,
  MC_MULSUBX_OP1,
  MC_MULSUBX_OP2,
  MC_MULADDXI_OP1,
  MC_MULSUBXI_OP1
};
} // end namespace MachineCombinerPattern
} // end namespace llvm

#endif
