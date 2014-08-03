//===- AArch64MachineCombinerPattern.h                                    -===//
//===- AArch64 instruction pattern supported by combiner                  -===//
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

#ifndef LLVM_TARGET_AArch64MACHINECOMBINERPATTERN_H
#define LLVM_TARGET_AArch64MACHINECOMBINERPATTERN_H

namespace llvm {

/// Enumeration of instruction pattern supported by machine combiner
///
///
namespace MachineCombinerPattern {
enum MC_PATTERN : int {
  MC_NONE = 0,
  MC_MULADDW_OP1 = 1,
  MC_MULADDW_OP2 = 2,
  MC_MULSUBW_OP1 = 3,
  MC_MULSUBW_OP2 = 4,
  MC_MULADDWI_OP1 = 5,
  MC_MULSUBWI_OP1 = 6,
  MC_MULADDX_OP1 = 7,
  MC_MULADDX_OP2 = 8,
  MC_MULSUBX_OP1 = 9,
  MC_MULSUBX_OP2 = 10,
  MC_MULADDXI_OP1 = 11,
  MC_MULSUBXI_OP1 = 12
};
} // end namespace MachineCombinerPattern
} // end namespace llvm

#endif
