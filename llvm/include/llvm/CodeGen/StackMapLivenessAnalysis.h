//===--- StackMapLivenessAnalysis - StackMap Liveness Analysis --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass calculates the liveness for each basic block in a function and
// attaches the register live-out information to a patchpoint intrinsic (if
// present).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_STACKMAP_LIVENESS_ANALYSIS_H
#define LLVM_CODEGEN_STACKMAP_LIVENESS_ANALYSIS_H

#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunctionPass.h"


namespace llvm {

/// \brief This pass calculates the liveness information for each basic block in
/// a function and attaches the register live-out information to a patchpoint
/// intrinsic if present.
///
/// This pass can be disabled via the -enable-patchpoint-liveness=false flag.
/// The pass skips functions that don't have any patchpoint intrinsics. The
/// information provided by this pass is optional and not required by the
/// aformentioned intrinsic to function.
class StackMapLiveness : public MachineFunctionPass {
  MachineFunction *MF;
  const TargetRegisterInfo *TRI;
  LivePhysRegs LiveRegs;
public:
  static char ID;

  /// \brief Default construct and initialize the pass.
  StackMapLiveness();

  /// \brief Tell the pass manager which passes we depend on and what
  /// information we preserve.
  void getAnalysisUsage(AnalysisUsage &AU) const override;

  /// \brief Calculate the liveness information for the given machine function.
  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  /// \brief Performs the actual liveness calculation for the function.
  bool calculateLiveness();

  /// \brief Add the current register live set to the instruction.
  void addLiveOutSetToMI(MachineInstr &MI);

  /// \brief Create a register mask and initialize it with the registers from
  /// the register live set.
  uint32_t *createRegisterMask() const;
};

} // llvm namespace

#endif // LLVM_CODEGEN_STACKMAP_LIVENESS_ANALYSIS_H
