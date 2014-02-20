//===- AArch64TargetTransformInfo.cpp - AArch64 specific TTI pass ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements a TargetTransformInfo analysis pass specific to the
/// AArch64 target machine. It uses the target's detailed information to provide
/// more precise answers to certain TTI queries, while letting the target
/// independent and default TTI implementations handle the rest.
///
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "aarch64tti"
#include "AArch64.h"
#include "AArch64TargetMachine.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/CostTable.h"
#include "llvm/Target/TargetLowering.h"
using namespace llvm;

// Declare the pass initialization routine locally as target-specific passes
// don't have a target-wide initialization entry point, and so we rely on the
// pass constructor initialization.
namespace llvm {
void initializeAArch64TTIPass(PassRegistry &);
}

namespace {

class AArch64TTI LLVM_FINAL : public ImmutablePass, public TargetTransformInfo {
  const AArch64TargetMachine *TM;
  const AArch64Subtarget *ST;
  const AArch64TargetLowering *TLI;

  /// Estimate the overhead of scalarizing an instruction. Insert and Extract
  /// are set if the result needs to be inserted and/or extracted from vectors.
  unsigned getScalarizationOverhead(Type *Ty, bool Insert, bool Extract) const;

public:
  AArch64TTI() : ImmutablePass(ID), TM(0), ST(0), TLI(0) {
    llvm_unreachable("This pass cannot be directly constructed");
  }

  AArch64TTI(const AArch64TargetMachine *TM)
      : ImmutablePass(ID), TM(TM), ST(TM->getSubtargetImpl()),
        TLI(TM->getTargetLowering()) {
    initializeAArch64TTIPass(*PassRegistry::getPassRegistry());
  }

  virtual void initializePass() LLVM_OVERRIDE {
    pushTTIStack(this);
  }

  virtual void finalizePass() {
    popTTIStack();
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const LLVM_OVERRIDE {
    TargetTransformInfo::getAnalysisUsage(AU);
  }

  /// Pass identification.
  static char ID;

  /// Provide necessary pointer adjustments for the two base classes.
  virtual void *getAdjustedAnalysisPointer(const void *ID) LLVM_OVERRIDE {
    if (ID == &TargetTransformInfo::ID)
      return (TargetTransformInfo*)this;
    return this;
  }

  /// \name Scalar TTI Implementations
  /// @{

  /// @}


  /// \name Vector TTI Implementations
  /// @{

  unsigned getNumberOfRegisters(bool Vector) const {
    if (Vector) {
      if (ST->hasNEON())
        return 32;
      return 0;
    }
    return 32;
  }

  unsigned getRegisterBitWidth(bool Vector) const {
    if (Vector) {
      if (ST->hasNEON())
        return 128;
      return 0;
    }
    return 64;
  }

  /// @}
};

} // end anonymous namespace

INITIALIZE_AG_PASS(AArch64TTI, TargetTransformInfo, "aarch64tti",
                   "AArch64 Target Transform Info", true, true, false)
char AArch64TTI::ID = 0;

ImmutablePass *
llvm::createAArch64TargetTransformInfoPass(const AArch64TargetMachine *TM) {
  return new AArch64TTI(TM);
}
