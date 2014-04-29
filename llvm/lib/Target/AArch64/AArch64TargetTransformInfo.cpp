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

#include "AArch64.h"
#include "AArch64TargetMachine.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/CostTable.h"
#include "llvm/Target/TargetLowering.h"
using namespace llvm;

#define DEBUG_TYPE "aarch64tti"

// Declare the pass initialization routine locally as target-specific passes
// don't have a target-wide initialization entry point, and so we rely on the
// pass constructor initialization.
namespace llvm {
void initializeAArch64TTIPass(PassRegistry &);
}

namespace {

class AArch64TTI final : public ImmutablePass, public TargetTransformInfo {
  const AArch64Subtarget *ST;
  const AArch64TargetLowering *TLI;

public:
  AArch64TTI() : ImmutablePass(ID), ST(nullptr), TLI(nullptr) {
    llvm_unreachable("This pass cannot be directly constructed");
  }

  AArch64TTI(const AArch64TargetMachine *TM)
      : ImmutablePass(ID), ST(TM->getSubtargetImpl()),
        TLI(TM->getTargetLowering()) {
    initializeAArch64TTIPass(*PassRegistry::getPassRegistry());
  }

  virtual void initializePass() override {
    pushTTIStack(this);
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const override {
    TargetTransformInfo::getAnalysisUsage(AU);
  }

  /// Pass identification.
  static char ID;

  /// Provide necessary pointer adjustments for the two base classes.
  virtual void *getAdjustedAnalysisPointer(const void *ID) override {
    if (ID == &TargetTransformInfo::ID)
      return (TargetTransformInfo*)this;
    return this;
  }

  /// \name Scalar TTI Implementations
  /// @{

  /// @}


  /// \name Vector TTI Implementations
  /// @{

  unsigned getNumberOfRegisters(bool Vector) const override {
    if (Vector) {
      if (ST->hasNEON())
        return 32;
      return 0;
    }
    return 32;
  }

  unsigned getRegisterBitWidth(bool Vector) const override {
    if (Vector) {
      if (ST->hasNEON())
        return 128;
      return 0;
    }
    return 64;
  }

  unsigned getMaximumUnrollFactor() const override { return 2; }
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
