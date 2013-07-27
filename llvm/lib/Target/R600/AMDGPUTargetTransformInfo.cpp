//===-- AMDGPUTargetTransformInfo.cpp - AMDGPU specific TTI pass ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// \file
// This file implements a TargetTransformInfo analysis pass specific to the
// AMDGPU target machine. It uses the target's detailed information to provide
// more precise answers to certain TTI queries, while letting the target
// independent and default TTI implementations handle the rest.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "AMDGPUtti"
#include "AMDGPU.h"
#include "AMDGPUTargetMachine.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/CostTable.h"
using namespace llvm;

// Declare the pass initialization routine locally as target-specific passes
// don't have a target-wide initialization entry point, and so we rely on the
// pass constructor initialization.
namespace llvm {
void initializeAMDGPUTTIPass(PassRegistry &);
}

namespace {

class AMDGPUTTI : public ImmutablePass, public TargetTransformInfo {
  const AMDGPUTargetMachine *TM;
  const AMDGPUSubtarget *ST;
  const AMDGPUTargetLowering *TLI;

  /// Estimate the overhead of scalarizing an instruction. Insert and Extract
  /// are set if the result needs to be inserted and/or extracted from vectors.
  unsigned getScalarizationOverhead(Type *Ty, bool Insert, bool Extract) const;

public:
  AMDGPUTTI() : ImmutablePass(ID), TM(0), ST(0), TLI(0) {
    llvm_unreachable("This pass cannot be directly constructed");
  }

  AMDGPUTTI(const AMDGPUTargetMachine *TM)
      : ImmutablePass(ID), TM(TM), ST(TM->getSubtargetImpl()),
        TLI(TM->getTargetLowering()) {
    initializeAMDGPUTTIPass(*PassRegistry::getPassRegistry());
  }

  virtual void initializePass() { pushTTIStack(this); }

  virtual void finalizePass() { popTTIStack(); }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    TargetTransformInfo::getAnalysisUsage(AU);
  }

  /// Pass identification.
  static char ID;

  /// Provide necessary pointer adjustments for the two base classes.
  virtual void *getAdjustedAnalysisPointer(const void *ID) {
    if (ID == &TargetTransformInfo::ID)
      return (TargetTransformInfo *)this;
    return this;
  }

  virtual bool hasBranchDivergence() const;

  /// @}
};

} // end anonymous namespace

INITIALIZE_AG_PASS(AMDGPUTTI, TargetTransformInfo, "AMDGPUtti",
                   "AMDGPU Target Transform Info", true, true, false)
char AMDGPUTTI::ID = 0;

ImmutablePass *
llvm::createAMDGPUTargetTransformInfoPass(const AMDGPUTargetMachine *TM) {
  return new AMDGPUTTI(TM);
}

bool AMDGPUTTI::hasBranchDivergence() const { return true; }
