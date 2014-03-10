//===-- XCoreTargetTransformInfo.cpp - XCore specific TTI pass ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements a TargetTransformInfo analysis pass specific to the
/// XCore target machine. It uses the target's detailed information to provide
/// more precise answers to certain TTI queries, while letting the target
/// independent and default TTI implementations handle the rest.
///
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "xcoretti"
#include "XCore.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/CostTable.h"
#include "llvm/Target/TargetLowering.h"
using namespace llvm;

// Declare the pass initialization routine locally as target-specific passes
// don't havve a target-wide initialization entry point, and so we rely on the
// pass constructor initialization.
namespace llvm {
void initializeXCoreTTIPass(PassRegistry &);
}

namespace {

class XCoreTTI final : public ImmutablePass, public TargetTransformInfo {
public:
  XCoreTTI() : ImmutablePass(ID) {
    llvm_unreachable("This pass cannot be directly constructed");
  }

  XCoreTTI(const XCoreTargetMachine *TM)
      : ImmutablePass(ID) {
    initializeXCoreTTIPass(*PassRegistry::getPassRegistry());
  }

  virtual void initializePass() override {
    pushTTIStack(this);
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const override {
    TargetTransformInfo::getAnalysisUsage(AU);
  }

  static char ID;

  virtual void *getAdjustedAnalysisPointer(const void *ID) override {
    if (ID == &TargetTransformInfo::ID)
      return (TargetTransformInfo*)this;
    return this;
  }

  unsigned getNumberOfRegisters(bool Vector) const override {
    if (Vector) {
       return 0;
    }
    return 12;
  }
};

} // end anonymous namespace

INITIALIZE_AG_PASS(XCoreTTI, TargetTransformInfo, "xcoretti",
                   "XCore Target Transform Info", true, true, false)
char XCoreTTI::ID = 0;


ImmutablePass *
llvm::createXCoreTargetTransformInfoPass(const XCoreTargetMachine *TM) {
  return new XCoreTTI(TM);
}
