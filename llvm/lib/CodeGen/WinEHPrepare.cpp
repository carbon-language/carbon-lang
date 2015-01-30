//===-- WinEHPrepare - Prepare exception handling for code generation ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass lowers LLVM IR exception handling into something closer to what the
// backend wants. It snifs the personality function to see which kind of
// preparation is necessary. If the personality function uses the Itanium LSDA,
// this pass delegates to the DWARF EH preparation pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/Passes.h"
#include "llvm/Analysis/LibCallSemantics.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"
#include <memory>

using namespace llvm;

#define DEBUG_TYPE "winehprepare"

namespace {
class WinEHPrepare : public FunctionPass {
  std::unique_ptr<FunctionPass> DwarfPrepare;

public:
  static char ID; // Pass identification, replacement for typeid.
  WinEHPrepare(const TargetMachine *TM = nullptr)
      : FunctionPass(ID), DwarfPrepare(createDwarfEHPass(TM)) {}

  bool runOnFunction(Function &Fn) override;

  bool doFinalization(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  const char *getPassName() const override {
    return "Windows exception handling preparation";
  }
};
} // end anonymous namespace

char WinEHPrepare::ID = 0;
INITIALIZE_TM_PASS(WinEHPrepare, "winehprepare",
                   "Prepare Windows exceptions", false, false)

FunctionPass *llvm::createWinEHPass(const TargetMachine *TM) {
  return new WinEHPrepare(TM);
}

static bool isMSVCPersonality(EHPersonality Pers) {
  return Pers == EHPersonality::MSVC_Win64SEH ||
         Pers == EHPersonality::MSVC_CXX;
}

bool WinEHPrepare::runOnFunction(Function &Fn) {
  SmallVector<LandingPadInst *, 4> LPads;
  SmallVector<ResumeInst *, 4> Resumes;
  for (BasicBlock &BB : Fn) {
    if (auto *LP = BB.getLandingPadInst())
      LPads.push_back(LP);
    if (auto *Resume = dyn_cast<ResumeInst>(BB.getTerminator()))
      Resumes.push_back(Resume);
  }

  // No need to prepare functions that lack landing pads.
  if (LPads.empty())
    return false;

  // Classify the personality to see what kind of preparation we need.
  EHPersonality Pers = ClassifyEHPersonality(LPads.back()->getPersonalityFn());

  // Delegate through to the DWARF pass if this is unrecognized.
  if (!isMSVCPersonality(Pers))
    return DwarfPrepare->runOnFunction(Fn);

  // FIXME: Cleanups are unimplemented. Replace them with unreachable.
  if (Resumes.empty())
    return false;

  for (ResumeInst *Resume : Resumes) {
    IRBuilder<>(Resume).CreateUnreachable();
    Resume->eraseFromParent();
  }

  return true;
}

bool WinEHPrepare::doFinalization(Module &M) {
  return DwarfPrepare->doFinalization(M);
}

void WinEHPrepare::getAnalysisUsage(AnalysisUsage &AU) const {
  DwarfPrepare->getAnalysisUsage(AU);
}
