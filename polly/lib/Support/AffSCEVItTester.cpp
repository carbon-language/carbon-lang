//===- AffSCEVItTester.cpp - Test the affine scev itertor. ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Test the affine scev itertor.
//
//===----------------------------------------------------------------------===//


#include "polly/Support/AffineSCEVIterator.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Support/InstIterator.h"
#include "llvm/Support/raw_ostream.h"

#include <map>

using namespace llvm;
using namespace polly;

static void printSCEVAffine(raw_ostream &OS, const SCEV* S,
                            ScalarEvolution *SE) {

  for (AffineSCEVIterator I = affine_begin(S, SE), E = affine_end();
    I != E; ++I) {
      OS << *I->second << " * " << *I->first;

      // The constant part of the SCEV will always be the last one.
      if (!isa<SCEVConstant>(S))
        OS << " + ";
  }
}

namespace {
struct AffSCEVItTester : public FunctionPass {
  static char ID;

  ScalarEvolution *SE;
  LoopInfo *LI;
  Function *F;

  explicit AffSCEVItTester() : FunctionPass(ID), SE(0), LI(0), F(0) {}

  virtual bool runOnFunction(Function &F) {
    SE = &getAnalysis<ScalarEvolution>();
    LI = &getAnalysis<LoopInfo>();
    this->F = &F;
    return false;
  }

  virtual void print(raw_ostream &OS, const Module *M) const {
    for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I)
      if (SE->isSCEVable(I->getType())) {
        OS << *I << '\n';
        OS << "  -->  ";
        const SCEV *SV = SE->getSCEV(&*I);

        if (Loop *L = LI->getLoopFor(I->getParent()))
          SV = SE->getSCEVAtScope(SV, L);
        SV->print(OS);
        OS << "\n";
        OS << "affine function  -->  ";
        printSCEVAffine(OS, SV, SE);
        OS << "\n";
      }

      for (LoopInfo::iterator I = LI->begin(), E = LI->end(); I != E; ++I)
        PrintLoopInfo(OS, *I);
  }

  void PrintLoopInfo(raw_ostream &OS, const Loop *L) const{
    // Print all inner loops first
    for (Loop::iterator I = L->begin(), E = L->end(); I != E; ++I)
      PrintLoopInfo(OS, *I);

    OS << "Loop ";
    WriteAsOperand(OS, L->getHeader(), /*PrintType=*/false);
    OS << ": ";

    if (SE->hasLoopInvariantBackedgeTakenCount(L)) {
      const SCEV *SV = SE->getBackedgeTakenCount(L);
      OS << "backedge-taken count is ";
      printSCEVAffine(OS, SV, SE);

      OS << "\nloop count in scev ";
      SV->print(OS);
      OS << "\n";
    }
    else {
      OS << "Unpredictable\n";
    }
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired<ScalarEvolution>();
    AU.addRequired<LoopInfo>();
    AU.setPreservesAll();
  }
};
} // end namespace


char AffSCEVItTester::ID = 0;

RegisterPass<AffSCEVItTester> B("print-scev-affine",
                                "Print the SCEV expressions in affine form.",
                                true,
                                true);

namespace polly {
Pass *createAffSCEVItTesterPass() {
  return new AffSCEVItTester();
}
}
