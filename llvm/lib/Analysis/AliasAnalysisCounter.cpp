//===- AliasAnalysisCounter.cpp - Alias Analysis Query Counter ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass which can be used to count how many alias queries
// are being made and how the alias analysis implementation being used responds.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/AliasAnalysisCounter.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

static cl::opt<bool> PrintAll("count-aa-print-all-queries", cl::ReallyHidden,
                              cl::init(true));
static cl::opt<bool> PrintAllFailures("count-aa-print-all-failed-queries",
                                      cl::ReallyHidden);

char AliasAnalysisCounter::ID = 0;
INITIALIZE_AG_PASS(AliasAnalysisCounter, AliasAnalysis, "count-aa",
                   "Count Alias Analysis Query Responses", false, true, false)

ModulePass *llvm::createAliasAnalysisCounterPass() {
  return new AliasAnalysisCounter();
}

AliasResult AliasAnalysisCounter::alias(const MemoryLocation &LocA,
                                        const MemoryLocation &LocB) {
  AliasResult R = getAnalysis<AliasAnalysis>().alias(LocA, LocB);

  const char *AliasString = nullptr;
  switch (R) {
  case NoAlias:
    No++;
    AliasString = "No alias";
    break;
  case MayAlias:
    May++;
    AliasString = "May alias";
    break;
  case PartialAlias:
    Partial++;
    AliasString = "Partial alias";
    break;
  case MustAlias:
    Must++;
    AliasString = "Must alias";
    break;
  }

  if (PrintAll || (PrintAllFailures && R == MayAlias)) {
    errs() << AliasString << ":\t";
    errs() << "[" << LocA.Size << "B] ";
    LocA.Ptr->printAsOperand(errs(), true, M);
    errs() << ", ";
    errs() << "[" << LocB.Size << "B] ";
    LocB.Ptr->printAsOperand(errs(), true, M);
    errs() << "\n";
  }

  return R;
}

ModRefInfo AliasAnalysisCounter::getModRefInfo(ImmutableCallSite CS,
                                               const MemoryLocation &Loc) {
  ModRefInfo R = getAnalysis<AliasAnalysis>().getModRefInfo(CS, Loc);

  const char *MRString = nullptr;
  switch (R) {
  case MRI_NoModRef:
    NoMR++;
    MRString = "MRI_NoModRef";
    break;
  case MRI_Ref:
    JustRef++;
    MRString = "JustRef";
    break;
  case MRI_Mod:
    JustMod++;
    MRString = "JustMod";
    break;
  case MRI_ModRef:
    MR++;
    MRString = "MRI_ModRef";
    break;
  }

  if (PrintAll || (PrintAllFailures && R == MRI_ModRef)) {
    errs() << MRString << ":  Ptr: ";
    errs() << "[" << Loc.Size << "B] ";
    Loc.Ptr->printAsOperand(errs(), true, M);
    errs() << "\t<->" << *CS.getInstruction() << '\n';
  }
  return R;
}
