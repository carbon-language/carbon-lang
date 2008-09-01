//===--------- MarkModRef.cpp - Mark functions readnone/readonly ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass marks functions readnone/readonly based on the results of alias
// analysis.  This requires a sufficiently powerful alias analysis, such as
// GlobalsModRef (invoke as "opt ... -globalsmodref-aa -markmodref ...").
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "markmodref"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Function.h"
#include "llvm/Pass.h"
using namespace llvm;

STATISTIC(NumReadNone, "Number of functions marked readnone");
STATISTIC(NumReadOnly, "Number of functions marked readonly");

namespace {
  struct VISIBILITY_HIDDEN MarkModRef : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    MarkModRef() : FunctionPass((intptr_t)&ID) {}

    bool runOnFunction(Function &F);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequired<AliasAnalysis>();
      AU.addPreserved<AliasAnalysis>();
    }
  };
}

char MarkModRef::ID = 0;
static RegisterPass<MarkModRef>
X("markmodref", "Mark functions readnone/readonly");

bool MarkModRef::runOnFunction(Function &F) {
  // FIXME: Wrong for functions with weak linkage.
  if (F.doesNotAccessMemory())
    // Cannot do better.
    return false;

  AliasAnalysis &AA = getAnalysis<AliasAnalysis>();
  AliasAnalysis::ModRefBehavior ModRef = AA.getModRefBehavior(&F);
  if (ModRef == AliasAnalysis::DoesNotAccessMemory) {
    F.setDoesNotAccessMemory();
    NumReadNone++;
    return true;
  } else if (ModRef == AliasAnalysis::OnlyReadsMemory && !F.onlyReadsMemory()) {
    F.setOnlyReadsMemory();
    NumReadOnly++;
    return true;
  }
  return false;
}

FunctionPass *llvm::createMarkModRefPass() {
  return new MarkModRef();
}
