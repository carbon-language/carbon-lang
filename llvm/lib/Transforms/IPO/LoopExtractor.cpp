//===- LoopExtractor.cpp - Extract each loop into a new function ----------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// A pass wrapper around the ExtractLoop() scalar transformation to extract each
// top-level loop into its own new function. If the loop is the ONLY loop in a
// given function, it is not touched. This is a pass most useful for debugging
// via bugpoint.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/FunctionUtils.h"
using namespace llvm;

namespace {
  // FIXME: PassManager should allow Module passes to require FunctionPasses
  struct LoopExtractor : public FunctionPass {
    virtual bool runOnFunction(Function &F);
    
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<LoopInfo>();
      AU.addRequiredID(LoopSimplifyID);
    }
  };

  RegisterOpt<LoopExtractor> 
  X("loop-extract", "Extract loops into new functions");
} // End anonymous namespace 

bool LoopExtractor::runOnFunction(Function &F) {
  LoopInfo &LI = getAnalysis<LoopInfo>();

  // We don't want to keep extracting the only loop of a function into a new one
  if (LI.begin() == LI.end() || LI.begin() + 1 == LI.end())
    return false;

  bool Changed = false;

  // Try to move each loop out of the code into separate function
  for (LoopInfo::iterator i = LI.begin(), e = LI.end(); i != e; ++i)
    Changed |= (ExtractLoop(*i) != 0);

  return Changed;
}

/// createLoopExtractorPass 
///
Pass* llvm::createLoopExtractorPass() {
  return new LoopExtractor();
}
