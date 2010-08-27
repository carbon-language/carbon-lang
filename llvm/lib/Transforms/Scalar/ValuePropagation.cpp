//===- ValuePropagation.cpp - Propagate information derived control flow --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Value Propagation pass.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "value-propagation"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/LazyValueInfo.h"
#include "llvm/Transforms/Utils/Local.h"
using namespace llvm;

namespace {
  class ValuePropagation : public FunctionPass {
    LazyValueInfo *LVI;
    
    bool processSelect(SelectInst *SI);
    bool processPHI(PHINode *P);
    
  public:
    static char ID;
    ValuePropagation(): FunctionPass(ID) { }
    
    bool runOnFunction(Function &F);
    
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<LazyValueInfo>();
    }
  };
}

char ValuePropagation::ID = 0;
INITIALIZE_PASS(ValuePropagation, "value-propagation",
                "Value Propagation", false, false);

// Public interface to the Value Propagation pass
Pass *llvm::createValuePropagationPass() {
  return new ValuePropagation();
}

bool ValuePropagation::processSelect(SelectInst *S) {
  Constant *C = LVI->getConstant(S->getOperand(0), S->getParent());
  if (!C) return false;
  
  ConstantInt *CI = dyn_cast<ConstantInt>(C);
  if (!CI) return false;
  
  if (CI->isZero()) {
    S->replaceAllUsesWith(S->getOperand(2));
    S->eraseFromParent();
  } else if (CI->isOne()) {
    S->replaceAllUsesWith(S->getOperand(1));
    S->eraseFromParent();
  } else {
    assert(0 && "Select on constant is neither 0 nor 1?");
  }
  
  return true;
}

bool ValuePropagation::processPHI(PHINode *P) {
  bool changed = false;
  
  BasicBlock *BB = P->getParent();
  for (unsigned i = 0; i < P->getNumIncomingValues(); ++i) {
    Constant *C = LVI->getConstantOnEdge(P->getIncomingValue(i),
                                         P->getIncomingBlock(i),
                                         BB);
    if (!C || C == P->getIncomingValue(i)) continue;
    
    P->setIncomingValue(i, C);
    changed = true;
  }
  
  if (Value *ConstVal = P->hasConstantValue()) {
    P->replaceAllUsesWith(ConstVal);
    P->eraseFromParent();
    changed = true;
  }
  
  return changed;
}

bool ValuePropagation::runOnFunction(Function &F) {
  LVI = &getAnalysis<LazyValueInfo>();
  
  bool changed = false;
  
  for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI)
    for (BasicBlock::iterator BI = FI->begin(), BE = FI->end(); BI != BE; ) {
      Instruction *II = BI++;
      if (SelectInst *SI = dyn_cast<SelectInst>(II))
        changed |= processSelect(SI);
      else if (PHINode *P = dyn_cast<PHINode>(II))
        changed |= processPHI(P);
    }
  
  if (changed)
    for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI)
      SimplifyInstructionsInBlock(FI);
  
  return changed;
}