//===- CorrelatedValuePropagation.cpp - Propagate CFG-derived info --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Correlated Value Propagation pass.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "correlated-value-propagation"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LazyValueInfo.h"
#include "llvm/Support/CFG.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

STATISTIC(NumPhis,      "Number of phis propagated");
STATISTIC(NumSelects,   "Number of selects propagated");
STATISTIC(NumMemAccess, "Number of memory access targets propagated");
STATISTIC(NumCmps,      "Number of comparisons propagated");

namespace {
  class CorrelatedValuePropagation : public FunctionPass {
    LazyValueInfo *LVI;

    bool processSelect(SelectInst *SI);
    bool processPHI(PHINode *P);
    bool processMemAccess(Instruction *I);
    bool processCmp(CmpInst *C);

  public:
    static char ID;
    CorrelatedValuePropagation(): FunctionPass(ID) {
     initializeCorrelatedValuePropagationPass(*PassRegistry::getPassRegistry());
    }

    bool runOnFunction(Function &F);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<LazyValueInfo>();
    }
  };
}

char CorrelatedValuePropagation::ID = 0;
INITIALIZE_PASS_BEGIN(CorrelatedValuePropagation, "correlated-propagation",
                "Value Propagation", false, false)
INITIALIZE_PASS_DEPENDENCY(LazyValueInfo)
INITIALIZE_PASS_END(CorrelatedValuePropagation, "correlated-propagation",
                "Value Propagation", false, false)

// Public interface to the Value Propagation pass
Pass *llvm::createCorrelatedValuePropagationPass() {
  return new CorrelatedValuePropagation();
}

bool CorrelatedValuePropagation::processSelect(SelectInst *S) {
  if (S->getType()->isVectorTy()) return false;
  if (isa<Constant>(S->getOperand(0))) return false;

  Constant *C = LVI->getConstant(S->getOperand(0), S->getParent());
  if (!C) return false;

  ConstantInt *CI = dyn_cast<ConstantInt>(C);
  if (!CI) return false;

  Value *ReplaceWith = S->getOperand(1);
  Value *Other = S->getOperand(2);
  if (!CI->isOne()) std::swap(ReplaceWith, Other);
  if (ReplaceWith == S) ReplaceWith = UndefValue::get(S->getType());

  S->replaceAllUsesWith(ReplaceWith);
  S->eraseFromParent();

  ++NumSelects;

  return true;
}

bool CorrelatedValuePropagation::processPHI(PHINode *P) {
  bool Changed = false;

  BasicBlock *BB = P->getParent();
  for (unsigned i = 0, e = P->getNumIncomingValues(); i < e; ++i) {
    Value *Incoming = P->getIncomingValue(i);
    if (isa<Constant>(Incoming)) continue;

    Constant *C = LVI->getConstantOnEdge(P->getIncomingValue(i),
                                         P->getIncomingBlock(i),
                                         BB);
    if (!C) continue;

    P->setIncomingValue(i, C);
    Changed = true;
  }

  if (Value *V = SimplifyInstruction(P)) {
    P->replaceAllUsesWith(V);
    P->eraseFromParent();
    Changed = true;
  }

  ++NumPhis;

  return Changed;
}

bool CorrelatedValuePropagation::processMemAccess(Instruction *I) {
  Value *Pointer = 0;
  if (LoadInst *L = dyn_cast<LoadInst>(I))
    Pointer = L->getPointerOperand();
  else
    Pointer = cast<StoreInst>(I)->getPointerOperand();

  if (isa<Constant>(Pointer)) return false;

  Constant *C = LVI->getConstant(Pointer, I->getParent());
  if (!C) return false;

  ++NumMemAccess;
  I->replaceUsesOfWith(Pointer, C);
  return true;
}

/// processCmp - If the value of this comparison could be determined locally,
/// constant propagation would already have figured it out.  Instead, walk
/// the predecessors and statically evaluate the comparison based on information
/// available on that edge.  If a given static evaluation is true on ALL
/// incoming edges, then it's true universally and we can simplify the compare.
bool CorrelatedValuePropagation::processCmp(CmpInst *C) {
  Value *Op0 = C->getOperand(0);
  if (isa<Instruction>(Op0) &&
      cast<Instruction>(Op0)->getParent() == C->getParent())
    return false;

  Constant *Op1 = dyn_cast<Constant>(C->getOperand(1));
  if (!Op1) return false;

  pred_iterator PI = pred_begin(C->getParent()), PE = pred_end(C->getParent());
  if (PI == PE) return false;

  LazyValueInfo::Tristate Result = LVI->getPredicateOnEdge(C->getPredicate(),
                                    C->getOperand(0), Op1, *PI, C->getParent());
  if (Result == LazyValueInfo::Unknown) return false;

  ++PI;
  while (PI != PE) {
    LazyValueInfo::Tristate Res = LVI->getPredicateOnEdge(C->getPredicate(),
                                    C->getOperand(0), Op1, *PI, C->getParent());
    if (Res != Result) return false;
    ++PI;
  }

  ++NumCmps;

  if (Result == LazyValueInfo::True)
    C->replaceAllUsesWith(ConstantInt::getTrue(C->getContext()));
  else
    C->replaceAllUsesWith(ConstantInt::getFalse(C->getContext()));

  C->eraseFromParent();

  return true;
}

bool CorrelatedValuePropagation::runOnFunction(Function &F) {
  LVI = &getAnalysis<LazyValueInfo>();

  bool FnChanged = false;

  for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI) {
    bool BBChanged = false;
    for (BasicBlock::iterator BI = FI->begin(), BE = FI->end(); BI != BE; ) {
      Instruction *II = BI++;
      switch (II->getOpcode()) {
      case Instruction::Select:
        BBChanged |= processSelect(cast<SelectInst>(II));
        break;
      case Instruction::PHI:
        BBChanged |= processPHI(cast<PHINode>(II));
        break;
      case Instruction::ICmp:
      case Instruction::FCmp:
        BBChanged |= processCmp(cast<CmpInst>(II));
        break;
      case Instruction::Load:
      case Instruction::Store:
        BBChanged |= processMemAccess(II);
        break;
      }
    }

    FnChanged |= BBChanged;
  }

  return FnChanged;
}
