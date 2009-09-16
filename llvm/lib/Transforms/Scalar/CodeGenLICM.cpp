//===- CodeGenLICM.cpp - LICM a function for code generation --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This function performs late LICM, hoisting constants out of loops that
// are not valid immediates. It should not be followed by instcombine,
// because instcombine would quickly stuff the constants back into the loop.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "codegen-licm"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/LLVMContext.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/IVUsers.h"
#include "llvm/ADT/DenseMap.h"
using namespace llvm;

namespace {
  class CodeGenLICM : public LoopPass {
    virtual bool runOnLoop(Loop *L, LPPassManager &LPM);
    virtual void getAnalysisUsage(AnalysisUsage &AU) const;
  public:
    static char ID; // Pass identification, replacement for typeid
    explicit CodeGenLICM() : LoopPass(&ID) {}
  };
}

char CodeGenLICM::ID = 0;
static RegisterPass<CodeGenLICM> X("codegen-licm",
                                   "hoist constants out of loops");

Pass *llvm::createCodeGenLICMPass() {
  return new CodeGenLICM();
}

bool CodeGenLICM::runOnLoop(Loop *L, LPPassManager &) {
  bool Changed = false;

  // Only visit outermost loops.
  if (L->getParentLoop()) return Changed;

  Instruction *PreheaderTerm = L->getLoopPreheader()->getTerminator();
  DenseMap<Constant *, BitCastInst *> HoistedConstants;

  for (Loop::block_iterator I = L->block_begin(), E = L->block_end();
       I != E; ++I) {
    BasicBlock *BB = *I;
    for (BasicBlock::iterator BBI = BB->begin(), BBE = BB->end();
         BBI != BBE; ++BBI) {
      Instruction *I = BBI;
      // Don't bother hoisting constants out of loop-header phi nodes.
      if (BB == L->getHeader() && isa<PHINode>(I))
        continue;
      // TODO: For now, skip all intrinsic instructions, because some of them
      // can require their operands to be constants, and we don't want to
      // break that.
      if (isa<IntrinsicInst>(I))
        continue;
      // LLVM represents fneg as -0.0-x; don't hoist the -0.0 out.
      if (BinaryOperator::isFNeg(I) ||
          BinaryOperator::isNeg(I) ||
          BinaryOperator::isNot(I))
        continue;
      for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
        // Don't hoist out switch case constants.
        if (isa<SwitchInst>(I) && i == 1)
          break;
        // Don't hoist out shuffle masks.
        if (isa<ShuffleVectorInst>(I) && i == 2)
          break;
        Value *Op = I->getOperand(i);
        Constant *C = dyn_cast<Constant>(Op);
        if (!C) continue;
        // TODO: Ask the target which constants are legal. This would allow
        // us to add support for hoisting ConstantInts and GlobalValues too.
        if (isa<ConstantFP>(C) ||
            isa<ConstantVector>(C) ||
            isa<ConstantAggregateZero>(C)) {
          BitCastInst *&BC = HoistedConstants[C];
          if (!BC)
            BC = new BitCastInst(C, C->getType(), "hoist", PreheaderTerm);
          I->setOperand(i, BC);
          Changed = true;
        }
      }
    }
  }

  return Changed;
}

void CodeGenLICM::getAnalysisUsage(AnalysisUsage &AU) const {
  // This pass preserves just about everything. List some popular things here.
  AU.setPreservesCFG();
  AU.addPreservedID(LoopSimplifyID);
  AU.addPreserved<LoopInfo>();
  AU.addPreserved<AliasAnalysis>();
  AU.addPreserved<DominanceFrontier>();
  AU.addPreserved<DominatorTree>();
  AU.addPreserved<ScalarEvolution>();
  AU.addPreserved<IVUsers>();

  // Hoisting requires a loop preheader.
  AU.addRequiredID(LoopSimplifyID);
}
