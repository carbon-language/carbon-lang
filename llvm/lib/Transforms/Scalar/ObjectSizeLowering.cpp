//===-- ObjectSizeLowering.cpp - Loop unroller pass -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass lowers Intrinsic::objectsize using SCEV to determine minimum or
// maximum space left in an allocated object.
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "objsize-lower"
#include "llvm/Constants.h"
#include "llvm/Module.h"
#include "llvm/Value.h"
#include "llvm/Target/TargetData.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
  class ObjSizeLower : public FunctionPass {
    ScalarEvolution *SE;
    TargetData *TD;
  public:
    static char ID; // Pass identification, replacement for typeid
    ObjSizeLower() : FunctionPass(&ID) {}
    
    bool runOnFunction(Function &F);
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequired<ScalarEvolution>();
      AU.addPreserved<ScalarEvolution>();
    }
  private:
    bool LowerCall(IntrinsicInst *);
    void ReplaceAllUsesWithUnknown(IntrinsicInst *, bool);
  };
}

char ObjSizeLower::ID = 0;
static RegisterPass<ObjSizeLower> X("objsize-lower",
                                    "Object Size Lowering");

// Public interface to the Object Size Lowering pass
FunctionPass *llvm::createObjectSizeLoweringPass() { 
  return new ObjSizeLower();
}

/// runOnFunction - Top level algorithm - Loop over each object size intrinsic
/// and use Scalar Evolutions to get the maximum or minimum size left in the
/// allocated object at any point.
bool ObjSizeLower::runOnFunction(Function &F) {
  SE = &getAnalysis<ScalarEvolution>();
  TD = getAnalysisIfAvailable<TargetData>();
  
  // We really need TargetData for size calculations.
  if (!TD) return false;
  
  bool Changed = false;
  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB) {
    for (BasicBlock::iterator I = BB->begin(), L = BB->end(); I != L; ) {
      CallInst *CI = dyn_cast<CallInst>(I++);
      if (!CI) continue;

      // The only thing we care about are Intrinsic::objectsize calls
      IntrinsicInst *II = dyn_cast<IntrinsicInst>(CI);
      if (!II || II->getIntrinsicID() != Intrinsic::objectsize) continue;

      Changed |= LowerCall(II);
    }
  }
  return Changed;
}

// Unknown for llvm.objsize is -1 for maximum size, and 0 for minimum size.
void ObjSizeLower::ReplaceAllUsesWithUnknown(IntrinsicInst *II, bool min) {
  const Type *ReturnTy = II->getCalledFunction()->getReturnType();
  II->replaceAllUsesWith(ConstantInt::get(ReturnTy, min ? 0 : -1ULL));
  II->eraseFromParent();
}

bool ObjSizeLower::LowerCall(IntrinsicInst *II) {
  ConstantInt *CI = cast<ConstantInt>(II->getOperand(2));
  bool minimum = (CI->getZExtValue() == 1);
  Value *Op = II->getOperand(1);
  const Type *ReturnTy = II->getCalledFunction()->getReturnType();

  // Grab the SCEV for our access.
  const SCEV *thisEV = SE->getSCEV(Op);

  if (const SCEVUnknown *SU = dyn_cast<SCEVUnknown>(thisEV)) {
    if (GlobalVariable *GV = dyn_cast<GlobalVariable>(SU->getValue())) {
      if (GV->hasDefinitiveInitializer()) {
        Constant *C = GV->getInitializer();
        size_t globalSize = TD->getTypeAllocSize(C->getType());
        II->replaceAllUsesWith(ConstantInt::get(ReturnTy, globalSize));
        II->eraseFromParent();
        return true;
      }
    }
  }

  ReplaceAllUsesWithUnknown(II, minimum);
  return true;
}
