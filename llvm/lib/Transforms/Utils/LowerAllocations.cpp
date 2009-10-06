//===- LowerAllocations.cpp - Reduce malloc & free insts to calls ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The LowerAllocations transformation is a target-dependent tranformation
// because it depends on the size of data types and alignment constraints.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "lowerallocs"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/UnifyFunctionExitNodes.h"
#include "llvm/Module.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Constants.h"
#include "llvm/LLVMContext.h"
#include "llvm/Pass.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Support/Compiler.h"
using namespace llvm;

STATISTIC(NumLowered, "Number of allocations lowered");

namespace {
  /// LowerAllocations - Turn malloc and free instructions into @malloc and
  /// @free calls.
  ///
  class VISIBILITY_HIDDEN LowerAllocations : public BasicBlockPass {
    Constant *FreeFunc;   // Functions in the module we are processing
                          // Initialized by doInitialization
    bool LowerMallocArgToInteger;
  public:
    static char ID; // Pass ID, replacement for typeid
    explicit LowerAllocations(bool LowerToInt = false)
      : BasicBlockPass(&ID), FreeFunc(0), 
        LowerMallocArgToInteger(LowerToInt) {}

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<TargetData>();
      AU.setPreservesCFG();

      // This is a cluster of orthogonal Transforms:
      AU.addPreserved<UnifyFunctionExitNodes>();
      AU.addPreservedID(PromoteMemoryToRegisterID);
      AU.addPreservedID(LowerSwitchID);
      AU.addPreservedID(LowerInvokePassID);
    }

    /// doPassInitialization - For the lower allocations pass, this ensures that
    /// a module contains a declaration for a malloc and a free function.
    ///
    bool doInitialization(Module &M);

    virtual bool doInitialization(Function &F) {
      return doInitialization(*F.getParent());
    }

    /// runOnBasicBlock - This method does the actual work of converting
    /// instructions over, assuming that the pass has already been initialized.
    ///
    bool runOnBasicBlock(BasicBlock &BB);
  };
}

char LowerAllocations::ID = 0;
static RegisterPass<LowerAllocations>
X("lowerallocs", "Lower allocations from instructions to calls");

// Publically exposed interface to pass...
const PassInfo *const llvm::LowerAllocationsID = &X;
// createLowerAllocationsPass - Interface to this file...
Pass *llvm::createLowerAllocationsPass(bool LowerMallocArgToInteger) {
  return new LowerAllocations(LowerMallocArgToInteger);
}


// doInitialization - For the lower allocations pass, this ensures that a
// module contains a declaration for a malloc and a free function.
//
// This function is always successful.
//
bool LowerAllocations::doInitialization(Module &M) {
  const Type *BPTy = Type::getInt8PtrTy(M.getContext());
  FreeFunc = M.getOrInsertFunction("free"  , Type::getVoidTy(M.getContext()),
                                   BPTy, (Type *)0);
  return true;
}

// runOnBasicBlock - This method does the actual work of converting
// instructions over, assuming that the pass has already been initialized.
//
bool LowerAllocations::runOnBasicBlock(BasicBlock &BB) {
  bool Changed = false;
  assert(FreeFunc && "Pass not initialized!");

  BasicBlock::InstListType &BBIL = BB.getInstList();

  const TargetData &TD = getAnalysis<TargetData>();
  const Type *IntPtrTy = TD.getIntPtrType(BB.getContext());

  // Loop over all of the instructions, looking for malloc or free instructions
  for (BasicBlock::iterator I = BB.begin(), E = BB.end(); I != E; ++I) {
    if (MallocInst *MI = dyn_cast<MallocInst>(I)) {
      Value *ArraySize = MI->getOperand(0);
      if (ArraySize->getType() != IntPtrTy)
        ArraySize = CastInst::CreateIntegerCast(ArraySize, IntPtrTy,
                                                false /*ZExt*/, "", I);
      Value *MCast = CallInst::CreateMalloc(I, IntPtrTy,
                                            MI->getAllocatedType(), ArraySize);

      // Replace all uses of the old malloc inst with the cast inst
      MI->replaceAllUsesWith(MCast);
      I = --BBIL.erase(I);         // remove and delete the malloc instr...
      Changed = true;
      ++NumLowered;
    } else if (FreeInst *FI = dyn_cast<FreeInst>(I)) {
      Value *PtrCast = 
        new BitCastInst(FI->getOperand(0),
               Type::getInt8PtrTy(BB.getContext()), "", I);

      // Insert a call to the free function...
      CallInst::Create(FreeFunc, PtrCast, "", I)->setTailCall();

      // Delete the old free instruction
      I = --BBIL.erase(I);
      Changed = true;
      ++NumLowered;
    }
  }

  return Changed;
}

