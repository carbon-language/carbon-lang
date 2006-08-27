//===- LowerAllocations.cpp - Reduce malloc & free insts to calls ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The LowerAllocations transformation is a target-dependent tranformation
// because it depends on the size of data types and alignment constraints.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/UnifyFunctionExitNodes.h"
#include "llvm/Module.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Constants.h"
#include "llvm/Pass.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Support/Compiler.h"
using namespace llvm;

namespace {
  Statistic<> NumLowered("lowerallocs", "Number of allocations lowered");

  /// LowerAllocations - Turn malloc and free instructions into %malloc and
  /// %free calls.
  ///
  class VISIBILITY_HIDDEN LowerAllocations : public BasicBlockPass {
    Function *MallocFunc;   // Functions in the module we are processing
    Function *FreeFunc;     // Initialized by doInitialization
    bool LowerMallocArgToInteger;
  public:
    LowerAllocations(bool LowerToInt = false)
      : MallocFunc(0), FreeFunc(0), LowerMallocArgToInteger(LowerToInt) {}

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<TargetData>();
      AU.setPreservesCFG();

      // This is a cluster of orthogonal Transforms:	
      AU.addPreserved<UnifyFunctionExitNodes>();
      AU.addPreservedID(PromoteMemoryToRegisterID);
      AU.addPreservedID(LowerSelectID);
      AU.addPreservedID(LowerSwitchID);
      AU.addPreservedID(LowerInvokePassID);
    }

    /// doPassInitialization - For the lower allocations pass, this ensures that
    /// a module contains a declaration for a malloc and a free function.
    ///
    bool doInitialization(Module &M);

    virtual bool doInitialization(Function &F) {
      return BasicBlockPass::doInitialization(F);
    }

    /// runOnBasicBlock - This method does the actual work of converting
    /// instructions over, assuming that the pass has already been initialized.
    ///
    bool runOnBasicBlock(BasicBlock &BB);
  };

  RegisterPass<LowerAllocations>
  X("lowerallocs", "Lower allocations from instructions to calls");
}

// Publically exposed interface to pass...
const PassInfo *llvm::LowerAllocationsID = X.getPassInfo();
// createLowerAllocationsPass - Interface to this file...
FunctionPass *llvm::createLowerAllocationsPass(bool LowerMallocArgToInteger) {
  return new LowerAllocations(LowerMallocArgToInteger);
}


// doInitialization - For the lower allocations pass, this ensures that a
// module contains a declaration for a malloc and a free function.
//
// This function is always successful.
//
bool LowerAllocations::doInitialization(Module &M) {
  const Type *SBPTy = PointerType::get(Type::SByteTy);
  MallocFunc = M.getNamedFunction("malloc");
  FreeFunc   = M.getNamedFunction("free");

  if (MallocFunc == 0) {
    // Prototype malloc as "void* malloc(...)", because we don't know in
    // doInitialization whether size_t is int or long.
    FunctionType *FT = FunctionType::get(SBPTy,std::vector<const Type*>(),true);
    MallocFunc = M.getOrInsertFunction("malloc", FT);
  }
  if (FreeFunc == 0)
    FreeFunc = M.getOrInsertFunction("free"  , Type::VoidTy, SBPTy, (Type *)0);

  return true;
}

// runOnBasicBlock - This method does the actual work of converting
// instructions over, assuming that the pass has already been initialized.
//
bool LowerAllocations::runOnBasicBlock(BasicBlock &BB) {
  bool Changed = false;
  assert(MallocFunc && FreeFunc && "Pass not initialized!");

  BasicBlock::InstListType &BBIL = BB.getInstList();

  const TargetData &TD = getAnalysis<TargetData>();
  const Type *IntPtrTy = TD.getIntPtrType();

  // Loop over all of the instructions, looking for malloc or free instructions
  for (BasicBlock::iterator I = BB.begin(), E = BB.end(); I != E; ++I) {
    if (MallocInst *MI = dyn_cast<MallocInst>(I)) {
      const Type *AllocTy = MI->getType()->getElementType();

      // malloc(type) becomes sbyte *malloc(size)
      Value *MallocArg;
      if (LowerMallocArgToInteger)
        MallocArg = ConstantUInt::get(Type::ULongTy, TD.getTypeSize(AllocTy));
      else
        MallocArg = ConstantExpr::getSizeOf(AllocTy);
      MallocArg = ConstantExpr::getCast(cast<Constant>(MallocArg), IntPtrTy);

      if (MI->isArrayAllocation()) {
        if (isa<ConstantInt>(MallocArg) &&
            cast<ConstantInt>(MallocArg)->getRawValue() == 1) {
          MallocArg = MI->getOperand(0);         // Operand * 1 = Operand
        } else if (Constant *CO = dyn_cast<Constant>(MI->getOperand(0))) {
          CO = ConstantExpr::getCast(CO, IntPtrTy);
          MallocArg = ConstantExpr::getMul(CO, cast<Constant>(MallocArg));
        } else {
          Value *Scale = MI->getOperand(0);
          if (Scale->getType() != IntPtrTy)
            Scale = new CastInst(Scale, IntPtrTy, "", I);

          // Multiply it by the array size if necessary...
          MallocArg = BinaryOperator::create(Instruction::Mul, Scale,
                                             MallocArg, "", I);
        }
      }

      const FunctionType *MallocFTy = MallocFunc->getFunctionType();
      std::vector<Value*> MallocArgs;

      if (MallocFTy->getNumParams() > 0 || MallocFTy->isVarArg()) {
        if (MallocFTy->isVarArg()) {
          if (MallocArg->getType() != IntPtrTy)
            MallocArg = new CastInst(MallocArg, IntPtrTy, "", I);
        } else if (MallocFTy->getNumParams() > 0 &&
                   MallocFTy->getParamType(0) != Type::UIntTy)
          MallocArg = new CastInst(MallocArg, MallocFTy->getParamType(0), "",I);
        MallocArgs.push_back(MallocArg);
      }

      // If malloc is prototyped to take extra arguments, pass nulls.
      for (unsigned i = 1; i < MallocFTy->getNumParams(); ++i)
       MallocArgs.push_back(Constant::getNullValue(MallocFTy->getParamType(i)));

      // Create the call to Malloc...
      CallInst *MCall = new CallInst(MallocFunc, MallocArgs, "", I);
      MCall->setTailCall();

      // Create a cast instruction to convert to the right type...
      Value *MCast;
      if (MCall->getType() != Type::VoidTy)
        MCast = new CastInst(MCall, MI->getType(), "", I);
      else
        MCast = Constant::getNullValue(MI->getType());

      // Replace all uses of the old malloc inst with the cast inst
      MI->replaceAllUsesWith(MCast);
      I = --BBIL.erase(I);         // remove and delete the malloc instr...
      Changed = true;
      ++NumLowered;
    } else if (FreeInst *FI = dyn_cast<FreeInst>(I)) {
      const FunctionType *FreeFTy = FreeFunc->getFunctionType();
      std::vector<Value*> FreeArgs;

      if (FreeFTy->getNumParams() > 0 || FreeFTy->isVarArg()) {
        Value *MCast = FI->getOperand(0);
        if (FreeFTy->getNumParams() > 0 &&
            FreeFTy->getParamType(0) != MCast->getType())
          MCast = new CastInst(MCast, FreeFTy->getParamType(0), "", I);
        FreeArgs.push_back(MCast);
      }

      // If malloc is prototyped to take extra arguments, pass nulls.
      for (unsigned i = 1; i < FreeFTy->getNumParams(); ++i)
       FreeArgs.push_back(Constant::getNullValue(FreeFTy->getParamType(i)));

      // Insert a call to the free function...
      (new CallInst(FreeFunc, FreeArgs, "", I))->setTailCall();

      // Delete the old free instruction
      I = --BBIL.erase(I);
      Changed = true;
      ++NumLowered;
    }
  }

  return Changed;
}

