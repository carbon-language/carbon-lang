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
#include "llvm/Module.h"
#include "llvm/DerivedTypes.h"
#include "llvm/iMemory.h"
#include "llvm/iOther.h"
#include "llvm/Constants.h"
#include "llvm/Pass.h"
#include "llvm/Target/TargetData.h"
#include "Support/Statistic.h"
using namespace llvm;

namespace {
  Statistic<> NumLowered("lowerallocs", "Number of allocations lowered");

  /// LowerAllocations - Turn malloc and free instructions into %malloc and
  /// %free calls.
  ///
  class LowerAllocations : public BasicBlockPass {
    Function *MallocFunc;   // Functions in the module we are processing
    Function *FreeFunc;     // Initialized by doInitialization
  public:
    LowerAllocations() : MallocFunc(0), FreeFunc(0) {}

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<TargetData>();
    }

    /// doPassInitialization - For the lower allocations pass, this ensures that
    /// a module contains a declaration for a malloc and a free function.
    ///
    bool doInitialization(Module &M);
    
    /// runOnBasicBlock - This method does the actual work of converting
    /// instructions over, assuming that the pass has already been initialized.
    ///
    bool runOnBasicBlock(BasicBlock &BB);
  };

  RegisterOpt<LowerAllocations>
  X("lowerallocs", "Lower allocations from instructions to calls");
}

// createLowerAllocationsPass - Interface to this file...
FunctionPass *llvm::createLowerAllocationsPass() {
  return new LowerAllocations();
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

  if (MallocFunc == 0)
    MallocFunc = M.getOrInsertFunction("malloc", SBPTy, Type::UIntTy, 0);
  if (FreeFunc == 0)
    FreeFunc   = M.getOrInsertFunction("free"  , Type::VoidTy, SBPTy, 0);

  return true;
}

// runOnBasicBlock - This method does the actual work of converting
// instructions over, assuming that the pass has already been initialized.
//
bool LowerAllocations::runOnBasicBlock(BasicBlock &BB) {
  bool Changed = false;
  assert(MallocFunc && FreeFunc && "Pass not initialized!");

  BasicBlock::InstListType &BBIL = BB.getInstList();
  TargetData &DataLayout = getAnalysis<TargetData>();

  // Loop over all of the instructions, looking for malloc or free instructions
  for (BasicBlock::iterator I = BB.begin(), E = BB.end(); I != E; ++I) {
    if (MallocInst *MI = dyn_cast<MallocInst>(I)) {
      const Type *AllocTy = MI->getType()->getElementType();
      
      // Get the number of bytes to be allocated for one element of the
      // requested type...
      unsigned Size = DataLayout.getTypeSize(AllocTy);
      
      // malloc(type) becomes sbyte *malloc(constint)
      Value *MallocArg = ConstantUInt::get(Type::UIntTy, Size);
      if (MI->getNumOperands() && Size == 1) {
        MallocArg = MI->getOperand(0);         // Operand * 1 = Operand
      } else if (MI->isArrayAllocation()) {
        // Multiply it by the array size if necessary...
        MallocArg = BinaryOperator::create(Instruction::Mul, MI->getOperand(0),
                                           MallocArg, "", I);
      }

      const FunctionType *MallocFTy = MallocFunc->getFunctionType();
      std::vector<Value*> MallocArgs;
      
      if (MallocFTy->getNumParams() > 0 || MallocFTy->isVarArg()) {
        if (MallocFTy->getNumParams() > 0 &&
            MallocFTy->getParamType(0) != Type::UIntTy)
          MallocArg = new CastInst(MallocArg, MallocFTy->getParamType(0), "",I);
        MallocArgs.push_back(MallocArg);
      }

      // If malloc is prototyped to take extra arguments, pass nulls.
      for (unsigned i = 1; i < MallocFTy->getNumParams(); ++i)
       MallocArgs.push_back(Constant::getNullValue(MallocFTy->getParamType(i)));

      // Create the call to Malloc...
      CallInst *MCall = new CallInst(MallocFunc, MallocArgs, "", I);
      
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
      new CallInst(FreeFunc, FreeArgs, "", I);
      
      // Delete the old free instruction
      I = --BBIL.erase(I);
      Changed = true;
      ++NumLowered;
    }
  }

  return Changed;
}

