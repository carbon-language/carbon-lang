//===- LowerAllocations.cpp - Reduce malloc & free insts to calls ---------===//
//
// The LowerAllocations transformation is a target dependent tranformation
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
FunctionPass *createLowerAllocationsPass() {
  return new LowerAllocations();
}


// doInitialization - For the lower allocations pass, this ensures that a
// module contains a declaration for a malloc and a free function.
//
// This function is always successful.
//
bool LowerAllocations::doInitialization(Module &M) {
  const FunctionType *MallocType = 
    FunctionType::get(PointerType::get(Type::SByteTy),
                      std::vector<const Type*>(1, Type::UIntTy), false);
  const FunctionType *FreeType = 
    FunctionType::get(Type::VoidTy,
                      std::vector<const Type*>(1,
                                               PointerType::get(Type::SByteTy)),
                      false);

  MallocFunc = M.getOrInsertFunction("malloc", MallocType);
  FreeFunc   = M.getOrInsertFunction("free"  , FreeType);

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
      } else if (MI->getNumOperands()) {
        // Multiply it by the array size if necessary...
        MallocArg = BinaryOperator::create(Instruction::Mul, MI->getOperand(0),
                                           MallocArg, "", I);
      }
      
      // Create the call to Malloc...
      CallInst *MCall = new CallInst(MallocFunc,
                                     std::vector<Value*>(1, MallocArg), "", I);
      
      // Create a cast instruction to convert to the right type...
      CastInst *MCast = new CastInst(MCall, MI->getType(), "", I);
      
      // Replace all uses of the old malloc inst with the cast inst
      MI->replaceAllUsesWith(MCast);
      I = --BBIL.erase(I);         // remove and delete the malloc instr...
      Changed = true;
      ++NumLowered;
    } else if (FreeInst *FI = dyn_cast<FreeInst>(I)) {
      // Cast the argument to free into a ubyte*...
      CastInst *MCast = new CastInst(FI->getOperand(0), 
                                     PointerType::get(Type::SByteTy), "", I);
      
      // Insert a call to the free function...
      CallInst *FCall = new CallInst(FreeFunc, std::vector<Value*>(1, MCast),
                                     "", I);
      
      // Delete the old free instruction
      I = --BBIL.erase(I);
      Changed = true;
      ++NumLowered;
    }
  }

  return Changed;
}
