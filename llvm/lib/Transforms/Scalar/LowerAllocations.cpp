//===- ChangeAllocations.cpp - Modify %malloc & %free calls -----------------=//
//
// This file defines two passes that convert malloc and free instructions to
// calls to and from %malloc & %free function calls.  The LowerAllocations
// transformation is a target dependant tranformation because it depends on the
// size of data types and alignment constraints.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/ChangeAllocations.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Module.h"
#include "llvm/Function.h"
#include "llvm/DerivedTypes.h"
#include "llvm/iMemory.h"
#include "llvm/iOther.h"
#include "llvm/Constants.h"
#include "llvm/Pass.h"
#include "TransformInternals.h"
using std::vector;

namespace {

// LowerAllocations - Turn malloc and free instructions into %malloc and %free
// calls.
//
class LowerAllocations : public BasicBlockPass {
  Function *MallocFunc;   // Functions in the module we are processing
  Function *FreeFunc;     // Initialized by doInitialization

  const TargetData &DataLayout;
public:
  inline LowerAllocations(const TargetData &TD) : DataLayout(TD) {
    MallocFunc = FreeFunc = 0;
  }

  // doPassInitialization - For the lower allocations pass, this ensures that a
  // module contains a declaration for a malloc and a free function.
  //
  bool doInitialization(Module *M);

  // runOnBasicBlock - This method does the actual work of converting
  // instructions over, assuming that the pass has already been initialized.
  //
  bool runOnBasicBlock(BasicBlock *BB);
};

// RaiseAllocations - Turn %malloc and %free calls into the appropriate
// instruction.
//
class RaiseAllocations : public BasicBlockPass {
  Function *MallocFunc;   // Functions in the module we are processing
  Function *FreeFunc;     // Initialized by doPassInitializationVirt
public:
  inline RaiseAllocations() : MallocFunc(0), FreeFunc(0) {}

  // doPassInitialization - For the raise allocations pass, this finds a
  // declaration for malloc and free if they exist.
  //
  bool doInitialization(Module *M);

  // runOnBasicBlock - This method does the actual work of converting
  // instructions over, assuming that the pass has already been initialized.
  //
  bool runOnBasicBlock(BasicBlock *BB);
};

}  // end anonymous namespace

// doInitialization - For the lower allocations pass, this ensures that a
// module contains a declaration for a malloc and a free function.
//
// This function is always successful.
//
bool LowerAllocations::doInitialization(Module *M) {
  const FunctionType *MallocType = 
    FunctionType::get(PointerType::get(Type::SByteTy),
                      vector<const Type*>(1, Type::UIntTy), false);
  const FunctionType *FreeType = 
    FunctionType::get(Type::VoidTy,
                      vector<const Type*>(1, PointerType::get(Type::SByteTy)),
                      false);

  MallocFunc = M->getOrInsertFunction("malloc", MallocType);
  FreeFunc   = M->getOrInsertFunction("free"  , FreeType);

  return false;
}

// runOnBasicBlock - This method does the actual work of converting
// instructions over, assuming that the pass has already been initialized.
//
bool LowerAllocations::runOnBasicBlock(BasicBlock *BB) {
  bool Changed = false;
  assert(MallocFunc && FreeFunc && BB && "Pass not initialized!");

  // Loop over all of the instructions, looking for malloc or free instructions
  for (unsigned i = 0; i < BB->size(); ++i) {
    BasicBlock::InstListType &BBIL = BB->getInstList();
    if (MallocInst *MI = dyn_cast<MallocInst>(*(BBIL.begin()+i))) {
      BBIL.remove(BBIL.begin()+i);   // remove the malloc instr...
        
      const Type *AllocTy =cast<PointerType>(MI->getType())->getElementType();
      
      // Get the number of bytes to be allocated for one element of the
      // requested type...
      unsigned Size = DataLayout.getTypeSize(AllocTy);
      
      // malloc(type) becomes sbyte *malloc(constint)
      Value *MallocArg = ConstantUInt::get(Type::UIntTy, Size);
      if (MI->getNumOperands() && Size == 1) {
        MallocArg = MI->getOperand(0);         // Operand * 1 = Operand
      } else if (MI->getNumOperands()) {
        // Multiply it by the array size if neccesary...
        MallocArg = BinaryOperator::create(Instruction::Mul,MI->getOperand(0),
                                           MallocArg);
        BBIL.insert(BBIL.begin()+i++, cast<Instruction>(MallocArg));
      }
      
      // Create the call to Malloc...
      CallInst *MCall = new CallInst(MallocFunc,
                                     vector<Value*>(1, MallocArg));
      BBIL.insert(BBIL.begin()+i, MCall);
      
      // Create a cast instruction to convert to the right type...
      CastInst *MCast = new CastInst(MCall, MI->getType());
      BBIL.insert(BBIL.begin()+i+1, MCast);
      
      // Replace all uses of the old malloc inst with the cast inst
      MI->replaceAllUsesWith(MCast);
      delete MI;                          // Delete the malloc inst
      Changed = true;
    } else if (FreeInst *FI = dyn_cast<FreeInst>(*(BBIL.begin()+i))) {
      BBIL.remove(BB->getInstList().begin()+i);
      
      // Cast the argument to free into a ubyte*...
      CastInst *MCast = new CastInst(FI->getOperand(0), 
                                     PointerType::get(Type::UByteTy));
      BBIL.insert(BBIL.begin()+i, MCast);
      
      // Insert a call to the free function...
      CallInst *FCall = new CallInst(FreeFunc,
                                     vector<Value*>(1, MCast));
      BBIL.insert(BBIL.begin()+i+1, FCall);
      
      // Delete the old free instruction
      delete FI;
      Changed = true;
    }
  }

  return Changed;
}

bool RaiseAllocations::doInitialization(Module *M) {
  // If the module has a symbol table, they might be referring to the malloc
  // and free functions.  If this is the case, grab the method pointers that 
  // the module is using.
  //
  // Lookup %malloc and %free in the symbol table, for later use.  If they
  // don't exist, or are not external, we do not worry about converting calls
  // to that function into the appropriate instruction.
  //
  const FunctionType *MallocType =   // Get the type for malloc
    FunctionType::get(PointerType::get(Type::SByteTy),
                      vector<const Type*>(1, Type::UIntTy), false);

  const FunctionType *FreeType =     // Get the type for free
    FunctionType::get(Type::VoidTy,
                      vector<const Type*>(1, PointerType::get(Type::SByteTy)),
                      false);

  MallocFunc = M->getFunction("malloc", MallocType);
  FreeFunc   = M->getFunction("free"  , FreeType);

  // Don't mess with locally defined versions of these functions...
  if (MallocFunc && !MallocFunc->isExternal()) MallocFunc = 0;
  if (FreeFunc && !FreeFunc->isExternal())     FreeFunc = 0;
  return false;
}

// doOneCleanupPass - Do one pass over the input method, fixing stuff up.
//
bool RaiseAllocations::runOnBasicBlock(BasicBlock *BB) {
  bool Changed = false;
  BasicBlock::InstListType &BIL = BB->getInstList();

  for (BasicBlock::iterator BI = BB->begin(); BI != BB->end();) {
    Instruction *I = *BI;

    if (CallInst *CI = dyn_cast<CallInst>(I)) {
      if (CI->getCalledValue() == MallocFunc) {      // Replace call to malloc?
        const Type *PtrSByte = PointerType::get(Type::SByteTy);
        MallocInst *MallocI = new MallocInst(PtrSByte, CI->getOperand(1),
                                             CI->getName());
        CI->setName("");
        ReplaceInstWithInst(BIL, BI, MallocI);
        Changed = true;
        continue;  // Skip the ++BI
      } else if (CI->getCalledValue() == FreeFunc) { // Replace call to free?
        ReplaceInstWithInst(BIL, BI, new FreeInst(CI->getOperand(1)));
        Changed = true;
        continue;  // Skip the ++BI
      }
    }

    ++BI;
  }

  return Changed;
}

Pass *createLowerAllocationsPass(const TargetData &TD) {
  return new LowerAllocations(TD);
}
Pass *createRaiseAllocationsPass() {
  return new RaiseAllocations();
}


