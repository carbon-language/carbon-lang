//===- ChangeAllocations.cpp - Modify %malloc & %free calls -----------------=//
//
// This file defines two passes that convert malloc and free instructions to
// calls to and from %malloc & %free function calls.  The LowerAllocations
// transformation is a target dependant tranformation because it depends on the
// size of data types and alignment constraints.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/ChangeAllocations.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Module.h"
#include "llvm/Function.h"
#include "llvm/DerivedTypes.h"
#include "llvm/iMemory.h"
#include "llvm/iOther.h"
#include "llvm/Pass.h"

namespace {

// RaiseAllocations - Turn %malloc and %free calls into the appropriate
// instruction.
//
class RaiseAllocations : public BasicBlockPass {
  Function *MallocFunc;   // Functions in the module we are processing
  Function *FreeFunc;     // Initialized by doPassInitializationVirt
public:
  inline RaiseAllocations() : MallocFunc(0), FreeFunc(0) {}

  const char *getPassName() const { return "Raise Allocations"; }

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


// createRaiseAllocationsPass - The interface to this file...
Pass *createRaiseAllocationsPass() {
  return new RaiseAllocations();
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
                    std::vector<const Type*>(1, Type::UIntTy), false);

  const FunctionType *FreeType =     // Get the type for free
    FunctionType::get(Type::VoidTy,
                   std::vector<const Type*>(1, PointerType::get(Type::SByteTy)),
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
