//===- StackSlots.cpp  - Specialize LLVM code for target machine ---------===//
//
// This pass adds 2 empty slots at the top of function stack.  These two slots
// are later used during code reoptimization for spilling the register values
// when rewriting branches.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/StackSlots.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/MachineInstrInfo.h"
#include "llvm/Constant.h"
#include "llvm/Function.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Pass.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionInfo.h"

namespace {
  class StackSlots : public FunctionPass {
    const TargetMachine &Target;
  public:
    StackSlots(const TargetMachine &T) : Target(T) {}
    
    const char *getPassName() const {
      return "Stack Slot Insertion for profiling code";
    }
    
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
    }
    
    bool runOnFunction(Function &F) {
      const Type *PtrInt = PointerType::get(Type::IntTy);
      unsigned Size = Target.getTargetData().getTypeSize(PtrInt);
      
      Value *V = Constant::getNullValue(Type::IntTy);
      MachineFunction::get(&F).getInfo()->allocateLocalVar(V, 2*Size);
      return true;
    }
  };
}

Pass *createStackSlotsPass(const TargetMachine &Target) {
  return new StackSlots(Target);
}
