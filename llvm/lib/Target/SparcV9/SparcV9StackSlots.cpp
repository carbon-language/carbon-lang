//===- SparcV9StackSlots.cpp - Add empty stack slots to functions ---------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This pass adds 2 empty slots at the top of function stack.  These two slots
// are later used during code reoptimization for spilling the register values
// when rewriting branches.
//
//===----------------------------------------------------------------------===//

#include "SparcV9Internals.h"
#include "llvm/Constant.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/MachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

namespace llvm {

namespace {
  class StackSlots : public MachineFunctionPass {
    const TargetMachine &Target;
  public:
    StackSlots(const TargetMachine &T) : Target(T) {}
    
    const char *getPassName() const {
      return "Stack Slot Insertion for profiling code";
    }
    
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
    }
    
    bool runOnMachineFunction(MachineFunction &MF) {
      const Type *PtrInt = PointerType::get(Type::IntTy);
      unsigned Size = Target.getTargetData().getTypeSize(PtrInt);
      
      Value *V = Constant::getNullValue(Type::IntTy);
      MF.getInfo()->allocateLocalVar(V, 2*Size);
      return true;
    }
  };
}

Pass *createStackSlotsPass(const TargetMachine &Target) {
  return new StackSlots(Target);
}

} // End llvm namespace
