//===- StackSlots.cpp  - Specialize LLVM code for target machine ---------===//
//
// This pass adds 2 empty slots at the top of function stack.
// These two slots are later used during code reoptimization
// for spilling the resgiter values when rewriting branches. 
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/MachineInstrInfo.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Pass.h"
#include "llvm/CodeGen/MachineCodeForMethod.h"

using std::map;
using std::cerr;


class StackSlots : public FunctionPass{
private:
  const TargetMachine &target;
public:
  StackSlots (const TargetMachine &T): target(T) {}

  bool runOnFunction(Function &F) {
    Value *v = ConstantSInt::get(Type::IntTy,0);
    MachineCodeForMethod &mcInfo = MachineCodeForMethod::get(&F);
    mcInfo.allocateLocalVar
      (target, v, 2*target.DataLayout.getTypeSize(PointerType::get(Type::IntTy)));
    
    return true;
  }
};


Pass* createStackSlotsPass(TargetMachine &T){
  return new StackSlots(T);
}

