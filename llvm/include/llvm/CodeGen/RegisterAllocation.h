//===-- CodeGen/RegisterAllocation.h - RegAlloc Pass -------------*- C++ -*--=//
//
// This pass register allocates a module, a method at a time.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_REGISTERALLOCATION_H
#define LLVM_CODEGEN_REGISTERALLOCATION_H

#include "llvm/Pass.h"
class TargetMachine;

//----------------------------------------------------------------------------
// Entry point for register allocation for a module
//----------------------------------------------------------------------------

class RegisterAllocation : public MethodPass {
  TargetMachine &Target;
public:
  inline RegisterAllocation(TargetMachine &T) : Target(T) {}
  bool runOnMethod(Method *M);
};

#endif
