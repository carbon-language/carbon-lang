//===-- CodeGen/RegisterAllocation.h - RegAlloc Pass -------------*- C++ -*--=//
//
// This pass register allocates a module, a method at a time.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_REGISTERALLOCATION_H
#define LLVM_CODEGEN_REGISTERALLOCATION_H

#include "llvm/Pass.h"
class TargetMachine;
class MethodPass;

//----------------------------------------------------------------------------
// Entry point for register allocation for a module
//----------------------------------------------------------------------------

MethodPass *getRegisterAllocator(TargetMachine &T);

#endif
