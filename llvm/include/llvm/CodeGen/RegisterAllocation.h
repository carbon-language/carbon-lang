//===-- CodeGen/RegisterAllocation.h - RegAlloc Pass -------------*- C++ -*--=//
//
// This pass register allocates a module, a method at a time.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_REGISTERALLOCATION_H
#define LLVM_CODEGEN_REGISTERALLOCATION_H

class Pass;
class TargetMachine;

//----------------------------------------------------------------------------
// Entry point for register allocation for a module
//----------------------------------------------------------------------------

Pass *getRegisterAllocator(TargetMachine &T);

#endif
