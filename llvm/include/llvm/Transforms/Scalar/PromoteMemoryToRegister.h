//===- PromoteMemoryToRegister.h - Convert memory refs to regs ---*- C++ -*--=//
//
// This pass is used to promote memory references to be register references.  A
// simple example of the transformation performed by this pass is:
//
//        FROM CODE                           TO CODE
//   %X = alloca int, uint 1                 ret int 42
//   store int 42, int *%X
//   %Y = load int* %X
//   ret int %Y
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_PROMOTEMEMORYTOREGISTER_H
#define LLVM_TRANSFORMS_SCALAR_PROMOTEMEMORYTOREGISTER_H

class Pass;

// createPromoteMemoryToRegister - Return the pass to perform this
// transformation.
Pass *createPromoteMemoryToRegister();

#endif
