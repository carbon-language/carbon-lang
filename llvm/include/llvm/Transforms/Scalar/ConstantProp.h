//===-- ConstantProp.h - Functions for Constant Propogation ------*- C++ -*--=//
//
// This family of functions are useful for performing constant propogation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_CONSTANT_PROPOGATION_H
#define LLVM_TRANSFORMS_SCALAR_CONSTANT_PROPOGATION_H

#include "llvm/Pass.h"
#include "llvm/BasicBlock.h"
class TerminatorInst;

struct ConstantPropogation : public MethodPass {
  // doConstantPropogation - Do trivial constant propogation and expression
  // folding
  static bool doConstantPropogation(Method *M);

  // doConstantPropogation - Constant prop a specific instruction.  Returns true
  // and potentially moves the iterator if constant propogation was performed.
  //
  static bool doConstantPropogation(BasicBlock *BB, BasicBlock::iterator &I);

  inline bool runOnMethod(Method *M) {
    return doConstantPropogation(M);
  }
};



// ConstantFoldTerminator - If a terminator instruction is predicated on a
// constant value, convert it into an unconditional branch to the constant
// destination.
//
bool ConstantFoldTerminator(TerminatorInst *T);


//===----------------------------------------------------------------------===//
// Sparse Conditional Constant Propogation Pass
//
struct SCCPPass : public MethodPass {
  static bool doSCCP(Method *M);

  inline bool runOnMethod(Method *M) {
    return doSCCP(M);
  }
};

#endif
