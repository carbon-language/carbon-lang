//===-- ConstantProp.h - Functions for Constant Propogation ------*- C++ -*--=//
//
// This family of functions are useful for performing constant propogation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPT_CONSTANT_PROPOGATION_H
#define LLVM_OPT_CONSTANT_PROPOGATION_H

#include "llvm/Transforms/Pass.h"
class TerminatorInst;

namespace opt {

struct ConstantPropogation : public Pass {
  // doConstantPropogation - Do trivial constant propogation and expression
  // folding
  static bool doConstantPropogation(Method *M);

  inline bool doPerMethodWork(Method *M) {
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
struct SCCPPass : public Pass {
  static bool doSCCP(Method *M);

  inline bool doPerMethodWork(Method *M) {
    return doSCCP(M);
  }
};

}  // End Namespace opt

#endif
