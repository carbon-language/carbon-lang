//===-- ConstantProp.h - Functions for Constant Propogation ------*- C++ -*--=//
//
// This family of functions are useful for performing constant propogation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPT_CONSTANT_PROPOGATION_H
#define LLVM_OPT_CONSTANT_PROPOGATION_H

#include "llvm/Module.h"
class Method;
class TerminatorInst;
class ConstantPool;

namespace opt {

// DoConstantPropogation - Do trivial constant propogation and expression
// folding
bool DoConstantPropogation(Method *M);

static inline bool DoConstantPropogation(Module *M) { 
  return M->reduceApply(DoConstantPropogation); 
}


// ConstantFoldTerminator - If a terminator instruction is predicated on a
// constant value, convert it into an unconditional branch to the constant
// destination.
//
bool ConstantFoldTerminator(TerminatorInst *T);


//===----------------------------------------------------------------------===//
// Constant Pool Merging Pass
//
// This function merges all constants in the specified constant pool that have
// identical types and values.  This is useful for passes that generate lots of
// constants as a side effect of running.
//
bool DoConstantPoolMerging(ConstantPool &CP);
bool DoConstantPoolMerging(Method *M);

static inline bool DoConstantPoolMerging(Module *M) {
  return M->reduceApply(DoConstantPoolMerging) |
         DoConstantPoolMerging(M->getConstantPool());
}

//===----------------------------------------------------------------------===//
// Sparse Conditional Constant Propogation Pass
//

bool DoSCCP(Method *M);
static inline bool DoSCCP(Module *M) {
  return M->reduceApply(DoSCCP);
}

}  // End Namespace opt

#endif
