//===-- InductionVars.h - Induction Variable Recognition ---------*- C++ -*--=//
//
// This family of functions is useful for Induction variable recognition, 
// removal and optimizations.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPT_INDUCTION_VARS_H
#define LLVM_OPT_INDUCTION_VARS_H

#include "llvm/Module.h"

namespace opt {

// DoInductionVariableCannonicalize - Simplify induction variables in loops
//
bool DoInductionVariableCannonicalize(Method *M);
static inline bool DoInductionVariableCannonicalize(Module *M) { 
  return M->reduceApply(DoInductionVariableCannonicalize); 
}

}  // end namespace opt

#endif
