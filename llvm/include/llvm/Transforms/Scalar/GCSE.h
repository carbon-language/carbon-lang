//===-- GCSE.h - SSA based Global Common Subexpr Elimination -----*- C++ -*--=//
//
// This pass is designed to be a very quick global transformation that
// eliminates global common subexpressions from a function.  It does this by
// examining the SSA value graph of the function, instead of doing slow
// bit-vector computations.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_GCSE_H
#define LLVM_TRANSFORMS_SCALAR_GCSE_H

class Pass;
Pass *createGCSEPass();

#endif
