//===-- LevelChange.h - Passes for raising/lowering llvm code ----*- C++ -*--=//
//
// This family of passes is useful for changing the 'level' of a module. This
// can either be raising (f.e. converting direct addressing to use getelementptr
// for structs and arrays), or lowering (for instruction selection).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_LEVELCHANGE_H
#define LLVM_TRANSFORMS_LEVELCHANGE_H

#include "llvm/Pass.h"

// RaisePointerReferences - Try to eliminate as many pointer arithmetic
// expressions as possible, by converting expressions to use getelementptr and
// friends.
//
struct RaisePointerReferences : public Pass {
  static bool doit(Method *M);

  virtual bool doPerMethodWork(Method *M) { return doit(M); }
};


// EliminateAuxillaryInductionVariables - Eliminate all aux indvars.  This
// converts all induction variables to reference a cannonical induction
// variable (which starts at 0 and counts by 1).
//
struct EliminateAuxillaryInductionVariables : public Pass {
  static bool doit(Method *M) { return false; } // TODO!

  virtual bool doPerMethodWork(Method *M) { return doit(M); }
};

#endif
