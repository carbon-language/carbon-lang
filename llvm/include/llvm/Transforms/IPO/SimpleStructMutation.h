//===- llvm/Transforms/SwapStructContents.h - Permute Structs ----*- C++ -*--=//
//
// This pass does a simple transformation that swaps all of the elements of the
// struct types in the program around.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SWAPSTRUCTCONTENTS_H
#define LLVM_TRANSFORMS_SWAPSTRUCTCONTENTS_H

#include "llvm/Pass.h"

class SwapStructContents : public Pass {
  Pass *StructMutator;
public:
  // doPassInitialization - Figure out what transformation to do
  //
  bool doPassInitialization(Module *M);

  // doPerMethodWork - Virtual method overriden by subclasses to do the
  // per-method processing of the pass.
  //
  virtual bool doPerMethodWork(Method *M) {
    return StructMutator->doPerMethodWork(M);
  }

  // doPassFinalization - Forward to our worker.
  //
  virtual bool doPassFinalization(Module *M) {
    return StructMutator->doPassFinalization(M);
  }

};

#endif
