//===- llvm/Transforms/SwapStructContents.h - Permute Structs ----*- C++ -*--=//
//
// This pass does a simple transformation that swaps all of the elements of the
// struct types in the program around.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SWAPSTRUCTCONTENTS_H
#define LLVM_TRANSFORMS_SWAPSTRUCTCONTENTS_H

#include "llvm/Transforms/MutateStructTypes.h"

// FIXME: Move to correct location!
class PrebuiltStructMutation : public MutateStructTypes {
public:
  enum Transform { SwapElements, SortElements };

  PrebuiltStructMutation(Module *M, enum Transform XForm)
    : MutateStructTypes(getTransforms(M, XForm)) {}

private:
  static TransformsType getTransforms(Module *M, enum Transform);
};

#endif
