//===- llvm/Transforms/SimpleStructMutation.h - Permute Structs --*- C++ -*--=//
//
// This pass does is a wrapper that can do a few simple structure mutation
// transformations.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SIMPLESTRUCTMUTATION_H
#define LLVM_TRANSFORMS_SIMPLESTRUCTMUTATION_H

#include "llvm/Transforms/IPO/MutateStructTypes.h"

class SimpleStructMutation : public MutateStructTypes {
public:
  enum Transform { SwapElements, SortElements } CurrentXForm;

  SimpleStructMutation(enum Transform XForm) : CurrentXForm(XForm) {}

  virtual bool run(Module *M) {
    setTransforms(getTransforms(M, CurrentXForm));
    bool Changed = MutateStructTypes::run(M);
    clearTransforms();
    return Changed;
  }

  // getAnalysisUsageInfo - This function needs the results of the
  // FindUsedTypes and FindUnsafePointerTypes analysis passes...
  //
  virtual void getAnalysisUsageInfo(Pass::AnalysisSet &Required,
                                    Pass::AnalysisSet &Destroyed,
                                    Pass::AnalysisSet &Provided);

private:
  TransformsType getTransforms(Module *M, enum Transform);
};

#endif
