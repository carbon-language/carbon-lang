//===- llvm/Transforms/SimpleStructMutation.h - Permute Structs --*- C++ -*--=//
//
// This pass does is a wrapper that can do a few simple structure mutation
// transformations.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SIMPLESTRUCTMUTATION_H
#define LLVM_TRANSFORMS_SIMPLESTRUCTMUTATION_H

class Pass;
Pass *createSwapElementsPass();
Pass *createSortElementsPass();

#endif
