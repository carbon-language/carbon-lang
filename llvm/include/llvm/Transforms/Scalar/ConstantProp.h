//===-- ConstantProp.h - Functions for Constant Propogation ------*- C++ -*--=//
//
// This family of functions are useful for performing constant propogation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_CONSTANT_PROPOGATION_H
#define LLVM_TRANSFORMS_SCALAR_CONSTANT_PROPOGATION_H

class Pass;

//===----------------------------------------------------------------------===//
// Normal Constant Propogation Pass
//
Pass *createConstantPropogationPass();

//===----------------------------------------------------------------------===//
// Sparse Conditional Constant Propogation Pass
//
Pass *createSCCPPass();

#endif
