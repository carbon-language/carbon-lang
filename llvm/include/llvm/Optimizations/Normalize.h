//===-- Normalize.h - Functions that normalize code for the BE ---*- C++ -*--=//
//
// This file defines a family of transformations to normalize LLVM code for the
// code generation passes, so that the back end doesn't have to worry about
// annoying details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPT_NORMALIZE_H
#define LLVM_OPT_NORMALIZE_H

class Method;

// NormalizePhiConstantArgs - Insert loads of constants that are arguments to
// PHI in the appropriate predecessor basic block.
//
void NormalizePhiConstantArgs(Method *M);

#endif LLVM_OPT_NORMALIZE_H
