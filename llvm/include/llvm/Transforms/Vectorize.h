//===-- Vectorize.h - Vectorization Transformations -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for accessor functions that expose passes
// in the Vectorize transformations library.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_H
#define LLVM_TRANSFORMS_VECTORIZE_H

namespace llvm {
class BasicBlock;
class BasicBlockPass;

//===----------------------------------------------------------------------===//
//
// BBVectorize - A basic-block vectorization pass.
//
BasicBlockPass *createBBVectorizePass();

//===----------------------------------------------------------------------===//
/// @brief Vectorize the BasicBlock.
///
/// @param BB The BasicBlock to be vectorized
/// @param P  The current running pass, should require AliasAnalysis and
///           ScalarEvolution. After the vectorization, AliasAnalysis,
///           ScalarEvolution and CFG are preserved.
///
/// @return True if the BB is changed, false otherwise.
///
bool vectorizeBasicBlock(Pass *P, BasicBlock &BB);

} // End llvm namespace

#endif
