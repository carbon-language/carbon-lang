//===-- Transform/Utils/FunctionUtils.h - Function Utils --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This family of transformations manipulate LLVM functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_FUNCTION_H
#define LLVM_TRANSFORMS_UTILS_FUNCTION_H

#include "llvm/ADT/ArrayRef.h"
#include <vector>

namespace llvm {
  class BasicBlock;
  class DominatorTree;
  class Function;
  class Loop;

  /// \brief Test whether a basic block is a viable candidate for extraction.
  ///
  /// This tests whether a particular basic block is viable for extraction into
  /// a separate function. It can be used to prune code extraction eagerly
  /// rather than waiting for one of the Extract* methods below to reject
  /// a request.
  bool isBlockViableForExtraction(const BasicBlock &BB);

  /// ExtractCodeRegion - Rip out a sequence of basic blocks into a new
  /// function.
  ///
  Function* ExtractCodeRegion(DominatorTree& DT,
                              ArrayRef<BasicBlock*> code,
                              bool AggregateArgs = false);

  /// ExtractLoop - Rip out a natural loop into a new function.
  ///
  Function* ExtractLoop(DominatorTree& DT, Loop *L,
                        bool AggregateArgs = false);

  /// ExtractBasicBlock - Rip out a basic block (and the associated landing pad)
  /// into a new function.
  ///
  Function* ExtractBasicBlock(ArrayRef<BasicBlock*> BBs,
                              bool AggregateArgs = false);
}

#endif
