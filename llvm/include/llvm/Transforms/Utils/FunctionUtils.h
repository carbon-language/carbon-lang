//===-- Transform/Utils/FunctionUtils.h - Function Utils --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This family of transformations manipulate LLVM functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_FUNCTION_H
#define LLVM_TRANSFORMS_UTILS_FUNCTION_H

#include <llvm/Analysis/Dominators.h>
#include <vector>

namespace llvm {
  class BasicBlock;
  class Function;
  class Loop;

  /// ExtractCodeRegion - rip out a sequence of basic blocks into a new function
  ///
  Function* ExtractCodeRegion(ETForest &DS, DominatorTree& DT,
                              const std::vector<BasicBlock*> &code,
                              bool AggregateArgs = false);

  /// ExtractLoop - rip out a natural loop into a new function
  ///
  Function* ExtractLoop(ETForest &DS, DominatorTree& DT, Loop *L,
                        bool AggregateArgs = false);

  /// ExtractBasicBlock - rip out a basic block into a new function
  ///
  Function* ExtractBasicBlock(BasicBlock *BB, bool AggregateArgs = false);
}

#endif
