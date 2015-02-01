//===- EarlyCSE.h - Simple and fast CSE pass --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file provides the interface for a simple, fast CSE pass.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_EARLYCSE_H
#define LLVM_TRANSFORMS_SCALAR_EARLYCSE_H

#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

/// \brief A simple and fast domtree-based CSE pass.
///
/// This pass does a simple depth-first walk over the dominator tree,
/// eliminating trivially redundant instructions and using instsimplify to
/// canonicalize things as it goes. It is intended to be fast and catch obvious
/// cases so that instcombine and other passes are more effective. It is
/// expected that a later pass of GVN will catch the interesting/hard cases.
class EarlyCSEPass {
public:
  static StringRef name() { return "EarlyCSEPass"; }

  /// \brief Run the pass over the function.
  ///
  /// This will lower all of th expect intrinsic calls in this function into
  /// branch weight metadata. That metadata will subsequently feed the analysis
  /// of the probabilities and frequencies of the CFG. After running this pass,
  /// no more expect intrinsics remain, allowing the rest of the optimizer to
  /// ignore them.
  PreservedAnalyses run(Function &F, AnalysisManager<Function> *AM);
};

}

#endif
