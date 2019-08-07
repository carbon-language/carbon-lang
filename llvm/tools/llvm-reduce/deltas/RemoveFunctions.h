//===- llvm-reduce.cpp - The LLVM Delta Reduction utility -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a Specialized Delta Pass, which removes the functions that are
// not in the provided function-chunks.
//
//===----------------------------------------------------------------------===//

#include "Delta.h"
#include "llvm/Transforms/Utils/Cloning.h"

namespace llvm {

class RemoveFunctions {
public:
  /// Outputs the number of Functions in the given Module
  static int getTargetCount(Module *Program);
  /// Clones module and returns it with chunk functions only
  static std::unique_ptr<Module>
  extractChunksFromModule(std::vector<Chunk> ChunksToKeep, Module *Program);
};

} // namespace llvm
