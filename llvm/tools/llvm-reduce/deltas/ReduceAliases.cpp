//===- ReduceAliases.cpp - Specialized Delta Pass -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a function which calls the Generic Delta pass in order
// to reduce aliases in the provided Module.
//
//===----------------------------------------------------------------------===//

#include "ReduceAliases.h"
#include "Delta.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalValue.h"

using namespace llvm;

/// Removes all aliases aren't inside any of the
/// desired Chunks.
static void extractAliasesFromModule(const std::vector<Chunk> &ChunksToKeep,
                                     Module *Program) {
  Oracle O(ChunksToKeep);

  for (auto &GA : make_early_inc_range(Program->aliases())) {
    if (!O.shouldKeep()) {
      GA.replaceAllUsesWith(GA.getAliasee());
      GA.eraseFromParent();
    }
  }
}

/// Counts the amount of aliases and prints their respective name & index.
static int countAliases(Module *Program) {
  // TODO: Silence index with --quiet flag
  errs() << "----------------------------\n";
  errs() << "Aliases Index Reference:\n";
  int Count = 0;
  for (auto &GA : Program->aliases())
    errs() << "\t" << ++Count << ": " << GA.getName() << "\n";

  errs() << "----------------------------\n";
  return Count;
}

void llvm::reduceAliasesDeltaPass(TestRunner &Test) {
  errs() << "*** Reducing Aliases ...\n";
  int Functions = countAliases(Test.getProgram());
  runDeltaPass(Test, Functions, extractAliasesFromModule);
  errs() << "----------------------------\n";
}
