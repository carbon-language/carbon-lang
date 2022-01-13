//===- ReduceFunctions.cpp - Specialized Delta Pass -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a function which calls the Generic Delta pass in order
// to reduce function bodies in the provided Module.
//
//===----------------------------------------------------------------------===//

#include "ReduceFunctionBodies.h"
#include "Delta.h"
#include "llvm/IR/GlobalValue.h"

using namespace llvm;

/// Removes all the bodies of defined functions that aren't inside any of the
/// desired Chunks.
static void
extractFunctionBodiesFromModule(const std::vector<Chunk> &ChunksToKeep,
                                Module *Program) {
  Oracle O(ChunksToKeep);

  // Delete out-of-chunk function bodies
  std::vector<Function *> FuncDefsToReduce;
  for (auto &F : *Program)
    if (!F.isDeclaration() && !O.shouldKeep()) {
      F.deleteBody();
      F.setComdat(nullptr);
    }
}

/// Counts the amount of non-declaration functions and prints their
/// respective name & index
static int countFunctionDefinitions(Module *Program) {
  // TODO: Silence index with --quiet flag
  errs() << "----------------------------\n";
  errs() << "Function Definition Index Reference:\n";
  int FunctionDefinitionCount = 0;
  for (auto &F : *Program)
    if (!F.isDeclaration())
      errs() << "\t" << ++FunctionDefinitionCount << ": " << F.getName()
             << "\n";

  errs() << "----------------------------\n";
  return FunctionDefinitionCount;
}

void llvm::reduceFunctionBodiesDeltaPass(TestRunner &Test) {
  errs() << "*** Reducing Function Bodies...\n";
  int Functions = countFunctionDefinitions(Test.getProgram());
  runDeltaPass(Test, Functions, extractFunctionBodiesFromModule);
  errs() << "----------------------------\n";
}
