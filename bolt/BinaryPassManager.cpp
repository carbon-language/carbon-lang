//===--- BinaryPassManager.cpp - Binary-level analysis/optimization passes ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "BinaryPassManager.h"

namespace opts {

static llvm::cl::opt<bool>
OptimizeBodylessFunctions(
    "optimize-bodyless-functions",
    llvm::cl::desc("optimize functions that just do a tail call"),
    llvm::cl::Optional);

static llvm::cl::opt<bool>
InlineSmallFunctions(
    "inline-small-functions",
    llvm::cl::desc("inline functions with a single basic block"),
    llvm::cl::Optional);

} // namespace opts

namespace llvm {
namespace bolt {

void BinaryFunctionPassManager::runAllPasses(
    BinaryContext &BC,
    std::map<uint64_t, BinaryFunction> &Functions) {
  BinaryFunctionPassManager Manager(BC, Functions);

  // Here we manage dependencies/order manually, since passes are ran in the
  // order they're registered.

  Manager.registerPass(llvm::make_unique<OptimizeBodylessFunctions>(),
                       opts::OptimizeBodylessFunctions);

  Manager.registerPass(llvm::make_unique<InlineSmallFunctions>(),
                       opts::InlineSmallFunctions);

  Manager.runPasses();
}

} // namespace bolt
} // namespace llvm
