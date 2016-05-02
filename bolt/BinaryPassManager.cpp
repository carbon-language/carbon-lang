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
EliminateUnreachable("eliminate-unreachable",
                     llvm::cl::desc("eliminate unreachable code"),
                     llvm::cl::Optional);

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

cl::opt<bool> BinaryFunctionPassManager::AlwaysOn(
  "always-run-pass",
  llvm::cl::desc("Used for passes that are always enabled"),
  cl::init(true),
  cl::ReallyHidden);

bool BinaryFunctionPassManager::NagUser = false;

void BinaryFunctionPassManager::runAllPasses(
  BinaryContext &BC,
  std::map<uint64_t, BinaryFunction> &Functions,
  std::set<uint64_t> &LargeFunctions
) {
  BinaryFunctionPassManager Manager(BC, Functions, LargeFunctions);

  // Here we manage dependencies/order manually, since passes are ran in the
  // order they're registered.

  Manager.registerPass(
    std::move(llvm::make_unique<EliminateUnreachableBlocks>(Manager.NagUser)),
    opts::EliminateUnreachable);

  Manager.registerPass(std::move(llvm::make_unique<ReorderBasicBlocks>()));

  Manager.registerPass(llvm::make_unique<OptimizeBodylessFunctions>(),
                       opts::OptimizeBodylessFunctions);

  Manager.registerPass(llvm::make_unique<InlineSmallFunctions>(),
                       opts::InlineSmallFunctions);

  Manager.registerPass(std::move(llvm::make_unique<FixupFunctions>()));

  Manager.runPasses();
}

} // namespace bolt
} // namespace llvm
