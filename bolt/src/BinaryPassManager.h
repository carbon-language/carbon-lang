//===--- BinaryPassManager.h - Binary-level analysis/optimization passes --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// A very simple binary-level analysis/optimization passes system.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_BINARY_FUNCTION_PASS_MANAGER_H
#define LLVM_TOOLS_LLVM_BOLT_BINARY_FUNCTION_PASS_MANAGER_H

#include "BinaryFunction.h"
#include "Passes/BinaryPasses.h"
#include <map>
#include <memory>
#include <vector>

namespace llvm {
namespace bolt {

/// Simple class for managing analyses and optimizations on BinaryFunctions.
class BinaryFunctionPassManager {
private:
  BinaryContext &BC;
  std::vector<std::pair<const bool,
                        std::unique_ptr<BinaryFunctionPass>>> Passes;

 public:
  static const char TimerGroupName[];
  static const char TimerGroupDesc[];

  BinaryFunctionPassManager(BinaryContext &BC)
    : BC(BC) {}

  /// Adds a pass to this manager based on the value of its corresponding
  /// command-line option.
  void registerPass(std::unique_ptr<BinaryFunctionPass> Pass,
                    const bool Run) {
    Passes.emplace_back(Run, std::move(Pass));
  }

  /// Adds an unconditionally run pass to this manager.
  void registerPass(std::unique_ptr<BinaryFunctionPass> Pass) {
    Passes.emplace_back(true, std::move(Pass));
  }

  /// Run all registered passes in the order they were added.
  void runPasses();

  /// Runs all enabled implemented passes on all functions.
  static void runAllPasses(BinaryContext &BC);
};

} // namespace bolt
} // namespace llvm

#endif
