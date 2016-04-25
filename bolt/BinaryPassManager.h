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
#include "BinaryPasses.h"
#include "llvm/Support/Options.h"
#include "llvm/Support/CommandLine.h"
#include <map>
#include <memory>
#include <vector>

namespace llvm {
namespace bolt {

/// Simple class for managing analyses and optimizations on BinaryFunctions.
class BinaryFunctionPassManager {
private:
  BinaryContext &BC;
  std::map<uint64_t, BinaryFunction> &BFs;
  std::vector<std::pair<const cl::opt<bool> &,
                        std::unique_ptr<BinaryFunctionPass>>> Passes;

public:
  BinaryFunctionPassManager(BinaryContext &BC,
                            std::map<uint64_t, BinaryFunction> &BFs)
    : BC(BC), BFs(BFs) {}

  /// Adds a pass to this manager based on the value of its corresponding
  /// command-line option.
  void registerPass(std::unique_ptr<BinaryFunctionPass> Pass,
                    const cl::opt<bool> &Opt) {
    Passes.emplace_back(Opt, std::move(Pass));
  }

  /// Run all registered passes in the order they were added.
  void runPasses() {
    for (const auto &OptPassPair : Passes) {
      if (OptPassPair.first) {
        OptPassPair.second->runOnFunctions(BC, BFs);
      }
    }
  }

  /// Runs all enabled implemented passes on all functions.
  static void runAllPasses(BinaryContext &BC,
                           std::map<uint64_t, BinaryFunction> &Functions);

};

} // namespace bolt
} // namespace llvm

#endif
