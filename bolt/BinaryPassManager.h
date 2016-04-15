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
#include "llvm/Support/Options.h"
#include "llvm/Support/CommandLine.h"
#include <map>
#include <memory>
#include <vector>

namespace llvm {
namespace bolt {

/// An optimization/analysis pass that runs on functions.
class BinaryFunctionPass {
public:
  virtual ~BinaryFunctionPass() = default;
  virtual void runOnFunctions(BinaryContext &BC,
                              std::map<uint64_t, BinaryFunction> &BFs) = 0;
};

/// Simple class for managing analyses and optimizations on BinaryFunctions.
class BinaryFunctionPassManager {
private:
  BinaryContext *BC;
  std::map<uint64_t, BinaryFunction> *BFs;
  std::vector<std::pair<const cl::opt<bool> *,
                        std::unique_ptr<BinaryFunctionPass>>> Passes;

  /// Manager that contains all implemented passes.
  static std::unique_ptr<BinaryFunctionPassManager> GlobalPassManager;

public:
  BinaryFunctionPassManager(BinaryContext *BC = nullptr,
                            std::map<uint64_t, BinaryFunction> *BFs = nullptr)
    : BC(BC), BFs(BFs) {}

  static BinaryFunctionPassManager &getGlobalPassManager() {
    if (!GlobalPassManager) {
      GlobalPassManager = llvm::make_unique<BinaryFunctionPassManager>();
    }
    return *GlobalPassManager.get();
  }

  /// Adds a pass to this manager based on the value of its corresponding
  /// command-line option.
  void registerPass(std::unique_ptr<BinaryFunctionPass> Pass,
                    const cl::opt<bool> *Opt) {
    Passes.emplace_back(Opt, std::move(Pass));
  }

  /// Run all registered passes in the order they were added.
  void runPasses() {
    for (const auto &OptPassPair : Passes) {
      if (*OptPassPair.first) {
        OptPassPair.second->runOnFunctions(*BC, *BFs);
      }
    }
  }

  /// Runs all enabled implemented passes on all functions.
  static void runAllPasses(BinaryContext &BC,
                           std::map<uint64_t, BinaryFunction> &Functions);

};

template <typename T, cl::opt<bool> *Opt>
class RegisterBinaryPass {
public:
  RegisterBinaryPass() {
    BinaryFunctionPassManager::getGlobalPassManager().registerPass(
        std::move(llvm::make_unique<T>()), Opt);
  }
};

} // namespace bolt
} // namespace llvm

#endif
