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

namespace llvm {
namespace bolt {

std::unique_ptr<BinaryFunctionPassManager>
BinaryFunctionPassManager::GlobalPassManager;

void BinaryFunctionPassManager::runAllPasses(
    BinaryContext &BC,
    std::map<uint64_t, BinaryFunction> &Functions) {
  auto &Manager = getGlobalPassManager();
  Manager.BC = &BC;
  Manager.BFs = &Functions;
  Manager.runPasses();
}

} // namespace bolt
} // namespace llvm
