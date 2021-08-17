//===--------- Passes/ThreeWayBranch.h ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_PASSES_THREEWAYBRANCH_H
#define LLVM_TOOLS_LLVM_BOLT_PASSES_THREEWAYBRANCH_H

#include "BinaryPasses.h"

namespace llvm {
namespace bolt {

/// Pass for optimizing a three way branch namely a single comparison and 2
/// conditional jumps by reordering blocks, replacing successors, and replacing
/// jump conditions and destinations
class ThreeWayBranch : public BinaryFunctionPass {
  /// Record how many 3 way branches were adjusted
  uint64_t BranchesAltered = 0;

  /// Returns true if this pass should run on Function
  bool shouldRunOnFunction(BinaryFunction &Function);

  /// Runs pass on Function
  void runOnFunction(BinaryFunction &Function);

public:
  explicit ThreeWayBranch() : BinaryFunctionPass(false) {}

  const char *getName() const override { return "three way branch"; }

  void runOnFunctions(BinaryContext &BC) override;
};

} // namespace bolt
} // namespace llvm

#endif
