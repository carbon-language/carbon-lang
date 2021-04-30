//===--- Passes/VeneerElimination.h ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_VENEER_ELIMINATION_H
#define LLVM_TOOLS_LLVM_BOLT_VENEER_ELIMINATION_H

#include "BinaryPasses.h"

namespace llvm {
namespace bolt {

class VeneerElimination : public BinaryFunctionPass {
public:
  /// BinaryPass public interface
  explicit VeneerElimination(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {
    ;
  }

  const char *getName() const override { return "veneer-elimination"; }

  void runOnFunctions(BinaryContext &BC) override;
};
} // namespace bolt
} // namespace llvm

#endif
