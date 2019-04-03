//===--- IdenticalCodeFolding.h -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_PASSES_IDENTICAL_CODE_FOLDING_H
#define LLVM_TOOLS_LLVM_BOLT_PASSES_IDENTICAL_CODE_FOLDING_H

#include "BinaryContext.h"
#include "BinaryFunction.h"
#include "Passes/BinaryPasses.h"

namespace llvm {
namespace bolt {

/// An optimization that replaces references to identical functions with
/// references to a single one of them.
///
class IdenticalCodeFolding : public BinaryFunctionPass {
public:
  explicit IdenticalCodeFolding(const cl::opt<bool> &PrintPass)
    : BinaryFunctionPass(PrintPass) { }

  const char *getName() const override {
    return "identical-code-folding";
  }
  void runOnFunctions(BinaryContext &BC) override;
};

} // namespace bolt
} // namespace llvm

#endif
