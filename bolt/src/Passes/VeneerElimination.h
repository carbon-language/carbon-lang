//===--- Passes/VeneerElimination.h ---------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_VENEER_ELIMINATION_H
#define LLVM_TOOLS_LLVM_BOLT_VENEER_ELIMINATION_H

#include "BinaryFunctionCallGraph.h"
#include "BinaryPasses.h"
#include "MCPlus.h"
#include "MCPlusBuilder.h"

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
