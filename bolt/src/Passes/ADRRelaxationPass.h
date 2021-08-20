//===--------- Passes/ADRRelaxationPass.h ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_PASSES_ADRRELAXATIONPASS_H
#define LLVM_TOOLS_LLVM_BOLT_PASSES_ADRRELAXATIONPASS_H

#include "BinaryPasses.h"

// This pass replaces AArch64 non-local ADR instructions
// with ADRP + ADD due to small offset range of ADR instruction
// (+- 1MB) which could be easely overflowed after BOLT optimizations
// Such problems are usually connected with errata 843419
// https://developer.arm.com/documentation/epm048406/2100/
// The linker could replace ADRP instruction with ADR in some cases.

namespace llvm {
namespace bolt {

class ADRRelaxationPass : public BinaryFunctionPass {
public:
  explicit ADRRelaxationPass() : BinaryFunctionPass(false) {}

  const char *getName() const override { return "adr-relaxation"; }

  /// Pass entry point
  void runOnFunctions(BinaryContext &BC) override;
  void runOnFunction(BinaryContext &BC, BinaryFunction &BF);
};

} // namespace bolt
} // namespace llvm

#endif
