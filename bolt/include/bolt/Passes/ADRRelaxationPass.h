//===- bolt/Passes/ADRRelaxationPass.h --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the ADRRelaxationPass class, which replaces AArch64
// non-local ADR instructions with ADRP + ADD due to small offset range of ADR
// instruction (+- 1MB) which could be easily overflowed after BOLT
// optimizations. Such problems are usually connected with errata 843419
// https://developer.arm.com/documentation/epm048406/2100/
// The linker could replace ADRP instruction with ADR in some cases.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_ADRRELAXATIONPASS_H
#define BOLT_PASSES_ADRRELAXATIONPASS_H

#include "bolt/Passes/BinaryPasses.h"

namespace llvm {
namespace bolt {

class ADRRelaxationPass : public BinaryFunctionPass {
public:
  explicit ADRRelaxationPass() : BinaryFunctionPass(false) {}

  const char *getName() const override { return "adr-relaxation"; }

  /// Pass entry point
  void runOnFunctions(BinaryContext &BC) override;
  void runOnFunction(BinaryFunction &BF);
};

} // namespace bolt
} // namespace llvm

#endif
