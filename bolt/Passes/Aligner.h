//===--------- Passes/Aligner.h -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_PASSES_ALIGNER_H
#define LLVM_TOOLS_LLVM_BOLT_PASSES_ALIGNER_H

#include "BinaryPasses.h"

namespace llvm {
namespace bolt {

class AlignerPass : public BinaryFunctionPass {
 public:
  explicit AlignerPass() : BinaryFunctionPass(false) {}

  const char *getName() const override {
    return "aligner";
  }

  /// Pass entry point
  void runOnFunctions(BinaryContext &BC,
                      std::map<uint64_t, BinaryFunction> &BFs,
                      std::set<uint64_t> &LargeFunctions) override;
};

} // namespace bolt
} // namespace llvm


#endif
