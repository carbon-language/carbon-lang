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
private:

  /// Stats for usage of max bytes for basic block alignment.
  std::vector<uint32_t> AlignHistogram;

  /// Stats: execution count of blocks that were aligned.
  uint64_t AlignedBlocksCount{0};

  /// Assign alignment to basic blocks based on profile.
  void alignBlocks(BinaryFunction &Function);

public:
  explicit AlignerPass() : BinaryFunctionPass(false) {}

  const char *getName() const override {
    return "aligner";
  }

  /// Pass entry point
  void runOnFunctions(BinaryContext &BC) override;
};

} // namespace bolt
} // namespace llvm


#endif
