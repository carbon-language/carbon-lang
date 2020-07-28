//===--- SplitFunctions.h - pass for splitting function code --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_PASSES_SPLIT_FUNCTIONS_H
#define LLVM_TOOLS_LLVM_BOLT_PASSES_SPLIT_FUNCTIONS_H

#include "BinaryContext.h"
#include "BinaryFunction.h"
#include "Passes/BinaryPasses.h"
#include "llvm/Support/CommandLine.h"
#include <atomic>

namespace llvm {
namespace bolt {

/// Split function code in multiple parts.
class SplitFunctions : public BinaryFunctionPass {
public:
  /// Settings for splitting function bodies into hot/cold partitions.
  enum SplittingType : char {
    ST_NONE = 0,      /// Do not split functions.
    ST_LARGE,         /// In non-relocation mode, only split functions that
                      /// are too large to fit into the original space.
    ST_ALL,           /// Split all functions.
  };

private:
  /// Split function body into fragments.
  void splitFunction(BinaryFunction &Function);

  std::atomic<uint64_t> SplitBytesHot{0ull};
  std::atomic<uint64_t> SplitBytesCold{0ull};

public:
  explicit SplitFunctions(const cl::opt<bool> &PrintPass)
    : BinaryFunctionPass(PrintPass) { }

  bool shouldOptimize(const BinaryFunction &BF) const override;

  const char *getName() const override {
    return "split-functions";
  }

  void runOnFunctions(BinaryContext &BC) override;
};

} // namespace bolt
} // namespace llvm

#endif
