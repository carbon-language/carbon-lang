//===--- ReorderFunctions.h - Function reordering pass --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_PASSES_REORDER_FNCTIONS_H
#define LLVM_TOOLS_LLVM_BOLT_PASSES_REORDER_FNCTIONS_H

#include "BinaryPasses.h"
#include "BinaryFunctionCallGraph.h"

namespace llvm {
namespace bolt {

/// Modify function order for streaming based on hotness.
class ReorderFunctions : public BinaryFunctionPass {
  BinaryFunctionCallGraph Cg;

  void reorder(std::vector<Cluster> &&Clusters,
               std::map<uint64_t, BinaryFunction> &BFs);
 public:
  explicit ReorderFunctions(const cl::opt<bool> &PrintPass)
    : BinaryFunctionPass(PrintPass) { }

  const char *getName() const override {
    return "reorder-functions";
  }
  void runOnFunctions(BinaryContext &BC,
                      std::map<uint64_t, BinaryFunction> &BFs,
                      std::set<uint64_t> &LargeFunctions) override;
};

} // namespace bolt
} // namespace llvm

#endif
