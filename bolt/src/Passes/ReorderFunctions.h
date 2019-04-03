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
  enum ReorderType : char {
    RT_NONE = 0,
    RT_EXEC_COUNT,
    RT_HFSORT,
    RT_HFSORT_PLUS,
    RT_PETTIS_HANSEN,
    RT_RANDOM,
    RT_USER
  };

  explicit ReorderFunctions(const cl::opt<bool> &PrintPass)
    : BinaryFunctionPass(PrintPass) { }

  const char *getName() const override {
    return "reorder-functions";
  }
  void runOnFunctions(BinaryContext &BC) override;
};

} // namespace bolt
} // namespace llvm

#endif
