//===- CycleAnalysis.h - Cycle Info for LLVM IR -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file declares an analysis pass that computes CycleInfo for
/// LLVM IR, specialized from GenericCycleInfo.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_CYCLEANALYSIS_H
#define LLVM_ANALYSIS_CYCLEANALYSIS_H

#include "llvm/ADT/GenericCycleInfo.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/SSAContext.h"

namespace llvm {
extern template class GenericCycleInfo<SSAContext>;
extern template class GenericCycle<SSAContext>;

using CycleInfo = GenericCycleInfo<SSAContext>;
using Cycle = CycleInfo::CycleT;

/// Analysis pass which computes a \ref CycleInfo.
class CycleAnalysis : public AnalysisInfoMixin<CycleAnalysis> {
  friend AnalysisInfoMixin<CycleAnalysis>;
  static AnalysisKey Key;

public:
  /// Provide the result typedef for this analysis pass.
  using Result = CycleInfo;

  /// Run the analysis pass over a function and produce a dominator tree.
  CycleInfo run(Function &F, FunctionAnalysisManager &);

  // TODO: verify analysis?
};

/// Printer pass for the \c DominatorTree.
class CycleInfoPrinterPass : public PassInfoMixin<CycleInfoPrinterPass> {
  raw_ostream &OS;

public:
  explicit CycleInfoPrinterPass(raw_ostream &OS);

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

/// Legacy analysis pass which computes a \ref CycleInfo.
class CycleInfoWrapperPass : public FunctionPass {
  Function *F = nullptr;
  CycleInfo CI;

public:
  static char ID;

  CycleInfoWrapperPass();

  CycleInfo &getCycleInfo() { return CI; }
  const CycleInfo &getCycleInfo() const { return CI; }

  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  void releaseMemory() override;
  void print(raw_ostream &OS, const Module *M = nullptr) const override;

  // TODO: verify analysis?
};

} // end namespace llvm

#endif // LLVM_ANALYSIS_CYCLEANALYSIS_H
