//===- StackSafetyAnalysis.h - Stack memory safety analysis -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Stack Safety Analysis detects allocas and arguments with safe access.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_STACKSAFETYANALYSIS_H
#define LLVM_ANALYSIS_STACKSAFETYANALYSIS_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

/// Interface to access stack safety analysis results for single function.
class StackSafetyInfo {
public:
  struct FunctionInfo;

private:
  std::unique_ptr<FunctionInfo> Info;

public:
  StackSafetyInfo();
  StackSafetyInfo(FunctionInfo &&Info);
  StackSafetyInfo(StackSafetyInfo &&);
  StackSafetyInfo &operator=(StackSafetyInfo &&);
  ~StackSafetyInfo();

  // TODO: Add useful for client methods.
  void print(raw_ostream &O) const;
};

/// StackSafetyInfo wrapper for the new pass manager.
class StackSafetyAnalysis : public AnalysisInfoMixin<StackSafetyAnalysis> {
  friend AnalysisInfoMixin<StackSafetyAnalysis>;
  static AnalysisKey Key;

public:
  using Result = StackSafetyInfo;
  StackSafetyInfo run(Function &F, FunctionAnalysisManager &AM);
};

/// Printer pass for the \c StackSafetyAnalysis results.
class StackSafetyPrinterPass : public PassInfoMixin<StackSafetyPrinterPass> {
  raw_ostream &OS;

public:
  explicit StackSafetyPrinterPass(raw_ostream &OS) : OS(OS) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

/// StackSafetyInfo wrapper for the legacy pass manager
class StackSafetyInfoWrapperPass : public FunctionPass {
  StackSafetyInfo SSI;

public:
  static char ID;
  StackSafetyInfoWrapperPass();

  const StackSafetyInfo &getResult() const { return SSI; }

  void print(raw_ostream &O, const Module *M) const override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

  bool runOnFunction(Function &F) override;
};

} // end namespace llvm

#endif // LLVM_ANALYSIS_STACKSAFETYANALYSIS_H
