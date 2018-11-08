//===-- Uops.h --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// A BenchmarkRunner implementation to measure uop decomposition.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_EXEGESIS_ROBSIZE_H
#define LLVM_TOOLS_LLVM_EXEGESIS_ROBSIZE_H

#include "BenchmarkRunner.h"
#include "SnippetGenerator.h"

namespace llvm {
namespace exegesis {

class ROBSizeSnippetGenerator : public SnippetGenerator {
public:
  ROBSizeSnippetGenerator(const LLVMState &State) : SnippetGenerator(State) {}
  ~ROBSizeSnippetGenerator() override;

  llvm::Expected<std::vector<CodeTemplate>>
  generateCodeTemplates(const Instruction &Instr) const override;
};

} // namespace exegesis
} // namespace llvm

#endif // LLVM_TOOLS_LLVM_EXEGESIS_ROBSIZE_H
