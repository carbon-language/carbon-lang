//===-- NoopInsertion.h - Noop Insertion ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass adds fine-grained diversity by displacing code using randomly
// placed (optionally target supplied) Noop instructions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_NOOPINSERTION_H
#define LLVM_CODEGEN_NOOPINSERTION_H

#include "llvm/CodeGen/MachineFunctionPass.h"
#include <random>

namespace llvm {

class RandomNumberGenerator;

class NoopInsertion : public MachineFunctionPass {
public:
  static char ID;

  NoopInsertion();

private:
  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  std::unique_ptr<RandomNumberGenerator> RNG;

  // Uniform real distribution from 0 to 100
  std::uniform_real_distribution<double> Distribution =
      std::uniform_real_distribution<double>(0, 100);
};
}

#endif // LLVM_CODEGEN_NOOPINSERTION_H
