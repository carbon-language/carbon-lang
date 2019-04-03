//===--- Passes/RegReAssign.h ----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TOOLS_LLVM_BOLT_PASSES_REGREASSIGN_H
#define LLVM_TOOLS_LLVM_BOLT_PASSES_REGREASSIGN_H

#include "BinaryPasses.h"
#include "RegAnalysis.h"

namespace llvm {
namespace bolt {

class RegReAssign : public BinaryFunctionPass {
  std::vector<int64_t> RegScore;
  std::vector<size_t> RankedRegs;
  BitVector ClassicRegs;
  BitVector CalleeSaved;
  BitVector ClassicCSR;
  BitVector ExtendedCSR;
  BitVector GPRegs;

  /// Hooks to other passes
  std::unique_ptr<RegAnalysis> RA;
  std::unique_ptr<BinaryFunctionCallGraph> CG;

  /// Stats
  DenseSet<const BinaryFunction *> FuncsChanged;
  int64_t StaticBytesSaved{0};
  int64_t DynBytesSaved{0};

  void swap(BinaryContext &BC, BinaryFunction &Function, MCPhysReg A,
            MCPhysReg B);
  void rankRegisters(BinaryContext &BC, BinaryFunction &Function);
  void aggressivePassOverFunction(BinaryContext &BC, BinaryFunction &Function);
  bool conservativePassOverFunction(BinaryContext &BC,
                                    BinaryFunction &Function);
  void setupAggressivePass(BinaryContext &BC,
                           std::map<uint64_t, BinaryFunction> &BFs);
  void setupConservativePass(BinaryContext &BC,
                             std::map<uint64_t, BinaryFunction> &BFs);

public:
  /// BinaryPass public interface

  explicit RegReAssign(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override { return "regreassign"; }

  bool shouldPrint(const BinaryFunction &BF) const override {
    return BinaryFunctionPass::shouldPrint(BF) && FuncsChanged.count(&BF) > 0;
  }

  void runOnFunctions(BinaryContext &BC) override;
};
}
}

#endif
