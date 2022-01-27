//===- bolt/Passes/VeneerElimination.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This class implements a pass that removes linker-inserted veneers from the
// code and redirects veneer callers to call to veneers destinations
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/VeneerElimination.h"
#define DEBUG_TYPE "veneer-elim"

using namespace llvm;

namespace opts {

extern cl::OptionCategory BoltOptCategory;

static llvm::cl::opt<bool>
EliminateVeneers("elim-link-veneers",
  cl::desc("run veneer elimination pass"),
  cl::init(true),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltOptCategory));
} // namespace opts

namespace llvm {
namespace bolt {

void VeneerElimination::runOnFunctions(BinaryContext &BC) {
  if (!opts::EliminateVeneers || !BC.isAArch64())
    return;

  auto &BFs = BC.getBinaryFunctions();
  std::unordered_map<const MCSymbol *, const MCSymbol *> VeneerDestinations;
  uint64_t VeneersCount = 0;
  for (auto It = BFs.begin(); It != BFs.end();) {
    auto CurrentIt = It;
    ++It;

    if (CurrentIt->second.isAArch64Veneer()) {
      VeneersCount++;
      BinaryFunction &VeneerFunction = CurrentIt->second;

      MCInst &FirstInstruction = *(VeneerFunction.begin()->begin());
      const MCSymbol *VeneerTargetSymbol =
          BC.MIB->getTargetSymbol(FirstInstruction, 1);

      // Functions can have multiple symbols
      for (StringRef Name : VeneerFunction.getNames()) {
        MCSymbol *Symbol = BC.Ctx->lookupSymbol(Name);
        VeneerDestinations[Symbol] = VeneerTargetSymbol;
        BC.SymbolToFunctionMap.erase(Symbol);
      }

      BC.BinaryDataMap.erase(VeneerFunction.getAddress());
      BFs.erase(CurrentIt);
    }
  }

  LLVM_DEBUG(dbgs() << "BOLT-INFO: number of removed linker-inserted veneers :"
                    << VeneersCount << "\n");

  // Handle veneers to veneers in case they occur
  for (auto entry : VeneerDestinations) {
    const MCSymbol *src = entry.first;
    const MCSymbol *dest = entry.second;
    while (VeneerDestinations.find(dest) != VeneerDestinations.end()) {
      dest = VeneerDestinations[dest];
    }
    VeneerDestinations[src] = dest;
  }

  uint64_t VeneerCallers = 0;
  for (auto &It : BFs) {
    BinaryFunction &Function = It.second;
    for (BinaryBasicBlock &BB : Function) {
      for (MCInst &Instr : BB) {
        if (!BC.MIB->isCall(Instr) || BC.MIB->isIndirectCall(Instr))
          continue;

        const MCSymbol *TargetSymbol = BC.MIB->getTargetSymbol(Instr, 0);
        if (VeneerDestinations.find(TargetSymbol) == VeneerDestinations.end())
          continue;

        VeneerCallers++;
        if (!BC.MIB->replaceBranchTarget(
                Instr, VeneerDestinations[TargetSymbol], BC.Ctx.get()))
          assert(false && "updating veneer call destination failed");
      }
    }
  }

  LLVM_DEBUG(
      dbgs() << "BOLT-INFO: number of linker-inserted veneers call sites :"
             << VeneerCallers << "\n");
}

} // namespace bolt
} // namespace llvm
