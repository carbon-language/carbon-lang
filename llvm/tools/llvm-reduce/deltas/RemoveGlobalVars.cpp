//===- RemoveGlobalVars.cpp - Specialized Delta Pass ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a function which calls the Generic Delta pass in order
// to reduce initialized Global Variables in the provided Module.
//
//===----------------------------------------------------------------------===//

#include "RemoveGlobalVars.h"

/// Removes all the Initialized GVs that aren't inside the desired Chunks.
/// @returns the Module stripped of out-of-chunk GVs
static std::unique_ptr<Module>
extractGVsFromModule(std::vector<Chunk> ChunksToKeep, Module *Program) {
  std::unique_ptr<Module> Clone = CloneModule(*Program);

  // Get GVs inside desired chunks
  std::set<GlobalVariable *> GVsToKeep;
  unsigned I = 0, GVCount = 1;
  for (auto &GV : Clone->globals()) {
    if (GV.hasInitializer() && I < ChunksToKeep.size()) {
      if (GVCount >= ChunksToKeep[I].begin && GVCount <= ChunksToKeep[I].end)
        GVsToKeep.insert(&GV);
      if (GVCount == ChunksToKeep[I].end)
        ++I;
      ++GVCount;
    }
  }

  // Replace out-of-chunk GV uses with undef
  std::vector<GlobalVariable *> ToRemove;
  std::vector<Instruction *> InstToRemove;
  for (auto &GV : Clone->globals())
    if (GV.hasInitializer() && !GVsToKeep.count(&GV)) {
      for (auto U : GV.users())
        if (auto *Inst = dyn_cast<Instruction>(U))
          InstToRemove.push_back(Inst);

      GV.replaceAllUsesWith(UndefValue::get(GV.getType()));
      ToRemove.push_back(&GV);
    }

  // Delete Instruction uses of unwanted GVs
  for (auto *Inst : InstToRemove) {
    Inst->replaceAllUsesWith(UndefValue::get(Inst->getType()));
    Inst->eraseFromParent();
  }

  for (auto *GV : ToRemove)
    GV->eraseFromParent();

  return Clone;
}

/// Counts the amount of initialized GVs and displays their
/// respective name & index
static int countGVs(Module *Program) {
  // TODO: Silence index with --quiet flag
  outs() << "----------------------------\n";
  outs() << "GlobalVariable Index Reference:\n";
  int GVCount = 0;
  for (auto &GV : Program->globals())
    if (GV.hasInitializer()) {
      ++GVCount;
      outs() << "\t" << GVCount << ": " << GV.getName() << "\n";
    }
  outs() << "----------------------------\n";
  return GVCount;
}

void llvm::removeGlobalsDeltaPass(TestRunner &Test) {
  int GVCount = countGVs(Test.getProgram());
  runDeltaPass(Test, GVCount, extractGVsFromModule);
}
