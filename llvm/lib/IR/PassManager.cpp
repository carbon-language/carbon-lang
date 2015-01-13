//===- PassManager.cpp - Infrastructure for managing & running IR passes --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/PassManager.h"

using namespace llvm;
using llvm::detail::DebugPM;

cl::opt<bool> llvm::detail::DebugPM(
    "debug-pass-manager", cl::Hidden,
    cl::desc("Print pass management debugging information"));

PreservedAnalyses ModulePassManager::run(Module &M, ModuleAnalysisManager *AM) {
  PreservedAnalyses PA = PreservedAnalyses::all();

  if (DebugPM)
    dbgs() << "Starting module pass manager run.\n";

  for (unsigned Idx = 0, Size = Passes.size(); Idx != Size; ++Idx) {
    if (DebugPM)
      dbgs() << "Running module pass: " << Passes[Idx]->name() << "\n";

    PreservedAnalyses PassPA = Passes[Idx]->run(M, AM);

    // If we have an active analysis manager at this level we want to ensure we
    // update it as each pass runs and potentially invalidates analyses. We
    // also update the preserved set of analyses based on what analyses we have
    // already handled the invalidation for here and don't need to invalidate
    // when finished.
    if (AM)
      PassPA = AM->invalidate(M, std::move(PassPA));

    // Finally, we intersect the final preserved analyses to compute the
    // aggregate preserved set for this pass manager.
    PA.intersect(std::move(PassPA));

    M.getContext().yield();
  }

  if (DebugPM)
    dbgs() << "Finished module pass manager run.\n";

  return PA;
}

PreservedAnalyses FunctionPassManager::run(Function &F,
                                           FunctionAnalysisManager *AM) {
  PreservedAnalyses PA = PreservedAnalyses::all();

  if (DebugPM)
    dbgs() << "Starting function pass manager run.\n";

  for (unsigned Idx = 0, Size = Passes.size(); Idx != Size; ++Idx) {
    if (DebugPM)
      dbgs() << "Running function pass: " << Passes[Idx]->name() << "\n";

    PreservedAnalyses PassPA = Passes[Idx]->run(F, AM);

    // If we have an active analysis manager at this level we want to ensure we
    // update it as each pass runs and potentially invalidates analyses. We
    // also update the preserved set of analyses based on what analyses we have
    // already handled the invalidation for here and don't need to invalidate
    // when finished.
    if (AM)
      PassPA = AM->invalidate(F, std::move(PassPA));

    // Finally, we intersect the final preserved analyses to compute the
    // aggregate preserved set for this pass manager.
    PA.intersect(std::move(PassPA));

    F.getContext().yield();
  }

  if (DebugPM)
    dbgs() << "Finished function pass manager run.\n";

  return PA;
}

char FunctionAnalysisManagerModuleProxy::PassID;

FunctionAnalysisManagerModuleProxy::Result
FunctionAnalysisManagerModuleProxy::run(Module &M) {
  assert(FAM->empty() && "Function analyses ran prior to the module proxy!");
  return Result(*FAM);
}

FunctionAnalysisManagerModuleProxy::Result::~Result() {
  // Clear out the analysis manager if we're being destroyed -- it means we
  // didn't even see an invalidate call when we got invalidated.
  FAM->clear();
}

bool FunctionAnalysisManagerModuleProxy::Result::invalidate(
    Module &M, const PreservedAnalyses &PA) {
  // If this proxy isn't marked as preserved, then we can't even invalidate
  // individual function analyses, there may be an invalid set of Function
  // objects in the cache making it impossible to incrementally preserve them.
  // Just clear the entire manager.
  if (!PA.preserved(ID()))
    FAM->clear();

  // Return false to indicate that this result is still a valid proxy.
  return false;
}

char ModuleAnalysisManagerFunctionProxy::PassID;
