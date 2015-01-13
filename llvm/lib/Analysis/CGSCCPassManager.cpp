//===- CGSCCPassManager.cpp - Managing & running CGSCC passes -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

static cl::opt<bool>
    DebugPM("debug-cgscc-pass-manager", cl::Hidden,
            cl::desc("Print CGSCC pass management debugging information"));

PreservedAnalyses CGSCCPassManager::run(LazyCallGraph::SCC &C,
                                        CGSCCAnalysisManager *AM) {
  PreservedAnalyses PA = PreservedAnalyses::all();

  if (DebugPM)
    dbgs() << "Starting CGSCC pass manager run.\n";

  for (unsigned Idx = 0, Size = Passes.size(); Idx != Size; ++Idx) {
    if (DebugPM)
      dbgs() << "Running CGSCC pass: " << Passes[Idx]->name() << "\n";

    PreservedAnalyses PassPA = Passes[Idx]->run(C, AM);

    // If we have an active analysis manager at this level we want to ensure we
    // update it as each pass runs and potentially invalidates analyses. We
    // also update the preserved set of analyses based on what analyses we have
    // already handled the invalidation for here and don't need to invalidate
    // when finished.
    if (AM)
      PassPA = AM->invalidate(C, std::move(PassPA));

    // Finally, we intersect the final preserved analyses to compute the
    // aggregate preserved set for this pass manager.
    PA.intersect(std::move(PassPA));
  }

  if (DebugPM)
    dbgs() << "Finished CGSCC pass manager run.\n";

  return PA;
}

char CGSCCAnalysisManagerModuleProxy::PassID;

CGSCCAnalysisManagerModuleProxy::Result
CGSCCAnalysisManagerModuleProxy::run(Module &M) {
  assert(CGAM->empty() && "CGSCC analyses ran prior to the module proxy!");
  return Result(*CGAM);
}

CGSCCAnalysisManagerModuleProxy::Result::~Result() {
  // Clear out the analysis manager if we're being destroyed -- it means we
  // didn't even see an invalidate call when we got invalidated.
  CGAM->clear();
}

bool CGSCCAnalysisManagerModuleProxy::Result::invalidate(
    Module &M, const PreservedAnalyses &PA) {
  // If this proxy isn't marked as preserved, then we can't even invalidate
  // individual CGSCC analyses, there may be an invalid set of SCC objects in
  // the cache making it impossible to incrementally preserve them.
  // Just clear the entire manager.
  if (!PA.preserved(ID()))
    CGAM->clear();

  // Return false to indicate that this result is still a valid proxy.
  return false;
}

char ModuleAnalysisManagerCGSCCProxy::PassID;

char FunctionAnalysisManagerCGSCCProxy::PassID;

FunctionAnalysisManagerCGSCCProxy::Result
FunctionAnalysisManagerCGSCCProxy::run(LazyCallGraph::SCC &C) {
  assert(FAM->empty() && "Function analyses ran prior to the CGSCC proxy!");
  return Result(*FAM);
}

FunctionAnalysisManagerCGSCCProxy::Result::~Result() {
  // Clear out the analysis manager if we're being destroyed -- it means we
  // didn't even see an invalidate call when we got invalidated.
  FAM->clear();
}

bool FunctionAnalysisManagerCGSCCProxy::Result::invalidate(
    LazyCallGraph::SCC &C, const PreservedAnalyses &PA) {
  // If this proxy isn't marked as preserved, then we can't even invalidate
  // individual function analyses, there may be an invalid set of Function
  // objects in the cache making it impossible to incrementally preserve them.
  // Just clear the entire manager.
  if (!PA.preserved(ID()))
    FAM->clear();

  // Return false to indicate that this result is still a valid proxy.
  return false;
}

char CGSCCAnalysisManagerFunctionProxy::PassID;
