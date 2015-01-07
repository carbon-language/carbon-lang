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

bool CGSCCAnalysisManager::empty() const {
  assert(CGSCCAnalysisResults.empty() == CGSCCAnalysisResultLists.empty() &&
         "The storage and index of analysis results disagree on how many there "
         "are!");
  return CGSCCAnalysisResults.empty();
}

void CGSCCAnalysisManager::clear() {
  CGSCCAnalysisResults.clear();
  CGSCCAnalysisResultLists.clear();
}

CGSCCAnalysisManager::ResultConceptT &
CGSCCAnalysisManager::getResultImpl(void *PassID, LazyCallGraph::SCC &C) {
  CGSCCAnalysisResultMapT::iterator RI;
  bool Inserted;
  std::tie(RI, Inserted) = CGSCCAnalysisResults.insert(std::make_pair(
      std::make_pair(PassID, &C), CGSCCAnalysisResultListT::iterator()));

  // If we don't have a cached result for this function, look up the pass and
  // run it to produce a result, which we then add to the cache.
  if (Inserted) {
    auto &P = lookupPass(PassID);
    if (DebugPM)
      dbgs() << "Running CGSCC analysis: " << P.name() << "\n";
    CGSCCAnalysisResultListT &ResultList = CGSCCAnalysisResultLists[&C];
    ResultList.emplace_back(PassID, P.run(C, this));
    RI->second = std::prev(ResultList.end());
  }

  return *RI->second->second;
}

CGSCCAnalysisManager::ResultConceptT *
CGSCCAnalysisManager::getCachedResultImpl(void *PassID,
                                          LazyCallGraph::SCC &C) const {
  CGSCCAnalysisResultMapT::const_iterator RI =
      CGSCCAnalysisResults.find(std::make_pair(PassID, &C));
  return RI == CGSCCAnalysisResults.end() ? nullptr : &*RI->second->second;
}

void CGSCCAnalysisManager::invalidateImpl(void *PassID, LazyCallGraph::SCC &C) {
  CGSCCAnalysisResultMapT::iterator RI =
      CGSCCAnalysisResults.find(std::make_pair(PassID, &C));
  if (RI == CGSCCAnalysisResults.end())
    return;

  if (DebugPM)
    dbgs() << "Invalidating CGSCC analysis: " << lookupPass(PassID).name()
           << "\n";
  CGSCCAnalysisResultLists[&C].erase(RI->second);
  CGSCCAnalysisResults.erase(RI);
}

PreservedAnalyses CGSCCAnalysisManager::invalidateImpl(LazyCallGraph::SCC &C,
                                                       PreservedAnalyses PA) {
  // Short circuit for a common case of all analyses being preserved.
  if (PA.areAllPreserved())
    return std::move(PA);

  if (DebugPM)
    dbgs() << "Invalidating all non-preserved analyses for SCC: " << C.getName()
           << "\n";

  // Clear all the invalidated results associated specifically with this
  // function.
  SmallVector<void *, 8> InvalidatedPassIDs;
  CGSCCAnalysisResultListT &ResultsList = CGSCCAnalysisResultLists[&C];
  for (CGSCCAnalysisResultListT::iterator I = ResultsList.begin(),
                                          E = ResultsList.end();
       I != E;) {
    void *PassID = I->first;

    // Pass the invalidation down to the pass itself to see if it thinks it is
    // necessary. The analysis pass can return false if no action on the part
    // of the analysis manager is required for this invalidation event.
    if (I->second->invalidate(C, PA)) {
      if (DebugPM)
        dbgs() << "Invalidating CGSCC analysis: " << lookupPass(PassID).name()
               << "\n";

      InvalidatedPassIDs.push_back(I->first);
      I = ResultsList.erase(I);
    } else {
      ++I;
    }

    // After handling each pass, we mark it as preserved. Once we've
    // invalidated any stale results, the rest of the system is allowed to
    // start preserving this analysis again.
    PA.preserve(PassID);
  }
  while (!InvalidatedPassIDs.empty())
    CGSCCAnalysisResults.erase(
        std::make_pair(InvalidatedPassIDs.pop_back_val(), &C));
  CGSCCAnalysisResultLists.erase(&C);

  return std::move(PA);
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
