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
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

static cl::opt<bool>
    DebugPM("debug-pass-manager", cl::Hidden,
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

ModuleAnalysisManager::ResultConceptT &
ModuleAnalysisManager::getResultImpl(void *PassID, Module &M) {
  ModuleAnalysisResultMapT::iterator RI;
  bool Inserted;
  std::tie(RI, Inserted) = ModuleAnalysisResults.insert(std::make_pair(
      PassID, std::unique_ptr<detail::AnalysisResultConcept<Module &>>()));

  // If we don't have a cached result for this module, look up the pass and run
  // it to produce a result, which we then add to the cache.
  if (Inserted) {
    auto &P = lookupPass(PassID);
    if (DebugPM)
      dbgs() << "Running module analysis: " << P.name() << "\n";
    RI->second = P.run(M, this);
  }

  return *RI->second;
}

ModuleAnalysisManager::ResultConceptT *
ModuleAnalysisManager::getCachedResultImpl(void *PassID, Module &M) const {
  ModuleAnalysisResultMapT::const_iterator RI =
      ModuleAnalysisResults.find(PassID);
  return RI == ModuleAnalysisResults.end() ? nullptr : &*RI->second;
}

void ModuleAnalysisManager::invalidateImpl(void *PassID, Module &M) {
  if (DebugPM)
    dbgs() << "Invalidating module analysis: " << lookupPass(PassID).name()
           << "\n";
  ModuleAnalysisResults.erase(PassID);
}

PreservedAnalyses ModuleAnalysisManager::invalidateImpl(Module &M,
                                                        PreservedAnalyses PA) {
  // Short circuit for a common case of all analyses being preserved.
  if (PA.areAllPreserved())
    return std::move(PA);

  if (DebugPM)
    dbgs() << "Invalidating all non-preserved analyses for module: "
           << M.getModuleIdentifier() << "\n";

  // FIXME: This is a total hack based on the fact that erasure doesn't
  // invalidate iteration for DenseMap.
  for (ModuleAnalysisResultMapT::iterator I = ModuleAnalysisResults.begin(),
                                          E = ModuleAnalysisResults.end();
       I != E; ++I) {
    void *PassID = I->first;

    // Pass the invalidation down to the pass itself to see if it thinks it is
    // necessary. The analysis pass can return false if no action on the part
    // of the analysis manager is required for this invalidation event.
    if (I->second->invalidate(M, PA)) {
      if (DebugPM)
        dbgs() << "Invalidating module analysis: "
               << lookupPass(PassID).name() << "\n";

      ModuleAnalysisResults.erase(I);
    }

    // After handling each pass, we mark it as preserved. Once we've
    // invalidated any stale results, the rest of the system is allowed to
    // start preserving this analysis again.
    PA.preserve(PassID);
  }

  return std::move(PA);
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

bool FunctionAnalysisManager::empty() const {
  assert(FunctionAnalysisResults.empty() ==
             FunctionAnalysisResultLists.empty() &&
         "The storage and index of analysis results disagree on how many there "
         "are!");
  return FunctionAnalysisResults.empty();
}

void FunctionAnalysisManager::clear() {
  FunctionAnalysisResults.clear();
  FunctionAnalysisResultLists.clear();
}

FunctionAnalysisManager::ResultConceptT &
FunctionAnalysisManager::getResultImpl(void *PassID, Function &F) {
  FunctionAnalysisResultMapT::iterator RI;
  bool Inserted;
  std::tie(RI, Inserted) = FunctionAnalysisResults.insert(std::make_pair(
      std::make_pair(PassID, &F), FunctionAnalysisResultListT::iterator()));

  // If we don't have a cached result for this function, look up the pass and
  // run it to produce a result, which we then add to the cache.
  if (Inserted) {
    auto &P = lookupPass(PassID);
    if (DebugPM)
      dbgs() << "Running function analysis: " << P.name() << "\n";
    FunctionAnalysisResultListT &ResultList = FunctionAnalysisResultLists[&F];
    ResultList.emplace_back(PassID, P.run(F, this));
    RI->second = std::prev(ResultList.end());
  }

  return *RI->second->second;
}

FunctionAnalysisManager::ResultConceptT *
FunctionAnalysisManager::getCachedResultImpl(void *PassID, Function &F) const {
  FunctionAnalysisResultMapT::const_iterator RI =
      FunctionAnalysisResults.find(std::make_pair(PassID, &F));
  return RI == FunctionAnalysisResults.end() ? nullptr : &*RI->second->second;
}

void FunctionAnalysisManager::invalidateImpl(void *PassID, Function &F) {
  FunctionAnalysisResultMapT::iterator RI =
      FunctionAnalysisResults.find(std::make_pair(PassID, &F));
  if (RI == FunctionAnalysisResults.end())
    return;

  if (DebugPM)
    dbgs() << "Invalidating function analysis: " << lookupPass(PassID).name()
           << "\n";
  FunctionAnalysisResultLists[&F].erase(RI->second);
  FunctionAnalysisResults.erase(RI);
}

PreservedAnalyses
FunctionAnalysisManager::invalidateImpl(Function &F, PreservedAnalyses PA) {
  // Short circuit for a common case of all analyses being preserved.
  if (PA.areAllPreserved())
    return std::move(PA);

  if (DebugPM)
    dbgs() << "Invalidating all non-preserved analyses for function: "
           << F.getName() << "\n";

  // Clear all the invalidated results associated specifically with this
  // function.
  SmallVector<void *, 8> InvalidatedPassIDs;
  FunctionAnalysisResultListT &ResultsList = FunctionAnalysisResultLists[&F];
  for (FunctionAnalysisResultListT::iterator I = ResultsList.begin(),
                                             E = ResultsList.end();
       I != E;) {
    void *PassID = I->first;

    // Pass the invalidation down to the pass itself to see if it thinks it is
    // necessary. The analysis pass can return false if no action on the part
    // of the analysis manager is required for this invalidation event.
    if (I->second->invalidate(F, PA)) {
      if (DebugPM)
        dbgs() << "Invalidating function analysis: "
               << lookupPass(PassID).name() << "\n";

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
    FunctionAnalysisResults.erase(
        std::make_pair(InvalidatedPassIDs.pop_back_val(), &F));
  if (ResultsList.empty())
    FunctionAnalysisResultLists.erase(&F);

  return std::move(PA);
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
