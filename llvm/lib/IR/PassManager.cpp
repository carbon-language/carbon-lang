//===- PassManager.h - Infrastructure for managing & running IR passes ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/PassManager.h"
#include "llvm/ADT/STLExtras.h"

using namespace llvm;

void ModulePassManager::run(Module *M) {
  for (unsigned Idx = 0, Size = Passes.size(); Idx != Size; ++Idx)
    if (Passes[Idx]->run(M))
      if (AM)
        AM->invalidateAll(M);
}

void ModuleAnalysisManager::invalidateAll(Module *M) {
  // FIXME: This is a total hack based on the fact that erasure doesn't
  // invalidate iteration for DenseMap.
  for (ModuleAnalysisResultMapT::iterator I = ModuleAnalysisResults.begin(),
                                          E = ModuleAnalysisResults.end();
       I != E; ++I)
    if (I->second->invalidate(M))
      ModuleAnalysisResults.erase(I);
}

const detail::AnalysisResultConcept<Module> &
ModuleAnalysisManager::getResultImpl(void *PassID, Module *M) {
  ModuleAnalysisResultMapT::iterator RI;
  bool Inserted;
  llvm::tie(RI, Inserted) = ModuleAnalysisResults.insert(std::make_pair(
      PassID, polymorphic_ptr<detail::AnalysisResultConcept<Module> >()));

  if (Inserted) {
    // We don't have a cached result for this result. Look up the pass and run
    // it to produce a result, which we then add to the cache.
    ModuleAnalysisPassMapT::const_iterator PI =
        ModuleAnalysisPasses.find(PassID);
    assert(PI != ModuleAnalysisPasses.end() &&
           "Analysis passes must be registered prior to being queried!");
    RI->second = PI->second->run(M);
  }

  return *RI->second;
}

void ModuleAnalysisManager::invalidateImpl(void *PassID, Module *M) {
  ModuleAnalysisResults.erase(PassID);
}

bool FunctionPassManager::run(Module *M) {
  bool Changed = false;
  for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I)
    for (unsigned Idx = 0, Size = Passes.size(); Idx != Size; ++Idx)
      if (Passes[Idx]->run(I)) {
        Changed = true;
        if (AM)
          AM->invalidateAll(I);
      }
  return Changed;
}

void FunctionAnalysisManager::invalidateAll(Function *F) {
  // Clear all the invalidated results associated specifically with this
  // function.
  SmallVector<void *, 8> InvalidatedPassIDs;
  FunctionAnalysisResultListT &ResultsList = FunctionAnalysisResultLists[F];
  for (FunctionAnalysisResultListT::iterator I = ResultsList.begin(),
                                             E = ResultsList.end();
       I != E;)
    if (I->second->invalidate(F)) {
      InvalidatedPassIDs.push_back(I->first);
      I = ResultsList.erase(I);
    } else {
      ++I;
    }
  while (!InvalidatedPassIDs.empty())
    FunctionAnalysisResults.erase(
        std::make_pair(InvalidatedPassIDs.pop_back_val(), F));
}

const detail::AnalysisResultConcept<Function> &
FunctionAnalysisManager::getResultImpl(void *PassID, Function *F) {
  FunctionAnalysisResultMapT::iterator RI;
  bool Inserted;
  llvm::tie(RI, Inserted) = FunctionAnalysisResults.insert(std::make_pair(
      std::make_pair(PassID, F), FunctionAnalysisResultListT::iterator()));

  if (Inserted) {
    // We don't have a cached result for this result. Look up the pass and run
    // it to produce a result, which we then add to the cache.
    FunctionAnalysisPassMapT::const_iterator PI =
        FunctionAnalysisPasses.find(PassID);
    assert(PI != FunctionAnalysisPasses.end() &&
           "Analysis passes must be registered prior to being queried!");
    FunctionAnalysisResultListT &ResultList = FunctionAnalysisResultLists[F];
    ResultList.push_back(std::make_pair(PassID, PI->second->run(F)));
    RI->second = llvm::prior(ResultList.end());
  }

  return *RI->second->second;
}

void FunctionAnalysisManager::invalidateImpl(void *PassID, Function *F) {
  FunctionAnalysisResultMapT::iterator RI =
      FunctionAnalysisResults.find(std::make_pair(PassID, F));
  if (RI == FunctionAnalysisResults.end())
    return;

  FunctionAnalysisResultLists[F].erase(RI->second);
}
