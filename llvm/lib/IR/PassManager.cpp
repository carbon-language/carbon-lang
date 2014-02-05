//===- PassManager.h - Infrastructure for managing & running IR passes ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

static cl::opt<bool>
DebugPM("debug-pass-manager", cl::Hidden,
        cl::desc("Print pass management debugging information"));

PreservedAnalyses ModulePassManager::run(Module *M, ModuleAnalysisManager *AM) {
  PreservedAnalyses PA = PreservedAnalyses::all();

  if (DebugPM)
    dbgs() << "Starting module pass manager run.\n";

  for (unsigned Idx = 0, Size = Passes.size(); Idx != Size; ++Idx) {
    if (DebugPM)
      dbgs() << "Running module pass: " << Passes[Idx]->name() << "\n";

    PreservedAnalyses PassPA = Passes[Idx]->run(M, AM);
    if (AM)
      AM->invalidate(M, PassPA);
    PA.intersect(llvm_move(PassPA));
  }

  if (DebugPM)
    dbgs() << "Finished module pass manager run.\n";

  return PA;
}

ModuleAnalysisManager::ResultConceptT &
ModuleAnalysisManager::getResultImpl(void *PassID, Module *M) {
  ModuleAnalysisResultMapT::iterator RI;
  bool Inserted;
  llvm::tie(RI, Inserted) = ModuleAnalysisResults.insert(std::make_pair(
      PassID, polymorphic_ptr<detail::AnalysisResultConcept<Module *> >()));

  // If we don't have a cached result for this module, look up the pass and run
  // it to produce a result, which we then add to the cache.
  if (Inserted)
    RI->second = lookupPass(PassID).run(M, this);

  return *RI->second;
}

ModuleAnalysisManager::ResultConceptT *
ModuleAnalysisManager::getCachedResultImpl(void *PassID, Module *M) const {
  ModuleAnalysisResultMapT::const_iterator RI = ModuleAnalysisResults.find(PassID);
  return RI == ModuleAnalysisResults.end() ? 0 : &*RI->second;
}

void ModuleAnalysisManager::invalidateImpl(void *PassID, Module *M) {
  ModuleAnalysisResults.erase(PassID);
}

void ModuleAnalysisManager::invalidateImpl(Module *M,
                                           const PreservedAnalyses &PA) {
  // FIXME: This is a total hack based on the fact that erasure doesn't
  // invalidate iteration for DenseMap.
  for (ModuleAnalysisResultMapT::iterator I = ModuleAnalysisResults.begin(),
                                          E = ModuleAnalysisResults.end();
       I != E; ++I)
    if (I->second->invalidate(M, PA))
      ModuleAnalysisResults.erase(I);
}

PreservedAnalyses FunctionPassManager::run(Function *F, FunctionAnalysisManager *AM) {
  PreservedAnalyses PA = PreservedAnalyses::all();

  if (DebugPM)
    dbgs() << "Starting function pass manager run.\n";

  for (unsigned Idx = 0, Size = Passes.size(); Idx != Size; ++Idx) {
    if (DebugPM)
      dbgs() << "Running function pass: " << Passes[Idx]->name() << "\n";

    PreservedAnalyses PassPA = Passes[Idx]->run(F, AM);
    if (AM)
      AM->invalidate(F, PassPA);
    PA.intersect(llvm_move(PassPA));
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
FunctionAnalysisManager::getResultImpl(void *PassID, Function *F) {
  FunctionAnalysisResultMapT::iterator RI;
  bool Inserted;
  llvm::tie(RI, Inserted) = FunctionAnalysisResults.insert(std::make_pair(
      std::make_pair(PassID, F), FunctionAnalysisResultListT::iterator()));

  // If we don't have a cached result for this function, look up the pass and
  // run it to produce a result, which we then add to the cache.
  if (Inserted) {
    FunctionAnalysisResultListT &ResultList = FunctionAnalysisResultLists[F];
    ResultList.push_back(std::make_pair(PassID, lookupPass(PassID).run(F, this)));
    RI->second = llvm::prior(ResultList.end());
  }

  return *RI->second->second;
}

FunctionAnalysisManager::ResultConceptT *
FunctionAnalysisManager::getCachedResultImpl(void *PassID, Function *F) const {
  FunctionAnalysisResultMapT::const_iterator RI =
      FunctionAnalysisResults.find(std::make_pair(PassID, F));
  return RI == FunctionAnalysisResults.end() ? 0 : &*RI->second->second;
}

void FunctionAnalysisManager::invalidateImpl(void *PassID, Function *F) {
  FunctionAnalysisResultMapT::iterator RI =
      FunctionAnalysisResults.find(std::make_pair(PassID, F));
  if (RI == FunctionAnalysisResults.end())
    return;

  FunctionAnalysisResultLists[F].erase(RI->second);
}

void FunctionAnalysisManager::invalidateImpl(Function *F,
                                             const PreservedAnalyses &PA) {
  // Clear all the invalidated results associated specifically with this
  // function.
  SmallVector<void *, 8> InvalidatedPassIDs;
  FunctionAnalysisResultListT &ResultsList = FunctionAnalysisResultLists[F];
  for (FunctionAnalysisResultListT::iterator I = ResultsList.begin(),
                                             E = ResultsList.end();
       I != E;)
    if (I->second->invalidate(F, PA)) {
      InvalidatedPassIDs.push_back(I->first);
      I = ResultsList.erase(I);
    } else {
      ++I;
    }
  while (!InvalidatedPassIDs.empty())
    FunctionAnalysisResults.erase(
        std::make_pair(InvalidatedPassIDs.pop_back_val(), F));
}

char FunctionAnalysisManagerModuleProxy::PassID;

FunctionAnalysisManagerModuleProxy::Result
FunctionAnalysisManagerModuleProxy::run(Module *M) {
  assert(FAM.empty() && "Function analyses ran prior to the module proxy!");
  return Result(FAM);
}

FunctionAnalysisManagerModuleProxy::Result::~Result() {
  // Clear out the analysis manager if we're being destroyed -- it means we
  // didn't even see an invalidate call when we got invalidated.
  FAM.clear();
}

bool FunctionAnalysisManagerModuleProxy::Result::invalidate(
    Module *M, const PreservedAnalyses &PA) {
  // If this proxy isn't marked as preserved, then we can't even invalidate
  // individual function analyses, there may be an invalid set of Function
  // objects in the cache making it impossible to incrementally preserve them.
  // Just clear the entire manager.
  if (!PA.preserved(ID()))
    FAM.clear();

  // Return false to indicate that this result is still a valid proxy.
  return false;
}

char ModuleAnalysisManagerFunctionProxy::PassID;
