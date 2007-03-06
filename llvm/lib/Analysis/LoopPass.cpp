//===- LoopPass.cpp - Loop Pass and Loop Pass Manager ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Devang Patel and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements LoopPass and LPPassManager. All loop optimization
// and transformation passes are derived from LoopPass. LPPassManager is
// responsible for managing LoopPasses.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LoopPass.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
// LPPassManager
//
/// LPPassManager manages FPPassManagers and CalLGraphSCCPasses.

LPPassManager::LPPassManager(int Depth) : PMDataManager(Depth) { 
  skipThisLoop = false;
  redoThisLoop = false;
}

/// Delete loop from the loop queue. This is used by Loop pass to inform
/// Loop Pass Manager that it should skip rest of the passes for this loop.
void LPPassManager::deleteLoopFromQueue(Loop *L) {
  // Do not pop loop from LQ here. It will be done by runOnFunction while loop.
  skipThisLoop = true;
}

// Reoptimize this loop. LPPassManager will re-insert this loop into the
// queue. This allows LoopPass to change loop nest for the loop. This
// utility may send LPPassManager into infinite loops so use caution.
void LPPassManager::redoLoop(Loop *L) {
  redoThisLoop = true;
}

// Recurse through all subloops and all loops  into LQ.
static void addLoopIntoQueue(Loop *L, std::deque<Loop *> &LQ) {
  for (Loop::iterator I = L->begin(), E = L->end(); I != E; ++I)
    addLoopIntoQueue(*I, LQ);
  LQ.push_back(L);
}

/// run - Execute all of the passes scheduled for execution.  Keep track of
/// whether any of the passes modifies the function, and if so, return true.
bool LPPassManager::runOnFunction(Function &F) {
  LoopInfo &LI = getAnalysis<LoopInfo>();
  bool Changed = false;

  // Populate Loop Queue
  for (LoopInfo::iterator I = LI.begin(), E = LI.end(); I != E; ++I)
    addLoopIntoQueue(*I, LQ);

  // Initialization
  for (std::deque<Loop *>::const_iterator I = LQ.begin(), E = LQ.end();
       I != E; ++I) {
    Loop *L = *I;
    for (unsigned Index = 0; Index < getNumContainedPasses(); ++Index) {  
      Pass *P = getContainedPass(Index);
      LoopPass *LP = dynamic_cast<LoopPass *>(P);
      if (LP)
        Changed |= LP->doInitialization(L, *this);
    }
  }

  // Walk Loops
  while (!LQ.empty()) {
      
    Loop *L  = LQ.back();
    skipThisLoop = false;
    redoThisLoop = false;

    // Run all passes on current SCC
    for (unsigned Index = 0; Index < getNumContainedPasses(); ++Index) {  
        
      Pass *P = getContainedPass(Index);
      AnalysisUsage AnUsage;
      P->getAnalysisUsage(AnUsage);

      dumpPassInfo(P, EXECUTION_MSG, ON_LOOP_MSG, "");
      dumpAnalysisSetInfo("Required", P, AnUsage.getRequiredSet());

      initializeAnalysisImpl(P);

      StartPassTimer(P);
      LoopPass *LP = dynamic_cast<LoopPass *>(P);
      assert (LP && "Invalid LPPassManager member");
      LP->runOnLoop(L, *this);
      StopPassTimer(P);

      if (Changed)
        dumpPassInfo(P, MODIFICATION_MSG, ON_LOOP_MSG, "");
      dumpAnalysisSetInfo("Preserved", P, AnUsage.getPreservedSet());
      
      removeNotPreservedAnalysis(P);
      recordAvailableAnalysis(P);
      removeDeadPasses(P, "", ON_LOOP_MSG);

      if (skipThisLoop)
        // Do not run other passes on this loop.
        break;
    }
    
    // Pop the loop from queue after running all passes.
    LQ.pop_back();
    
    if (redoThisLoop)
      LQ.push_back(L);
  }
  
  // Finalization
  for (unsigned Index = 0; Index < getNumContainedPasses(); ++Index) {
    Pass *P = getContainedPass(Index);
    LoopPass *LP = dynamic_cast <LoopPass *>(P);
    if (LP)
      Changed |= LP->doFinalization();
  }

  return Changed;
}


//===----------------------------------------------------------------------===//
// LoopPass

/// Assign pass manager to manage this pass.
void LoopPass::assignPassManager(PMStack &PMS,
                                 PassManagerType PreferredType) {
  // Find LPPassManager 
  while (!PMS.empty()) {
    if (PMS.top()->getPassManagerType() > PMT_LoopPassManager)
      PMS.pop();
    else;
    break;
  }

  LPPassManager *LPPM = dynamic_cast<LPPassManager *>(PMS.top());

  // Create new Loop Pass Manager if it does not exist. 
  if (!LPPM) {

    assert (!PMS.empty() && "Unable to create Loop Pass Manager");
    PMDataManager *PMD = PMS.top();

    // [1] Create new Call Graph Pass Manager
    LPPM = new LPPassManager(PMD->getDepth() + 1);

    // [2] Set up new manager's top level manager
    PMTopLevelManager *TPM = PMD->getTopLevelManager();
    TPM->addIndirectPassManager(LPPM);

    // [3] Assign manager to manage this new manager. This may create
    // and push new managers into PMS
    Pass *P = dynamic_cast<Pass *>(LPPM);
    P->assignPassManager(PMS);

    // [4] Push new manager into PMS
    PMS.push(LPPM);
  }

  LPPM->add(this);
}

