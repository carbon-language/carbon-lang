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

/// run - Execute all of the passes scheduled for execution.  Keep track of
/// whether any of the passes modifies the function, and if so, return true.
bool LPPassManager::runOnFunction(Function &F) {
  LoopInfo &LI = getAnalysis<LoopInfo>();
  bool Changed = false;

  std::string Msg1 = "Executing Pass '";
  std::string Msg3 = "' Made Modification '";

  // Walk Loops
  for (LoopInfo::iterator I = LI.begin(), E = LI.end(); I != E; ++I) {

    Loop *L  = *I;
    // Run all passes on current SCC
    for (unsigned Index = 0; Index < getNumContainedPasses(); ++Index) {  

      Pass *P = getContainedPass(Index);
      AnalysisUsage AnUsage;
      P->getAnalysisUsage(AnUsage);

      std::string Msg2 = "' on Loop ...\n'";
      dumpPassInfo(P, Msg1, Msg2);
      dumpAnalysisSetInfo("Required", P, AnUsage.getRequiredSet());

      initializeAnalysisImpl(P);

      StartPassTimer(P);
      LoopPass *LP = dynamic_cast<LoopPass *>(P);
      assert (LP && "Invalid LPPassManager member");
      LP->runOnLoop(*L, *this);
      StopPassTimer(P);

      if (Changed)
	dumpPassInfo(P, Msg3, Msg2);
      dumpAnalysisSetInfo("Preserved", P, AnUsage.getPreservedSet());
      
      removeNotPreservedAnalysis(P);
      recordAvailableAnalysis(P);
      removeDeadPasses(P, Msg2);
    }
  }

  return Changed;
}


