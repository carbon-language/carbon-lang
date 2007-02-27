//===- LoopPass.h - LoopPass class ----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Devang Patel and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines LoopPass class. All loop optimization
// and transformation passes are derived from LoopPass.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LOOP_PASS_H
#define LLVM_LOOP_PASS_H

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Pass.h"
#include "llvm/PassManagers.h"
#include "llvm/Function.h"

namespace llvm {

class LPPassManager;
class Loop;
class Function;
class LoopQueue;

class LoopPass : public Pass {

 public:
  // runOnLoop - THis method should be implemented by the subclass to perform
  // whatever action is necessary for the specfied Loop. 
  virtual bool runOnLoop (Loop *L, LPPassManager &LPM) = 0;
  virtual bool runOnFunctionBody (Function &F, LPPassManager &LPM) { 
    return false; 
  }

  /// Assign pass manager to manager this pass
  virtual void assignPassManager(PMStack &PMS,
				 PassManagerType PMT = PMT_LoopPassManager);

};

class LPPassManager : public FunctionPass, public PMDataManager {

public:
  LPPassManager(int Depth);
  ~LPPassManager();

  /// run - Execute all of the passes scheduled for execution.  Keep track of
  /// whether any of the passes modifies the module, and if so, return true.
  bool runOnFunction(Function &F);

  /// Pass Manager itself does not invalidate any analysis info.
  void getAnalysisUsage(AnalysisUsage &Info) const {
    // LPPassManager needs LoopInfo. In the long term LoopInfo class will 
    // be consumed by LPPassManager.
    Info.addRequired<LoopInfo>();
    Info.setPreservesAll();
  }
  
  virtual const char *getPassName() const {
    return "Loop Pass Manager";
  }
  
  // Print passes managed by this manager
  void dumpPassStructure(unsigned Offset) {
    llvm::cerr << std::string(Offset*2, ' ') << "Loop Pass Manager\n";
    for (unsigned Index = 0; Index < getNumContainedPasses(); ++Index) {
      Pass *P = getContainedPass(Index);
      P->dumpPassStructure(Offset + 1);
      dumpLastUses(P, Offset+1);
    }
  }
  
  Pass *getContainedPass(unsigned N) {
    assert ( N < PassVector.size() && "Pass number out of range!");
    Pass *FP = static_cast<Pass *>(PassVector[N]);
    return FP;
  }

  virtual PassManagerType getPassManagerType() const { 
    return PMT_LoopPassManager; 
  }

public:
  // Delete loop from the loop queue. This is used by Loop pass to inform
  // Loop Pass Manager that it should skip rest of the passes for this loop.
  void deleteLoopFromQueue(Loop *L);

  // Reoptimize this loop. LPPassManager will re-insert this loop into the
  // queue. This allows LoopPass to change loop nest for the loop. This
  // utility may send LPPassManager into infinite loops so use caution.
  void redoLoop(Loop *L);
private:
  LoopQueue *LQ;
  bool skipThisLoop;
  bool redoThisLoop;
};

} // End llvm namespace

#endif
