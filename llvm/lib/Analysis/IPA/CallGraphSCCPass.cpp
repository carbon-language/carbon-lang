//===- CallGraphSCCPass.cpp - Pass that operates BU on call graph ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the CallGraphSCCPass class, which is used for passes
// which are implemented as bottom-up traversals on the call graph.  Because
// there may be cycles in the call graph, passes of this type operate on the
// call-graph in SCC order: that is, they process function bottom-up, except for
// recursive functions, which they process all at once.
//
//===----------------------------------------------------------------------===//

#include "llvm/CallGraphSCCPass.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/PassManagers.h"
#include "llvm/Function.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
// CGPassManager
//
/// CGPassManager manages FPPassManagers and CalLGraphSCCPasses.

namespace {

class CGPassManager : public ModulePass, public PMDataManager {
public:
  static char ID;
  explicit CGPassManager(int Depth) 
    : ModulePass(&ID), PMDataManager(Depth) { }

  /// run - Execute all of the passes scheduled for execution.  Keep track of
  /// whether any of the passes modifies the module, and if so, return true.
  bool runOnModule(Module &M);

  bool doInitialization(CallGraph &CG);
  bool doFinalization(CallGraph &CG);

  /// Pass Manager itself does not invalidate any analysis info.
  void getAnalysisUsage(AnalysisUsage &Info) const {
    // CGPassManager walks SCC and it needs CallGraph.
    Info.addRequired<CallGraph>();
    Info.setPreservesAll();
  }

  virtual const char *getPassName() const {
    return "CallGraph Pass Manager";
  }

  // Print passes managed by this manager
  void dumpPassStructure(unsigned Offset) {
    errs().indent(Offset*2) << "Call Graph SCC Pass Manager\n";
    for (unsigned Index = 0; Index < getNumContainedPasses(); ++Index) {
      Pass *P = getContainedPass(Index);
      P->dumpPassStructure(Offset + 1);
      dumpLastUses(P, Offset+1);
    }
  }

  Pass *getContainedPass(unsigned N) {
    assert(N < PassVector.size() && "Pass number out of range!");
    return static_cast<Pass *>(PassVector[N]);
  }

  virtual PassManagerType getPassManagerType() const { 
    return PMT_CallGraphPassManager; 
  }
  
private:
  bool RunPassOnSCC(Pass *P, std::vector<CallGraphNode*> &CurSCC);
};

} // end anonymous namespace.

char CGPassManager::ID = 0;

bool CGPassManager::RunPassOnSCC(Pass *P, std::vector<CallGraphNode*> &CurSCC) {
  bool Changed = false;
  if (CallGraphSCCPass *CGSP = dynamic_cast<CallGraphSCCPass*>(P)) {
    StartPassTimer(P);
    Changed = CGSP->runOnSCC(CurSCC);
    StopPassTimer(P);
    return Changed;
  }
  
  StartPassTimer(P);
  FPPassManager *FPP = dynamic_cast<FPPassManager *>(P);
  assert(FPP && "Invalid CGPassManager member");
  
  // Run pass P on all functions in the current SCC.
  for (unsigned i = 0, e = CurSCC.size(); i != e; ++i) {
    if (Function *F = CurSCC[i]->getFunction()) {
      dumpPassInfo(P, EXECUTION_MSG, ON_FUNCTION_MSG, F->getName());
      Changed |= FPP->runOnFunction(*F);
    }
  }
  StopPassTimer(P);
  
  return Changed;
}


/// run - Execute all of the passes scheduled for execution.  Keep track of
/// whether any of the passes modifies the module, and if so, return true.
bool CGPassManager::runOnModule(Module &M) {
  CallGraph &CG = getAnalysis<CallGraph>();
  bool Changed = doInitialization(CG);

  std::vector<CallGraphNode*> CurSCC;
  
  // Walk the callgraph in bottom-up SCC order.
  for (scc_iterator<CallGraph*> CGI = scc_begin(&CG), E = scc_end(&CG);
       CGI != E;) {
    // Copy the current SCC and increment past it so that the pass can hack
    // on the SCC if it wants to without invalidating our iterator.
    CurSCC = *CGI;
    ++CGI;
    
    
    // Run all passes on current SCC.
    for (unsigned PassNo = 0, e = getNumContainedPasses();
         PassNo != e; ++PassNo) {
      Pass *P = getContainedPass(PassNo);

      dumpPassInfo(P, EXECUTION_MSG, ON_CG_MSG, "");
      dumpRequiredSet(P);

      initializeAnalysisImpl(P);

      // Actually run this pass on the current SCC.
      Changed |= RunPassOnSCC(P, CurSCC);

      if (Changed)
        dumpPassInfo(P, MODIFICATION_MSG, ON_CG_MSG, "");
      dumpPreservedSet(P);

      verifyPreservedAnalysis(P);      
      removeNotPreservedAnalysis(P);
      recordAvailableAnalysis(P);
      removeDeadPasses(P, "", ON_CG_MSG);
    }
  }
  Changed |= doFinalization(CG);
  return Changed;
}

/// Initialize CG
bool CGPassManager::doInitialization(CallGraph &CG) {
  bool Changed = false;
  for (unsigned Index = 0; Index < getNumContainedPasses(); ++Index) {  
    Pass *P = getContainedPass(Index);
    if (CallGraphSCCPass *CGSP = dynamic_cast<CallGraphSCCPass *>(P)) {
      Changed |= CGSP->doInitialization(CG);
    } else {
      FPPassManager *FP = dynamic_cast<FPPassManager *>(P);
      assert (FP && "Invalid CGPassManager member");
      Changed |= FP->doInitialization(CG.getModule());
    }
  }
  return Changed;
}

/// Finalize CG
bool CGPassManager::doFinalization(CallGraph &CG) {
  bool Changed = false;
  for (unsigned Index = 0; Index < getNumContainedPasses(); ++Index) {  
    Pass *P = getContainedPass(Index);
    if (CallGraphSCCPass *CGSP = dynamic_cast<CallGraphSCCPass *>(P)) {
      Changed |= CGSP->doFinalization(CG);
    } else {
      FPPassManager *FP = dynamic_cast<FPPassManager *>(P);
      assert (FP && "Invalid CGPassManager member");
      Changed |= FP->doFinalization(CG.getModule());
    }
  }
  return Changed;
}

/// Assign pass manager to manage this pass.
void CallGraphSCCPass::assignPassManager(PMStack &PMS,
                                         PassManagerType PreferredType) {
  // Find CGPassManager 
  while (!PMS.empty() &&
         PMS.top()->getPassManagerType() > PMT_CallGraphPassManager)
    PMS.pop();

  assert (!PMS.empty() && "Unable to handle Call Graph Pass");
  CGPassManager *CGP = dynamic_cast<CGPassManager *>(PMS.top());

  // Create new Call Graph SCC Pass Manager if it does not exist. 
  if (!CGP) {

    assert (!PMS.empty() && "Unable to create Call Graph Pass Manager");
    PMDataManager *PMD = PMS.top();

    // [1] Create new Call Graph Pass Manager
    CGP = new CGPassManager(PMD->getDepth() + 1);

    // [2] Set up new manager's top level manager
    PMTopLevelManager *TPM = PMD->getTopLevelManager();
    TPM->addIndirectPassManager(CGP);

    // [3] Assign manager to manage this new manager. This may create
    // and push new managers into PMS
    Pass *P = dynamic_cast<Pass *>(CGP);
    TPM->schedulePass(P);

    // [4] Push new manager into PMS
    PMS.push(CGP);
  }

  CGP->add(this);
}

/// getAnalysisUsage - For this class, we declare that we require and preserve
/// the call graph.  If the derived class implements this method, it should
/// always explicitly call the implementation here.
void CallGraphSCCPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<CallGraph>();
  AU.addPreserved<CallGraph>();
}
