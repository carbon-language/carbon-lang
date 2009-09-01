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

#define DEBUG_TYPE "cgscc-passmgr"
#include "llvm/CallGraphSCCPass.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/PassManagers.h"
#include "llvm/Function.h"
#include "llvm/Support/Debug.h"
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
  bool RunPassOnSCC(Pass *P, std::vector<CallGraphNode*> &CurSCC,
                    CallGraph &CG, bool &CallGraphUpToDate);
  void RefreshCallGraph(std::vector<CallGraphNode*> &CurSCC, CallGraph &CG);
};

} // end anonymous namespace.

char CGPassManager::ID = 0;

bool CGPassManager::RunPassOnSCC(Pass *P, std::vector<CallGraphNode*> &CurSCC,
                                 CallGraph &CG, bool &CallGraphUpToDate) {
  bool Changed = false;
  if (CallGraphSCCPass *CGSP = dynamic_cast<CallGraphSCCPass*>(P)) {
    if (!CallGraphUpToDate) {
      RefreshCallGraph(CurSCC, CG);
      CallGraphUpToDate = true;
    }
    
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
  
  // The function pass(es) modified the IR, they may have clobbered the
  // callgraph.
  if (Changed && CallGraphUpToDate) {
    DEBUG(errs() << "CGSCCPASSMGR: Pass Dirtied SCC: "
                 << P->getPassName() << '\n');
    CallGraphUpToDate = false;
  }
  return Changed;
}

void CGPassManager::RefreshCallGraph(std::vector<CallGraphNode*> &CurSCC,
                                     CallGraph &CG) {
  DenseMap<Value*, CallGraphNode*> CallSites;
  
  DEBUG(errs() << "CGSCCPASSMGR: Refreshing SCC with " << CurSCC.size()
               << " nodes:\n";
        for (unsigned i = 0, e = CurSCC.size(); i != e; ++i)
          CurSCC[i]->dump();
        );

  bool MadeChange = false;
  
  // Scan all functions in the SCC.
  for (unsigned sccidx = 0, e = CurSCC.size(); sccidx != e; ++sccidx) {
    CallGraphNode *CGN = CurSCC[sccidx];
    Function *F = CGN->getFunction();
    if (F == 0 || F->isDeclaration()) continue;
    
    // Walk the function body looking for call sites.  Sync up the call sites in
    // CGN with those actually in the function.
    
    // Get the set of call sites currently in the function.
    for (CallGraphNode::iterator I = CGN->begin(), E = CGN->end(); I != E; ){
      // If this call site is null, then the function pass deleted the call
      // entirely and the WeakVH nulled it out.  
      if (I->first == 0 ||
          // If we've already seen this call site, then the FunctionPass RAUW'd
          // one call with another, which resulted in two "uses" in the edge
          // list of the same call.
          CallSites.count(I->first) ||

          // If the call edge is not from a call or invoke, then the function
          // pass RAUW'd a call with another value.  This can happen when
          // constant folding happens of well known functions etc.
          CallSite::get(I->first).getInstruction() == 0) {
        // Just remove the edge from the set of callees.
        CGN->removeCallEdge(I);
        E = CGN->end();
        continue;
      }
      
      assert(!CallSites.count(I->first) &&
             "Call site occurs in node multiple times");
      CallSites.insert(std::make_pair(I->first, I->second));
      ++I;
    }
    
    // Loop over all of the instructions in the function, getting the callsites.
    for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB)
      for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I) {
        CallSite CS = CallSite::get(I);
        if (!CS.getInstruction()) continue;
        
        // If this call site already existed in the callgraph, just verify it
        // matches up to expectations and remove it from CallSites.
        DenseMap<Value*, CallGraphNode*>::iterator ExistingIt =
          CallSites.find(CS.getInstruction());
        if (ExistingIt != CallSites.end()) {
          CallGraphNode *ExistingNode = ExistingIt->second;

          // Remove from CallSites since we have now seen it.
          CallSites.erase(ExistingIt);
          
          // Verify that the callee is right.
          if (ExistingNode->getFunction() == CS.getCalledFunction())
            continue;
          
          // If not, we either went from a direct call to indirect, indirect to
          // direct, or direct to different direct.
          CallGraphNode *CalleeNode;
          if (Function *Callee = CS.getCalledFunction())
            CalleeNode = CG.getOrInsertFunction(Callee);
          else
            CalleeNode = CG.getCallsExternalNode();
          
          CGN->replaceCallSite(CS, CS, CalleeNode);
          MadeChange = true;
          continue;
        }
        
        // If the call site didn't exist in the CGN yet, add it.  We assume that
        // newly introduced call sites won't be indirect.  This could be fixed
        // in the future.
        CallGraphNode *CalleeNode;
        if (Function *Callee = CS.getCalledFunction())
          CalleeNode = CG.getOrInsertFunction(Callee);
        else
          CalleeNode = CG.getCallsExternalNode();
        
        CGN->addCalledFunction(CS, CalleeNode);
        MadeChange = true;
      }
    
    // After scanning this function, if we still have entries in callsites, then
    // they are dangling pointers.  WeakVH should save us for this, so abort if
    // this happens.
    assert(CallSites.empty() && "Dangling pointers found in call sites map");
    
    // Periodically do an explicit clear to remove tombstones when processing
    // large scc's.
    if ((sccidx & 15) == 0)
      CallSites.clear();
  }

  DEBUG(if (MadeChange) {
          errs() << "CGSCCPASSMGR: Refreshed SCC is now:\n";
          for (unsigned i = 0, e = CurSCC.size(); i != e; ++i)
            CurSCC[i]->dump();
         } else {
           errs() << "CGSCCPASSMGR: SCC Refresh didn't change call graph.\n";
         }
        );
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
    
    
    // CallGraphUpToDate - Keep track of whether the callgraph is known to be
    // up-to-date or not.  The CGSSC pass manager runs two types of passes:
    // CallGraphSCC Passes and other random function passes.  Because other
    // random function passes are not CallGraph aware, they may clobber the
    // call graph by introducing new calls or deleting other ones.  This flag
    // is set to false when we run a function pass so that we know to clean up
    // the callgraph when we need to run a CGSCCPass again.
    bool CallGraphUpToDate = true;
    
    // Run all passes on current SCC.
    for (unsigned PassNo = 0, e = getNumContainedPasses();
         PassNo != e; ++PassNo) {
      Pass *P = getContainedPass(PassNo);

      dumpPassInfo(P, EXECUTION_MSG, ON_CG_MSG, "");
      dumpRequiredSet(P);

      initializeAnalysisImpl(P);

      // Actually run this pass on the current SCC.
      Changed |= RunPassOnSCC(P, CurSCC, CG, CallGraphUpToDate);

      if (Changed)
        dumpPassInfo(P, MODIFICATION_MSG, ON_CG_MSG, "");
      dumpPreservedSet(P);

      verifyPreservedAnalysis(P);      
      removeNotPreservedAnalysis(P);
      recordAvailableAnalysis(P);
      removeDeadPasses(P, "", ON_CG_MSG);
    }
    
    // If the callgraph was left out of date (because the last pass run was a
    // functionpass), refresh it before we move on to the next SCC.
    if (!CallGraphUpToDate)
      RefreshCallGraph(CurSCC, CG);
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
