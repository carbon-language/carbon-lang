//===- Inliner.cpp - Code common to all inliners --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the mechanics required to implement inlining without
// missing any calls and updating the call graph.  The decisions of which calls
// are profitable to inline are implemented elsewhere.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "inline"
#include "llvm/Module.h"
#include "llvm/Instructions.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/IPO/InlinerPass.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/Statistic.h"
#include <set>
using namespace llvm;

STATISTIC(NumInlined, "Number of functions inlined");
STATISTIC(NumDeleted, "Number of functions deleted because all callers found");

static cl::opt<int>
InlineLimit("inline-threshold", cl::Hidden, cl::init(200),
        cl::desc("Control the amount of inlining to perform (default = 200)"));

Inliner::Inliner(void *ID) 
  : CallGraphSCCPass(ID), InlineThreshold(InlineLimit) {}

Inliner::Inliner(void *ID, int Threshold) 
  : CallGraphSCCPass(ID), InlineThreshold(Threshold) {}

/// getAnalysisUsage - For this class, we declare that we require and preserve
/// the call graph.  If the derived class implements this method, it should
/// always explicitly call the implementation here.
void Inliner::getAnalysisUsage(AnalysisUsage &Info) const {
  Info.addRequired<TargetData>();
  CallGraphSCCPass::getAnalysisUsage(Info);
}

// InlineCallIfPossible - If it is possible to inline the specified call site,
// do so and update the CallGraph for this operation.
static bool InlineCallIfPossible(CallSite CS, CallGraph &CG,
                                 const std::set<Function*> &SCCFunctions,
                                 const TargetData &TD) {
  Function *Callee = CS.getCalledFunction();
  if (!InlineFunction(CS, &CG, &TD)) return false;

  // If we inlined the last possible call site to the function, delete the
  // function body now.
  if (Callee->use_empty() && Callee->hasInternalLinkage() &&
      !SCCFunctions.count(Callee)) {
    DOUT << "    -> Deleting dead function: " << Callee->getName() << "\n";
    CallGraphNode *CalleeNode = CG[Callee];

    // Remove any call graph edges from the callee to its callees.
    CalleeNode->removeAllCalledFunctions();

    // Removing the node for callee from the call graph and delete it.
    delete CG.removeFunctionFromModule(CalleeNode);
    ++NumDeleted;
  }
  return true;
}

bool Inliner::runOnSCC(const std::vector<CallGraphNode*> &SCC) {
  CallGraph &CG = getAnalysis<CallGraph>();

  std::set<Function*> SCCFunctions;
  DOUT << "Inliner visiting SCC:";
  for (unsigned i = 0, e = SCC.size(); i != e; ++i) {
    Function *F = SCC[i]->getFunction();
    if (F) SCCFunctions.insert(F);
    DOUT << " " << (F ? F->getName() : "INDIRECTNODE");
  }

  // Scan through and identify all call sites ahead of time so that we only
  // inline call sites in the original functions, not call sites that result
  // from inlining other functions.
  std::vector<CallSite> CallSites;

  for (unsigned i = 0, e = SCC.size(); i != e; ++i)
    if (Function *F = SCC[i]->getFunction())
      for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB)
        for (BasicBlock::iterator I = BB->begin(); I != BB->end(); ++I) {
          CallSite CS = CallSite::get(I);
          if (CS.getInstruction() && (!CS.getCalledFunction() ||
                                      !CS.getCalledFunction()->isDeclaration()))
            CallSites.push_back(CS);
        }

  DOUT << ": " << CallSites.size() << " call sites.\n";

  // Now that we have all of the call sites, move the ones to functions in the
  // current SCC to the end of the list.
  unsigned FirstCallInSCC = CallSites.size();
  for (unsigned i = 0; i < FirstCallInSCC; ++i)
    if (Function *F = CallSites[i].getCalledFunction())
      if (SCCFunctions.count(F))
        std::swap(CallSites[i--], CallSites[--FirstCallInSCC]);

  // Now that we have all of the call sites, loop over them and inline them if
  // it looks profitable to do so.
  bool Changed = false;
  bool LocalChange;
  do {
    LocalChange = false;
    // Iterate over the outer loop because inlining functions can cause indirect
    // calls to become direct calls.
    for (unsigned CSi = 0; CSi != CallSites.size(); ++CSi)
      if (Function *Callee = CallSites[CSi].getCalledFunction()) {
        // Calls to external functions are never inlinable.
        if (Callee->isDeclaration() ||
            CallSites[CSi].getInstruction()->getParent()->getParent() ==Callee){
          if (SCC.size() == 1) {
            std::swap(CallSites[CSi], CallSites.back());
            CallSites.pop_back();
          } else {
            // Keep the 'in SCC / not in SCC' boundary correct.
            CallSites.erase(CallSites.begin()+CSi);
          }
          --CSi;
          continue;
        }

        // If the policy determines that we should inline this function,
        // try to do so.
        CallSite CS = CallSites[CSi];
        int InlineCost = getInlineCost(CS);
        float FudgeFactor = getInlineFudgeFactor(CS);
        
        int CurrentThreshold = InlineThreshold;
        Function *Fn = CS.getCaller();
        if (Fn && Fn->hasNote(ParamAttr::FN_NOTE_OptimizeForSize)
            && InlineThreshold != 50) {
          CurrentThreshold = 50;
        }
        
        if (InlineCost >= (int)(CurrentThreshold * FudgeFactor)) {
          DOUT << "    NOT Inlining: cost=" << InlineCost
               << ", Call: " << *CS.getInstruction();
        } else {
          DOUT << "    Inlining: cost=" << InlineCost
               << ", Call: " << *CS.getInstruction();

          // Attempt to inline the function...
          if (InlineCallIfPossible(CS, CG, SCCFunctions, 
                                   getAnalysis<TargetData>())) {
            // Remove this call site from the list.  If possible, use 
            // swap/pop_back for efficiency, but do not use it if doing so would
            // move a call site to a function in this SCC before the
            // 'FirstCallInSCC' barrier.
            if (SCC.size() == 1) {
              std::swap(CallSites[CSi], CallSites.back());
              CallSites.pop_back();
            } else {
              CallSites.erase(CallSites.begin()+CSi);
            }
            --CSi;

            ++NumInlined;
            Changed = true;
            LocalChange = true;
          }
        }
      }
  } while (LocalChange);

  return Changed;
}

// doFinalization - Remove now-dead linkonce functions at the end of
// processing to avoid breaking the SCC traversal.
bool Inliner::doFinalization(CallGraph &CG) {
  std::set<CallGraphNode*> FunctionsToRemove;

  // Scan for all of the functions, looking for ones that should now be removed
  // from the program.  Insert the dead ones in the FunctionsToRemove set.
  for (CallGraph::iterator I = CG.begin(), E = CG.end(); I != E; ++I) {
    CallGraphNode *CGN = I->second;
    if (Function *F = CGN ? CGN->getFunction() : 0) {
      // If the only remaining users of the function are dead constants, remove
      // them.
      F->removeDeadConstantUsers();

      if ((F->hasLinkOnceLinkage() || F->hasInternalLinkage()) &&
          F->use_empty()) {

        // Remove any call graph edges from the function to its callees.
        CGN->removeAllCalledFunctions();

        // Remove any edges from the external node to the function's call graph
        // node.  These edges might have been made irrelegant due to
        // optimization of the program.
        CG.getExternalCallingNode()->removeAnyCallEdgeTo(CGN);

        // Removing the node for callee from the call graph and delete it.
        FunctionsToRemove.insert(CGN);
      }
    }
  }

  // Now that we know which functions to delete, do so.  We didn't want to do
  // this inline, because that would invalidate our CallGraph::iterator
  // objects. :(
  bool Changed = false;
  for (std::set<CallGraphNode*>::iterator I = FunctionsToRemove.begin(),
         E = FunctionsToRemove.end(); I != E; ++I) {
    delete CG.removeFunctionFromModule(*I);
    ++NumDeleted;
    Changed = true;
  }

  return Changed;
}
