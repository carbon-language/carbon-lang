//===- InlineCommon.cpp - Code common to all inliners ---------------------===//
//
// This file implements the code shared between the LLVM inliners.  This
// implements all of the boring mechanics of the bottom-up inlining.
//
//===----------------------------------------------------------------------===//

#include "Inliner.h"
#include "llvm/Module.h"
#include "llvm/iOther.h"
#include "llvm/iTerminators.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "Support/CommandLine.h"
#include "Support/Debug.h"
#include "Support/Statistic.h"

namespace {
  Statistic<> NumInlined("inline", "Number of functions inlined");
  Statistic<> NumDeleted("inline", "Number of functions deleted because all callers found");
  cl::opt<unsigned>             // FIXME: 200 is VERY conservative
  InlineLimit("inline-threshold", cl::Hidden, cl::init(200),
              cl::desc("Control the amount of inlining to perform (default = 200)"));
}

Inliner::Inliner() : InlineThreshold(InlineLimit) {}

int Inliner::getRecursiveInlineCost(CallSite CS) {
  return getInlineCost(CS);
}

bool Inliner::runOnSCC(const std::vector<CallGraphNode*> &SCC) {
  CallGraph &CG = getAnalysis<CallGraph>();

  std::set<Function*> SCCFunctions;
  DEBUG(std::cerr << "Inliner visiting SCC:");
  for (unsigned i = 0, e = SCC.size(); i != e; ++i) {
    SCCFunctions.insert(SCC[i]->getFunction());
    DEBUG(std::cerr << " " << (SCC[i]->getFunction() ?
              SCC[i]->getFunction()->getName() : "INDIRECTNODE"));
  }
  DEBUG(std::cerr << "\n");

  bool Changed = false;
  for (std::set<Function*>::iterator SCCI = SCCFunctions.begin(),
         E = SCCFunctions.end(); SCCI != E; ++SCCI) {
    Function *F = *SCCI;
    if (F == 0 || F->isExternal()) continue;
    DEBUG(std::cerr << "  Inspecting function: " << F->getName() << "\n");

    for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB)
      for (BasicBlock::iterator I = BB->begin(); I != BB->end(); ) {
        bool ShouldInc = true;
        // Found a call or invoke instruction?
        if (isa<CallInst>(I) || isa<InvokeInst>(I)) {
          CallSite CS = CallSite::get(I);
          if (Function *Callee = CS.getCalledFunction())
            if (!Callee->isExternal()) {
              // Determine whether this is a function IN the SCC...
              bool inSCC = SCCFunctions.count(Callee);

              // If the policy determines that we should inline this function,
              // try to do so...
              int InlineCost = inSCC ? getRecursiveInlineCost(CS) :
                                       getInlineCost(CS);
              if (InlineCost < (int)InlineThreshold) {
                DEBUG(std::cerr << "    Inlining: cost=" << InlineCost
                                << ", Call: " << *CS.getInstruction());

                // Save an iterator to the instruction before the call if it
                // exists, otherwise get an iterator at the end of the
                // block... because the call will be destroyed.
                //
                BasicBlock::iterator SI;
                if (I != BB->begin()) {
                  SI = I; --SI;           // Instruction before the call...
                } else {
                  SI = BB->end();
                }
                
                if (performInlining(CS, SCCFunctions)) {
                  // Move to instruction before the call...
                  I = (SI == BB->end()) ? BB->begin() : SI;
                  ShouldInc = false; // Don't increment iterator until next time
                  Changed = true;
                }
              }
            }
        }
        if (ShouldInc) ++I;
      }
  }
  return Changed;
}

bool Inliner::performInlining(CallSite CS, std::set<Function*> &SCC) {
  Function *Callee = CS.getCalledFunction();
  Function *Caller = CS.getInstruction()->getParent()->getParent();

  // Attempt to inline the function...
  if (!InlineFunction(CS)) return false;
  ++NumInlined;
              
  // If we inlined the last possible call site to the function,
  // delete the function body now.
  if (Callee->use_empty() && Callee != Caller &&
      (Callee->hasInternalLinkage() || Callee->hasLinkOnceLinkage())) {
    DEBUG(std::cerr << "    -> Deleting dead function: "
                    << Callee->getName() << "\n");
    std::set<Function*>::iterator I = SCC.find(Callee);
    if (I != SCC.end())       // Remove function from this SCC...
      SCC.erase(I);

    Callee->getParent()->getFunctionList().erase(Callee);
    ++NumDeleted;
  }
  return true; 
}
