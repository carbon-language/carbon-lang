//===- InlineCommon.cpp - Code common to all inliners ---------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the code shared between the LLVM inliners.  This
// implements all of the boring mechanics of the bottom-up inlining.
//
//===----------------------------------------------------------------------===//

#include "Inliner.h"
#include "llvm/Constants.h"   // ConstantPointerRef should die
#include "llvm/Module.h"
#include "llvm/iOther.h"
#include "llvm/iTerminators.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "Support/CommandLine.h"
#include "Support/Debug.h"
#include "Support/Statistic.h"
using namespace llvm;

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

    // Scan through and identify all call sites ahead of time so that we only
    // inline call sites in the original functions, not call sites that result
    // in inlining other functions.
    std::vector<CallSite> CallSites;
    for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB)
      for (BasicBlock::iterator I = BB->begin(); I != BB->end(); ++I) {
        CallSite CS = CallSite::get(I);
        if (CS.getInstruction() && CS.getCalledFunction() &&
            !CS.getCalledFunction()->isExternal())
          CallSites.push_back(CS);
      }

    // Now that we have all of the call sites, loop over them and inline them if
    // it looks profitable to do so.
    for (unsigned i = 0, e = CallSites.size(); i != e; ++i) {
      CallSite CS = CallSites[i];
      Function *Callee = CS.getCalledFunction();
      // Determine whether this is a function IN the SCC...
      bool inSCC = SCCFunctions.count(Callee);
    
      // If the policy determines that we should inline this function,
      // try to do so...
      int InlineCost = inSCC ? getRecursiveInlineCost(CS) : getInlineCost(CS);
      if (InlineCost >= (int)InlineThreshold) {
        DEBUG(std::cerr << "    NOT Inlining: cost=" << InlineCost
              << ", Call: " << *CS.getInstruction());
      } else {
        DEBUG(std::cerr << "    Inlining: cost=" << InlineCost
              << ", Call: " << *CS.getInstruction());
      
        Function *Caller = CS.getInstruction()->getParent()->getParent();

        // Attempt to inline the function...
        if (InlineFunction(CS)) {
          ++NumInlined;
  
          if (Callee->hasOneUse())
            if (ConstantPointerRef *CPR =
                dyn_cast<ConstantPointerRef>(Callee->use_back()))
              if (CPR->use_empty())
                CPR->destroyConstant();
        
          // If we inlined the last possible call site to the function,
          // delete the function body now.
          if (Callee->use_empty() && Callee != Caller &&
              (Callee->hasInternalLinkage() || Callee->hasLinkOnceLinkage())) {
            DEBUG(std::cerr << "    -> Deleting dead function: "
                  << (void*)Callee << Callee->getName() << "\n");
            std::set<Function*>::iterator I = SCCFunctions.find(Callee);
            if (I != SCCFunctions.end())    // Remove function from this SCC.
              SCCFunctions.erase(I);

            Callee->getParent()->getFunctionList().erase(Callee);
            ++NumDeleted;
          }
          Changed = true;
        }
      }
    }
  }

  return Changed;
}

