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
#include "llvm/IntrinsicInst.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/IPO/InlinerPass.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include <set>
using namespace llvm;

STATISTIC(NumInlined, "Number of functions inlined");
STATISTIC(NumCallsDeleted, "Number of call sites deleted, not inlined");
STATISTIC(NumDeleted, "Number of functions deleted because all callers found");
STATISTIC(NumMergedAllocas, "Number of allocas merged together");

static cl::opt<int>
InlineLimit("inline-threshold", cl::Hidden, cl::init(225), cl::ZeroOrMore,
        cl::desc("Control the amount of inlining to perform (default = 225)"));

Inliner::Inliner(void *ID) 
  : CallGraphSCCPass(ID), InlineThreshold(InlineLimit) {}

Inliner::Inliner(void *ID, int Threshold) 
  : CallGraphSCCPass(ID), InlineThreshold(Threshold) {}

/// getAnalysisUsage - For this class, we declare that we require and preserve
/// the call graph.  If the derived class implements this method, it should
/// always explicitly call the implementation here.
void Inliner::getAnalysisUsage(AnalysisUsage &Info) const {
  CallGraphSCCPass::getAnalysisUsage(Info);
}


typedef DenseMap<const ArrayType*, std::vector<AllocaInst*> >
InlinedArrayAllocasTy;

/// InlineCallIfPossible - If it is possible to inline the specified call site,
/// do so and update the CallGraph for this operation.
///
/// This function also does some basic book-keeping to update the IR.  The
/// InlinedArrayAllocas map keeps track of any allocas that are already
/// available from other  functions inlined into the caller.  If we are able to
/// inline this call site we attempt to reuse already available allocas or add
/// any new allocas to the set if not possible.
static bool InlineCallIfPossible(CallSite CS, CallGraph &CG,
                                 const TargetData *TD,
                                 InlinedArrayAllocasTy &InlinedArrayAllocas) {
  Function *Callee = CS.getCalledFunction();
  Function *Caller = CS.getCaller();

  // Try to inline the function.  Get the list of static allocas that were
  // inlined.
  SmallVector<AllocaInst*, 16> StaticAllocas;
  if (!InlineFunction(CS, &CG, TD, &StaticAllocas))
    return false;

  // If the inlined function had a higher stack protection level than the
  // calling function, then bump up the caller's stack protection level.
  if (Callee->hasFnAttr(Attribute::StackProtectReq))
    Caller->addFnAttr(Attribute::StackProtectReq);
  else if (Callee->hasFnAttr(Attribute::StackProtect) &&
           !Caller->hasFnAttr(Attribute::StackProtectReq))
    Caller->addFnAttr(Attribute::StackProtect);

  
  // Look at all of the allocas that we inlined through this call site.  If we
  // have already inlined other allocas through other calls into this function,
  // then we know that they have disjoint lifetimes and that we can merge them.
  //
  // There are many heuristics possible for merging these allocas, and the
  // different options have different tradeoffs.  One thing that we *really*
  // don't want to hurt is SRoA: once inlining happens, often allocas are no
  // longer address taken and so they can be promoted.
  //
  // Our "solution" for that is to only merge allocas whose outermost type is an
  // array type.  These are usually not promoted because someone is using a
  // variable index into them.  These are also often the most important ones to
  // merge.
  //
  // A better solution would be to have real memory lifetime markers in the IR
  // and not have the inliner do any merging of allocas at all.  This would
  // allow the backend to do proper stack slot coloring of all allocas that
  // *actually make it to the backend*, which is really what we want.
  //
  // Because we don't have this information, we do this simple and useful hack.
  //
  SmallPtrSet<AllocaInst*, 16> UsedAllocas;
  
  // Loop over all the allocas we have so far and see if they can be merged with
  // a previously inlined alloca.  If not, remember that we had it.
  for (unsigned AllocaNo = 0, e = StaticAllocas.size();
       AllocaNo != e; ++AllocaNo) {
    AllocaInst *AI = StaticAllocas[AllocaNo];
    
    // Don't bother trying to merge array allocations (they will usually be
    // canonicalized to be an allocation *of* an array), or allocations whose
    // type is not itself an array (because we're afraid of pessimizing SRoA).
    const ArrayType *ATy = dyn_cast<ArrayType>(AI->getAllocatedType());
    if (ATy == 0 || AI->isArrayAllocation())
      continue;
    
    // Get the list of all available allocas for this array type.
    std::vector<AllocaInst*> &AllocasForType = InlinedArrayAllocas[ATy];
    
    // Loop over the allocas in AllocasForType to see if we can reuse one.  Note
    // that we have to be careful not to reuse the same "available" alloca for
    // multiple different allocas that we just inlined, we use the 'UsedAllocas'
    // set to keep track of which "available" allocas are being used by this
    // function.  Also, AllocasForType can be empty of course!
    bool MergedAwayAlloca = false;
    for (unsigned i = 0, e = AllocasForType.size(); i != e; ++i) {
      AllocaInst *AvailableAlloca = AllocasForType[i];
      
      // The available alloca has to be in the right function, not in some other
      // function in this SCC.
      if (AvailableAlloca->getParent() != AI->getParent())
        continue;
      
      // If the inlined function already uses this alloca then we can't reuse
      // it.
      if (!UsedAllocas.insert(AvailableAlloca))
        continue;
      
      // Otherwise, we *can* reuse it, RAUW AI into AvailableAlloca and declare
      // success!
      DEBUG(dbgs() << "    ***MERGED ALLOCA: " << *AI);
      
      AI->replaceAllUsesWith(AvailableAlloca);
      AI->eraseFromParent();
      MergedAwayAlloca = true;
      ++NumMergedAllocas;
      break;
    }

    // If we already nuked the alloca, we're done with it.
    if (MergedAwayAlloca)
      continue;

    // If we were unable to merge away the alloca either because there are no
    // allocas of the right type available or because we reused them all
    // already, remember that this alloca came from an inlined function and mark
    // it used so we don't reuse it for other allocas from this inline
    // operation.
    AllocasForType.push_back(AI);
    UsedAllocas.insert(AI);
  }
  
  return true;
}

unsigned Inliner::getInlineThreshold(Function* Caller) const {
  if (Caller && !Caller->isDeclaration() &&
      Caller->hasFnAttr(Attribute::OptimizeForSize) &&
      InlineLimit.getNumOccurrences() == 0)
    return 75;
  else
    return InlineThreshold;
}

/// shouldInline - Return true if the inliner should attempt to inline
/// at the given CallSite.
bool Inliner::shouldInline(CallSite CS) {
  InlineCost IC = getInlineCost(CS);
  
  if (IC.isAlways()) {
    DEBUG(dbgs() << "    Inlining: cost=always"
          << ", Call: " << *CS.getInstruction() << "\n");
    return true;
  }
  
  if (IC.isNever()) {
    DEBUG(dbgs() << "    NOT Inlining: cost=never"
          << ", Call: " << *CS.getInstruction() << "\n");
    return false;
  }
  
  int Cost = IC.getValue();
  Function *Caller = CS.getCaller();
  int CurrentThreshold = getInlineThreshold(Caller);
  float FudgeFactor = getInlineFudgeFactor(CS);
  if (Cost >= (int)(CurrentThreshold * FudgeFactor)) {
    DEBUG(dbgs() << "    NOT Inlining: cost=" << Cost
          << ", Call: " << *CS.getInstruction() << "\n");
    return false;
  }
  
  // Try to detect the case where the current inlining candidate caller
  // (call it B) is a static function and is an inlining candidate elsewhere,
  // and the current candidate callee (call it C) is large enough that
  // inlining it into B would make B too big to inline later.  In these
  // circumstances it may be best not to inline C into B, but to inline B
  // into its callers.
  if (Caller->hasLocalLinkage()) {
    int TotalSecondaryCost = 0;
    bool outerCallsFound = false;
    bool allOuterCallsWillBeInlined = true;
    bool someOuterCallWouldNotBeInlined = false;
    for (Value::use_iterator I = Caller->use_begin(), E =Caller->use_end(); 
         I != E; ++I) {
      CallSite CS2 = CallSite::get(*I);

      // If this isn't a call to Caller (it could be some other sort
      // of reference) skip it.
      if (CS2.getInstruction() == 0 || CS2.getCalledFunction() != Caller)
        continue;

      InlineCost IC2 = getInlineCost(CS2);
      if (IC2.isNever())
        allOuterCallsWillBeInlined = false;
      if (IC2.isAlways() || IC2.isNever())
        continue;

      outerCallsFound = true;
      int Cost2 = IC2.getValue();
      Function *Caller2 = CS2.getCaller();
      int CurrentThreshold2 = getInlineThreshold(Caller2);
      float FudgeFactor2 = getInlineFudgeFactor(CS2);

      if (Cost2 >= (int)(CurrentThreshold2 * FudgeFactor2))
        allOuterCallsWillBeInlined = false;

      // See if we have this case.  We subtract off the penalty
      // for the call instruction, which we would be deleting.
      if (Cost2 < (int)(CurrentThreshold2 * FudgeFactor2) &&
          Cost2 + Cost - (InlineConstants::CallPenalty + 1) >= 
                (int)(CurrentThreshold2 * FudgeFactor2)) {
        someOuterCallWouldNotBeInlined = true;
        TotalSecondaryCost += Cost2;
      }
    }
    // If all outer calls to Caller would get inlined, the cost for the last
    // one is set very low by getInlineCost, in anticipation that Caller will
    // be removed entirely.  We did not account for this above unless there
    // is only one caller of Caller.
    if (allOuterCallsWillBeInlined && Caller->use_begin() != Caller->use_end())
      TotalSecondaryCost += InlineConstants::LastCallToStaticBonus;

    if (outerCallsFound && someOuterCallWouldNotBeInlined && 
        TotalSecondaryCost < Cost) {
      DEBUG(dbgs() << "    NOT Inlining: " << *CS.getInstruction() << 
           " Cost = " << Cost << 
           ", outer Cost = " << TotalSecondaryCost << '\n');
      return false;
    }
  }

  DEBUG(dbgs() << "    Inlining: cost=" << Cost
        << ", Call: " << *CS.getInstruction() << '\n');
  return true;
}

bool Inliner::runOnSCC(std::vector<CallGraphNode*> &SCC) {
  CallGraph &CG = getAnalysis<CallGraph>();
  const TargetData *TD = getAnalysisIfAvailable<TargetData>();

  SmallPtrSet<Function*, 8> SCCFunctions;
  DEBUG(dbgs() << "Inliner visiting SCC:");
  for (unsigned i = 0, e = SCC.size(); i != e; ++i) {
    Function *F = SCC[i]->getFunction();
    if (F) SCCFunctions.insert(F);
    DEBUG(dbgs() << " " << (F ? F->getName() : "INDIRECTNODE"));
  }

  // Scan through and identify all call sites ahead of time so that we only
  // inline call sites in the original functions, not call sites that result
  // from inlining other functions.
  SmallVector<CallSite, 16> CallSites;

  for (unsigned i = 0, e = SCC.size(); i != e; ++i) {
    Function *F = SCC[i]->getFunction();
    if (!F) continue;
    
    for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB)
      for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I) {
        CallSite CS = CallSite::get(I);
        // If this isn't a call, or it is a call to an intrinsic, it can
        // never be inlined.
        if (CS.getInstruction() == 0 || isa<IntrinsicInst>(I))
          continue;
        
        // If this is a direct call to an external function, we can never inline
        // it.  If it is an indirect call, inlining may resolve it to be a
        // direct call, so we keep it.
        if (CS.getCalledFunction() && CS.getCalledFunction()->isDeclaration())
          continue;
        
        CallSites.push_back(CS);
      }
  }

  DEBUG(dbgs() << ": " << CallSites.size() << " call sites.\n");

  // Now that we have all of the call sites, move the ones to functions in the
  // current SCC to the end of the list.
  unsigned FirstCallInSCC = CallSites.size();
  for (unsigned i = 0; i < FirstCallInSCC; ++i)
    if (Function *F = CallSites[i].getCalledFunction())
      if (SCCFunctions.count(F))
        std::swap(CallSites[i--], CallSites[--FirstCallInSCC]);

  
  InlinedArrayAllocasTy InlinedArrayAllocas;
  
  // Now that we have all of the call sites, loop over them and inline them if
  // it looks profitable to do so.
  bool Changed = false;
  bool LocalChange;
  do {
    LocalChange = false;
    // Iterate over the outer loop because inlining functions can cause indirect
    // calls to become direct calls.
    for (unsigned CSi = 0; CSi != CallSites.size(); ++CSi) {
      CallSite CS = CallSites[CSi];
      
      Function *Caller = CS.getCaller();
      Function *Callee = CS.getCalledFunction();

      // If this call site is dead and it is to a readonly function, we should
      // just delete the call instead of trying to inline it, regardless of
      // size.  This happens because IPSCCP propagates the result out of the
      // call and then we're left with the dead call.
      if (isInstructionTriviallyDead(CS.getInstruction())) {
        DEBUG(dbgs() << "    -> Deleting dead call: "
                     << *CS.getInstruction() << "\n");
        // Update the call graph by deleting the edge from Callee to Caller.
        CG[Caller]->removeCallEdgeFor(CS);
        CS.getInstruction()->eraseFromParent();
        ++NumCallsDeleted;
      } else {
        // We can only inline direct calls to non-declarations.
        if (Callee == 0 || Callee->isDeclaration()) continue;
      
        // If the policy determines that we should inline this function,
        // try to do so.
        if (!shouldInline(CS))
          continue;

        // Attempt to inline the function...
        if (!InlineCallIfPossible(CS, CG, TD, InlinedArrayAllocas))
          continue;
        ++NumInlined;
      }
      
      // If we inlined or deleted the last possible call site to the function,
      // delete the function body now.
      if (Callee && Callee->use_empty() && Callee->hasLocalLinkage() &&
          // TODO: Can remove if in SCC now.
          !SCCFunctions.count(Callee) &&
          
          // The function may be apparently dead, but if there are indirect
          // callgraph references to the node, we cannot delete it yet, this
          // could invalidate the CGSCC iterator.
          CG[Callee]->getNumReferences() == 0) {
        DEBUG(dbgs() << "    -> Deleting dead function: "
              << Callee->getName() << "\n");
        CallGraphNode *CalleeNode = CG[Callee];
        
        // Remove any call graph edges from the callee to its callees.
        CalleeNode->removeAllCalledFunctions();
        
        resetCachedCostInfo(Callee);
        
        // Removing the node for callee from the call graph and delete it.
        delete CG.removeFunctionFromModule(CalleeNode);
        ++NumDeleted;
      }
      
      // Remove any cached cost info for this caller, as inlining the
      // callee has increased the size of the caller (which may be the
      // same as the callee).
      resetCachedCostInfo(Caller);

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

      Changed = true;
      LocalChange = true;
    }
  } while (LocalChange);

  return Changed;
}

// doFinalization - Remove now-dead linkonce functions at the end of
// processing to avoid breaking the SCC traversal.
bool Inliner::doFinalization(CallGraph &CG) {
  return removeDeadFunctions(CG);
}

/// removeDeadFunctions - Remove dead functions that are not included in
/// DNR (Do Not Remove) list.
bool Inliner::removeDeadFunctions(CallGraph &CG, 
                                  SmallPtrSet<const Function *, 16> *DNR) {
  SmallPtrSet<CallGraphNode*, 16> FunctionsToRemove;

  // Scan for all of the functions, looking for ones that should now be removed
  // from the program.  Insert the dead ones in the FunctionsToRemove set.
  for (CallGraph::iterator I = CG.begin(), E = CG.end(); I != E; ++I) {
    CallGraphNode *CGN = I->second;
    if (CGN->getFunction() == 0)
      continue;
    
    Function *F = CGN->getFunction();
    
    // If the only remaining users of the function are dead constants, remove
    // them.
    F->removeDeadConstantUsers();

    if (DNR && DNR->count(F))
      continue;
    if (!F->hasLinkOnceLinkage() && !F->hasLocalLinkage() &&
        !F->hasAvailableExternallyLinkage())
      continue;
    if (!F->use_empty())
      continue;
    
    // Remove any call graph edges from the function to its callees.
    CGN->removeAllCalledFunctions();

    // Remove any edges from the external node to the function's call graph
    // node.  These edges might have been made irrelegant due to
    // optimization of the program.
    CG.getExternalCallingNode()->removeAnyCallEdgeTo(CGN);

    // Removing the node for callee from the call graph and delete it.
    FunctionsToRemove.insert(CGN);
  }

  // Now that we know which functions to delete, do so.  We didn't want to do
  // this inline, because that would invalidate our CallGraph::iterator
  // objects. :(
  //
  // Note that it doesn't matter that we are iterating over a non-stable set
  // here to do this, it doesn't matter which order the functions are deleted
  // in.
  bool Changed = false;
  for (SmallPtrSet<CallGraphNode*, 16>::iterator I = FunctionsToRemove.begin(),
       E = FunctionsToRemove.end(); I != E; ++I) {
    resetCachedCostInfo((*I)->getFunction());
    delete CG.removeFunctionFromModule(*I);
    ++NumDeleted;
    Changed = true;
  }

  return Changed;
}
