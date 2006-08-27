//===- Steensgaard.cpp - Context Insensitive Alias Analysis ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass uses the data structure graphs to implement a simple context
// insensitive alias analysis.  It does this by computing the local analysis
// graphs for all of the functions, then merging them together into a single big
// graph without cloning.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DataStructure/DataStructure.h"
#include "llvm/Analysis/DataStructure/DSGraph.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Module.h"
#include "llvm/Support/Debug.h"
#include <iostream>
using namespace llvm;

namespace {
  class Steens : public ModulePass, public AliasAnalysis {
    DSGraph *ResultGraph;

    EquivalenceClasses<GlobalValue*> GlobalECs;  // Always empty
  public:
    Steens() : ResultGraph(0) {}
    ~Steens() {
      releaseMyMemory();
      assert(ResultGraph == 0 && "releaseMemory not called?");
    }

    //------------------------------------------------
    // Implement the Pass API
    //

    // run - Build up the result graph, representing the pointer graph for the
    // program.
    //
    bool runOnModule(Module &M);

    virtual void releaseMyMemory() { delete ResultGraph; ResultGraph = 0; }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AliasAnalysis::getAnalysisUsage(AU);
      AU.setPreservesAll();                    // Does not transform code...
      AU.addRequired<LocalDataStructures>();   // Uses local dsgraph
    }

    // print - Implement the Pass::print method...
    void print(std::ostream &O, const Module *M) const {
      assert(ResultGraph && "Result graph has not yet been computed!");
      ResultGraph->writeGraphToFile(O, "steensgaards");
    }

    //------------------------------------------------
    // Implement the AliasAnalysis API
    //

    AliasResult alias(const Value *V1, unsigned V1Size,
                      const Value *V2, unsigned V2Size);

    ModRefResult getModRefInfo(CallSite CS, Value *P, unsigned Size);

  private:
    void ResolveFunctionCall(Function *F, const DSCallSite &Call,
                             DSNodeHandle &RetVal);
  };

  // Register the pass...
  RegisterPass<Steens> X("steens-aa",
                         "Steensgaard's alias analysis (DSGraph based)");

  // Register as an implementation of AliasAnalysis
  RegisterAnalysisGroup<AliasAnalysis, Steens> Y;
}

ModulePass *llvm::createSteensgaardPass() { return new Steens(); }

/// ResolveFunctionCall - Resolve the actual arguments of a call to function F
/// with the specified call site descriptor.  This function links the arguments
/// and the return value for the call site context-insensitively.
///
void Steens::ResolveFunctionCall(Function *F, const DSCallSite &Call,
                                 DSNodeHandle &RetVal) {
  assert(ResultGraph != 0 && "Result graph not allocated!");
  DSGraph::ScalarMapTy &ValMap = ResultGraph->getScalarMap();

  // Handle the return value of the function...
  if (Call.getRetVal().getNode() && RetVal.getNode())
    RetVal.mergeWith(Call.getRetVal());

  // Loop over all pointer arguments, resolving them to their provided pointers
  unsigned PtrArgIdx = 0;
  for (Function::arg_iterator AI = F->arg_begin(), AE = F->arg_end();
       AI != AE && PtrArgIdx < Call.getNumPtrArgs(); ++AI) {
    DSGraph::ScalarMapTy::iterator I = ValMap.find(AI);
    if (I != ValMap.end())    // If its a pointer argument...
      I->second.mergeWith(Call.getPtrArg(PtrArgIdx++));
  }
}


/// run - Build up the result graph, representing the pointer graph for the
/// program.
///
bool Steens::runOnModule(Module &M) {
  InitializeAliasAnalysis(this);
  assert(ResultGraph == 0 && "Result graph already allocated!");
  LocalDataStructures &LDS = getAnalysis<LocalDataStructures>();

  // Create a new, empty, graph...
  ResultGraph = new DSGraph(GlobalECs, getTargetData());
  ResultGraph->spliceFrom(LDS.getGlobalsGraph());

  // Loop over the rest of the module, merging graphs for non-external functions
  // into this graph.
  //
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    if (!I->isExternal())
      ResultGraph->spliceFrom(LDS.getDSGraph(*I));

  ResultGraph->removeTriviallyDeadNodes();

  // FIXME: Must recalculate and use the Incomplete markers!!

  // Now that we have all of the graphs inlined, we can go about eliminating
  // call nodes...
  //
  std::list<DSCallSite> &Calls = ResultGraph->getAuxFunctionCalls();
  assert(Calls.empty() && "Aux call list is already in use??");

  // Start with a copy of the original call sites.
  Calls = ResultGraph->getFunctionCalls();

  for (std::list<DSCallSite>::iterator CI = Calls.begin(), E = Calls.end();
       CI != E;) {
    DSCallSite &CurCall = *CI++;

    // Loop over the called functions, eliminating as many as possible...
    std::vector<Function*> CallTargets;
    if (CurCall.isDirectCall())
      CallTargets.push_back(CurCall.getCalleeFunc());
    else
      CurCall.getCalleeNode()->addFullFunctionList(CallTargets);

    for (unsigned c = 0; c != CallTargets.size(); ) {
      // If we can eliminate this function call, do so!
      Function *F = CallTargets[c];
      if (!F->isExternal()) {
        ResolveFunctionCall(F, CurCall, ResultGraph->getReturnNodes()[F]);
        CallTargets[c] = CallTargets.back();
        CallTargets.pop_back();
      } else
        ++c;  // Cannot eliminate this call, skip over it...
    }

    if (CallTargets.empty()) {        // Eliminated all calls?
      std::list<DSCallSite>::iterator I = CI;
      Calls.erase(--I);               // Remove entry
    }
  }

  // Remove our knowledge of what the return values of the functions are, except
  // for functions that are externally visible from this module (e.g. main).  We
  // keep these functions so that their arguments are marked incomplete.
  for (DSGraph::ReturnNodesTy::iterator I =
         ResultGraph->getReturnNodes().begin(),
         E = ResultGraph->getReturnNodes().end(); I != E; )
    if (I->first->hasInternalLinkage())
      ResultGraph->getReturnNodes().erase(I++);
    else
      ++I;

  // Update the "incomplete" markers on the nodes, ignoring unknownness due to
  // incoming arguments...
  ResultGraph->maskIncompleteMarkers();
  ResultGraph->markIncompleteNodes(DSGraph::IgnoreGlobals |
                                   DSGraph::MarkFormalArgs);

  // Remove any nodes that are dead after all of the merging we have done...
  // FIXME: We should be able to disable the globals graph for steens!
  //ResultGraph->removeDeadNodes(DSGraph::KeepUnreachableGlobals);

  DEBUG(print(std::cerr, &M));
  return false;
}

AliasAnalysis::AliasResult Steens::alias(const Value *V1, unsigned V1Size,
                                         const Value *V2, unsigned V2Size) {
  assert(ResultGraph && "Result graph has not been computed yet!");

  DSGraph::ScalarMapTy &GSM = ResultGraph->getScalarMap();

  DSGraph::ScalarMapTy::iterator I = GSM.find(const_cast<Value*>(V1));
  DSGraph::ScalarMapTy::iterator J = GSM.find(const_cast<Value*>(V2));
  if (I != GSM.end() && !I->second.isNull() &&
      J != GSM.end() && !J->second.isNull()) {
    DSNodeHandle &V1H = I->second;
    DSNodeHandle &V2H = J->second;

    // If at least one of the nodes is complete, we can say something about
    // this.  If one is complete and the other isn't, then they are obviously
    // different nodes.  If they are both complete, we can't say anything
    // useful.
    if (I->second.getNode()->isComplete() ||
        J->second.getNode()->isComplete()) {
      // If the two pointers point to different data structure graph nodes, they
      // cannot alias!
      if (V1H.getNode() != V2H.getNode())
        return NoAlias;

      // See if they point to different offsets...  if so, we may be able to
      // determine that they do not alias...
      unsigned O1 = I->second.getOffset(), O2 = J->second.getOffset();
      if (O1 != O2) {
        if (O2 < O1) {    // Ensure that O1 <= O2
          std::swap(V1, V2);
          std::swap(O1, O2);
          std::swap(V1Size, V2Size);
        }

        if (O1+V1Size <= O2)
          return NoAlias;
      }
    }
  }

  // If we cannot determine alias properties based on our graph, fall back on
  // some other AA implementation.
  //
  return AliasAnalysis::alias(V1, V1Size, V2, V2Size);
}

AliasAnalysis::ModRefResult
Steens::getModRefInfo(CallSite CS, Value *P, unsigned Size) {
  AliasAnalysis::ModRefResult Result = ModRef;

  // Find the node in question.
  DSGraph::ScalarMapTy &GSM = ResultGraph->getScalarMap();
  DSGraph::ScalarMapTy::iterator I = GSM.find(P);

  if (I != GSM.end() && !I->second.isNull()) {
    DSNode *N = I->second.getNode();
    if (N->isComplete()) {
      // If this is a direct call to an external function, and if the pointer
      // points to a complete node, the external function cannot modify or read
      // the value (we know it's not passed out of the program!).
      if (Function *F = CS.getCalledFunction())
        if (F->isExternal())
          return NoModRef;

      // Otherwise, if the node is complete, but it is only M or R, return this.
      // This can be useful for globals that should be marked const but are not.
      if (!N->isModified())
        Result = (ModRefResult)(Result & ~Mod);
      if (!N->isRead())
        Result = (ModRefResult)(Result & ~Ref);
    }
  }

  return (ModRefResult)(Result & AliasAnalysis::getModRefInfo(CS, P, Size));
}
