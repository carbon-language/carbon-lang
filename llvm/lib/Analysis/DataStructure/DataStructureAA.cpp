//===- DataStructureAA.cpp - Data Structure Based Alias Analysis ----------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This pass uses the top-down data structure graphs to implement a simple
// context sensitive alias analysis.
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/DataStructure.h"
#include "llvm/Analysis/DSGraph.h"
using namespace llvm;

namespace {
  class DSAA : public Pass, public AliasAnalysis {
    TDDataStructures *TD;
    BUDataStructures *BU;
  public:
    DSAA() : TD(0) {}

    //------------------------------------------------
    // Implement the Pass API
    //

    // run - Build up the result graph, representing the pointer graph for the
    // program.
    //
    bool run(Module &M) {
      InitializeAliasAnalysis(this);
      TD = &getAnalysis<TDDataStructures>();
      BU = &getAnalysis<BUDataStructures>();
      return false;
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AliasAnalysis::getAnalysisUsage(AU);
      AU.setPreservesAll();                         // Does not transform code
      AU.addRequiredTransitive<TDDataStructures>(); // Uses TD Datastructures
      AU.addRequiredTransitive<BUDataStructures>(); // Uses BU Datastructures
      AU.addRequired<AliasAnalysis>();              // Chains to another AA impl
    }

    //------------------------------------------------
    // Implement the AliasAnalysis API
    //  

    AliasResult alias(const Value *V1, unsigned V1Size,
                      const Value *V2, unsigned V2Size);

    void getMustAliases(Value *P, std::vector<Value*> &RetVals);

    bool pointsToConstantMemory(const Value *P) {
      return getAnalysis<AliasAnalysis>().pointsToConstantMemory(P);
    }
    
    AliasAnalysis::ModRefResult
    getModRefInfo(CallSite CS, Value *P, unsigned Size);

  private:
    DSGraph *getGraphForValue(const Value *V);
  };

  // Register the pass...
  RegisterOpt<DSAA> X("ds-aa", "Data Structure Graph Based Alias Analysis");

  // Register as an implementation of AliasAnalysis
  RegisterAnalysisGroup<AliasAnalysis, DSAA> Y;
}

// getGraphForValue - Return the DSGraph to use for queries about the specified
// value...
//
DSGraph *DSAA::getGraphForValue(const Value *V) {
  if (const Instruction *I = dyn_cast<Instruction>(V))
    return &TD->getDSGraph(*I->getParent()->getParent());
  else if (const Argument *A = dyn_cast<Argument>(V))
    return &TD->getDSGraph(*A->getParent());
  else if (const BasicBlock *BB = dyn_cast<BasicBlock>(V))
    return &TD->getDSGraph(*BB->getParent());
  return 0;
}

// isSinglePhysicalObject - For now, the only case that we know that there is
// only one memory object in the node is when there is a single global in the
// node, and the only composition bit set is Global.
//
static bool isSinglePhysicalObject(DSNode *N) {
  assert(N->isComplete() && "Can only tell if this is a complete object!");
  return N->isGlobalNode() && N->getGlobals().size() == 1 &&
         !N->isHeapNode() && !N->isAllocaNode() && !N->isUnknownNode();
}

// alias - This is the only method here that does anything interesting...
AliasAnalysis::AliasResult DSAA::alias(const Value *V1, unsigned V1Size,
                                       const Value *V2, unsigned V2Size) {
  if (V1 == V2) return MustAlias;

  DSGraph *G1 = getGraphForValue(V1);
  DSGraph *G2 = getGraphForValue(V2);
  assert((!G1 || !G2 || G1 == G2) && "Alias query for 2 different functions?");
  
  // Get the graph to use...
  DSGraph &G = *(G1 ? G1 : (G2 ? G2 : &TD->getGlobalsGraph()));

  const DSGraph::ScalarMapTy &GSM = G.getScalarMap();
  DSGraph::ScalarMapTy::const_iterator I = GSM.find((Value*)V1);
  if (I == GSM.end()) return NoAlias;

  assert(I->second.getNode() && "Scalar map points to null node?");
  DSGraph::ScalarMapTy::const_iterator J = GSM.find((Value*)V2);
  if (J == GSM.end()) return NoAlias;

  assert(J->second.getNode() && "Scalar map points to null node?");

  DSNode  *N1 = I->second.getNode(),  *N2 = J->second.getNode();
  unsigned O1 = I->second.getOffset(), O2 = J->second.getOffset();
        
  // We can only make a judgment of one of the nodes is complete...
  if (N1->isComplete() || N2->isComplete()) {
    if (N1 != N2)
      return NoAlias;   // Completely different nodes.

#if 0  // This does not correctly handle arrays!
    // Both point to the same node and same offset, and there is only one
    // physical memory object represented in the node, return must alias.
    //
    // FIXME: This isn't correct because we do not handle array indexing
    // correctly.

    if (O1 == O2 && isSinglePhysicalObject(N1))
      return MustAlias; // Exactly the same object & offset
#endif

    // See if they point to different offsets...  if so, we may be able to
    // determine that they do not alias...
    if (O1 != O2) {
      if (O2 < O1) {    // Ensure that O1 <= O2
        std::swap(V1, V2);
        std::swap(O1, O2);
        std::swap(V1Size, V2Size);
      }

      // FIXME: This is not correct because we do not handle array
      // indexing correctly with this check!
      //if (O1+V1Size <= O2) return NoAlias;
    }
  }

  // FIXME: we could improve on this by checking the globals graph for aliased
  // global queries...
  return getAnalysis<AliasAnalysis>().alias(V1, V1Size, V2, V2Size);
}

/// getModRefInfo - does a callsite modify or reference a value?
///
AliasAnalysis::ModRefResult
DSAA::getModRefInfo(CallSite CS, Value *P, unsigned Size) {
  Function *F = CS.getCalledFunction();
  if (!F) return pointsToConstantMemory(P) ? Ref : ModRef;
  if (F->isExternal()) return ModRef;

  // Clone the function TD graph, clearing off Mod/Ref flags
  const Function *csParent = CS.getInstruction()->getParent()->getParent();
  DSGraph TDGraph(TD->getDSGraph(*csParent));
  TDGraph.maskNodeTypes(0);
  
  // Insert the callee's BU graph into the TD graph
  const DSGraph &BUGraph = BU->getDSGraph(*F);
  TDGraph.mergeInGraph(TDGraph.getDSCallSiteForCallSite(CS),
                       *F, BUGraph, 0);

  // Report the flags that have been added
  const DSNodeHandle &DSH = TDGraph.getNodeForValue(P);
  if (const DSNode *N = DSH.getNode())
    if (N->isModified())
      return N->isRead() ? ModRef : Mod;
    else
      return N->isRead() ? Ref : NoModRef;
  return NoModRef;
}


/// getMustAliases - If there are any pointers known that must alias this
/// pointer, return them now.  This allows alias-set based alias analyses to
/// perform a form a value numbering (which is exposed by load-vn).  If an alias
/// analysis supports this, it should ADD any must aliased pointers to the
/// specified vector.
///
void DSAA::getMustAliases(Value *P, std::vector<Value*> &RetVals) {
#if 0    // This does not correctly handle arrays!
  // Currently the only must alias information we can provide is to say that
  // something is equal to a global value. If we already have a global value,
  // don't get worked up about it.
  if (!isa<GlobalValue>(P)) {
    DSGraph *G = getGraphForValue(P);
    if (!G) G = &TD->getGlobalsGraph();
    
    // The only must alias information we can currently determine occurs when
    // the node for P is a global node with only one entry.
    DSGraph::ScalarMapTy::const_iterator I = G->getScalarMap().find(P);
    if (I != G->getScalarMap().end()) {
      DSNode *N = I->second.getNode();
      if (N->isComplete() && isSinglePhysicalObject(N))
        RetVals.push_back(N->getGlobals()[0]);
    }
  }
#endif
  return getAnalysis<AliasAnalysis>().getMustAliases(P, RetVals);
}

