//===- DataStructureAA.cpp - Data Structure Based Alias Analysis ----------===//
//
// This pass uses the top-down data structure graphs to implement a simple
// context sensitive alias analysis.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DataStructure.h"
#include "llvm/Analysis/DSGraph.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Module.h"

namespace {
  class DSAA : public Pass, public AliasAnalysis {
    TDDataStructures *TD;
  public:
    DSAA() : TD(0) {}

    //------------------------------------------------
    // Implement the Pass API
    //

    // run - Build up the result graph, representing the pointer graph for the
    // program.
    //
    bool run(Module &M) {
      TD = &getAnalysis<TDDataStructures>();
      return false;
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();                    // Does not transform code...
      AU.addRequired<TDDataStructures>();      // Uses TD Datastructures
      AU.addRequired<AliasAnalysis>();         // Chains to another AA impl...
    }

    //------------------------------------------------
    // Implement the AliasAnalysis API
    //  

    // alias - This is the only method here that does anything interesting...
    Result alias(const Value *V1, const Value *V2);
    
    /// canCallModify - Not implemented yet: FIXME
    ///
    Result canCallModify(const CallInst &CI, const Value *Ptr) {
      return MayAlias;
    }
    
    /// canInvokeModify - Not implemented yet: FIXME
    ///
    Result canInvokeModify(const InvokeInst &I, const Value *Ptr) {
      return MayAlias;
    }
  };

  // Register the pass...
  RegisterOpt<DSAA> X("ds-aa", "Data Structure Graph Based Alias Analysis");

  // Register as an implementation of AliasAnalysis
  RegisterAnalysisGroup<AliasAnalysis, DSAA> Y;
}

// getValueFunction - return the function containing the specified value if
// available, or null otherwise.
//
static const Function *getValueFunction(const Value *V) {
  if (const Instruction *I = dyn_cast<Instruction>(V))
    return I->getParent()->getParent();
  else if (const Argument *A = dyn_cast<Argument>(V))
    return A->getParent();
  else if (const BasicBlock *BB = dyn_cast<BasicBlock>(V))
    return BB->getParent();
  return 0;
}

// alias - This is the only method here that does anything interesting...
AliasAnalysis::Result DSAA::alias(const Value *V1, const Value *V2) {
  const Function *F1 = getValueFunction(V1);
  const Function *F2 = getValueFunction(V2);
  assert((!F1 || !F2 || F1 == F2) && "Alias query for 2 different functions?");
  

  // FIXME: This can return must alias if querying a DSNode for a global value
  // where the node has only the G composition bit set, and only one entry in
  // the globals list...
  if (F2) F1 = F2;
  if (F1) {
    // Get the graph for a function...
    DSGraph &G = TD->getDSGraph(*F1);
    hash_map<Value*, DSNodeHandle> &GSM = G.getScalarMap();
    hash_map<Value*, DSNodeHandle>::iterator I = GSM.find((Value*)V1);
    if (I != GSM.end()) {
      assert(I->second.getNode() && "Scalar map points to null node?");
      hash_map<Value*, DSNodeHandle>::iterator J = GSM.find((Value*)V2);
      if (J != GSM.end()) {
        assert(J->second.getNode() && "Scalar map points to null node?");
        if (I->second.getNode() != J->second.getNode()) {
          // Return noalias if one of the nodes is complete...
          if ((~I->second.getNode()->NodeType | ~J->second.getNode()->NodeType)
              & DSNode::Incomplete)
            return NoAlias;
          // both are incomplete, they may alias...
        } else {
          // Both point to the same node, see if they point to different
          // offsets...  FIXME: This needs to know the size of the alias query
          if (I->second.getOffset() != J->second.getOffset())
            return NoAlias;
        }
      }
    }
  }

  // FIXME: we could improve on this by checking the globals graph for aliased
  // global queries...
  return getAnalysis<AliasAnalysis>().alias(V1, V2);
}
