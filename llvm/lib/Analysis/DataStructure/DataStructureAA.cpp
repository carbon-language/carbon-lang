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
      InitializeAliasAnalysis(this);
      TD = &getAnalysis<TDDataStructures>();
      return false;
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AliasAnalysis::getAnalysisUsage(AU);
      AU.setPreservesAll();                    // Does not transform code...
      AU.addRequired<TDDataStructures>();      // Uses TD Datastructures
      AU.addRequired<AliasAnalysis>();         // Chains to another AA impl...
    }

    //------------------------------------------------
    // Implement the AliasAnalysis API
    //  

    // alias - This is the only method here that does anything interesting...
    AliasResult alias(const Value *V1, unsigned V1Size,
                      const Value *V2, unsigned V2Size);
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
AliasAnalysis::AliasResult DSAA::alias(const Value *V1, unsigned V1Size,
                                       const Value *V2, unsigned V2Size) {
  if (V1 == V2) return MustAlias;

  const Function *F1 = getValueFunction(V1);
  const Function *F2 = getValueFunction(V2);
  assert((!F1 || !F2 || F1 == F2) && "Alias query for 2 different functions?");
  
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

        DSNode  *N1 = I->second.getNode(),  *N2 = J->second.getNode();
        unsigned O1 = I->second.getOffset(), O2 = J->second.getOffset();
        
        // We can only make a judgement of one of the nodes is complete...
        if (N1->isComplete() || N2->isComplete()) {
          if (N1 != N2)
            return NoAlias;   // Completely different nodes.

          // Both point to the same node and same offset, and there is only one
          // physical memory object represented in the node, return must alias.
          //if (O1 == O2 && !N1->isMultiObject())
          //  return MustAlias; // Exactly the same object & offset

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
      }
    }
  }

  // FIXME: we could improve on this by checking the globals graph for aliased
  // global queries...
  return getAnalysis<AliasAnalysis>().alias(V1, V1Size, V2, V2Size);
}
