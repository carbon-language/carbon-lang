//===-- PoolAllocate.cpp - Pool Allocation Pass ---------------------------===//
//
// This transform changes programs so that disjoint data structures are
// allocated out of different pools of memory, increasing locality and shrinking
// pointer size.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/PoolAllocate.h"
#include "llvm/Analysis/DataStructure.h"
#include "llvm/Pass.h"
#include "llvm/Module.h"
#include "llvm/Function.h"
#include "llvm/iMemory.h"
#include <algorithm>

// Define the pass class that we implement...
namespace {
  class PoolAllocate : public Pass {
    // PoolTy - The type of a scalar value that contains a pool pointer.
    PointerType *PoolTy;
  public:

    PoolAllocate() {
      // Initialize the PoolTy instance variable, since the type never changes.
      vector<const Type*> PoolElements;
      PoolElements.push_back(PointerType::get(Type::SByteTy));
      PoolElements.push_back(Type::UIntTy);
      PoolTy = PointerType::get(StructType::get(PoolElements));
      // PoolTy = { sbyte*, uint }*

      CurModule = 0; DS = 0;
      PoolInit = PoolDestroy = PoolAlloc = PoolFree = 0;
    }

    bool run(Module *M);

    // getAnalysisUsageInfo - This function requires data structure information
    // to be able to see what is pool allocatable.
    //
    virtual void getAnalysisUsageInfo(Pass::AnalysisSet &Required,
                                      Pass::AnalysisSet &,Pass::AnalysisSet &) {
      Required.push_back(DataStructure::ID);
    }

  private:
    // CurModule - The module being processed.
    Module *CurModule;

    // DS - The data structure graph for the module being processed.
    DataStructure *DS;

    // Prototypes that we add to support pool allocation...
    Function *PoolInit, *PoolDestroy, *PoolAlloc, *PoolFree;

    // addPoolPrototypes - Add prototypes for the pool methods to the specified
    // module and update the Pool* instance variables to point to them.
    //
    void addPoolPrototypes(Module *M);

    // processFunction - Convert a function to use pool allocation where
    // available.
    //
    bool processFunction(Function *F);
  };
}



// isNotPoolableAlloc - This is a predicate that returns true if the specified 
// allocation node in a data structure graph is eligable for pool allocation.
//
static bool isNotPoolableAlloc(const AllocDSNode *DS) {
  if (DS->isAllocaNode()) return false;  // Do not pool allocate alloca's.

  MallocInst *MI = cast<MallocInst>(DS->getAllocation());
  if (MI->isArrayAllocation() && !isa<Constant>(MI->getArraySize()))
    return false;   // Do not allow variable size allocations...

  return true;
}


// processFunction - Convert a function to use pool allocation where
// available.
//
bool PoolAllocate::processFunction(Function *F) {
  // Get the closed datastructure graph for the current function... if there are
  // any allocations in this graph that are not escaping, we need to pool
  // allocate them here!
  //
  FunctionDSGraph &IPGraph = DS->getClosedDSGraph(F);

  // Get all of the allocations that do not escape the current function.  Since
  // they are still live (they exist in the graph at all), this means we must
  // have scalar references to these nodes, but the scalars are never returned.
  // 
  std::vector<AllocDSNode*> Allocs;
  IPGraph.getNonEscapingAllocations(Allocs);

  // Filter out allocations that we cannot handle.  Currently, this includes
  // variable sized array allocations and alloca's (which we do not want to
  // pool allocate)
  //
  Allocs.erase(remove_if(Allocs.begin(), Allocs.end(), isNotPoolableAlloc),
               Allocs.end());


  if (Allocs.empty()) return false;  // Nothing to do.

  // Loop through the value map looking for scalars that refer to nonescaping
  // allocations.
  //
  map<Value*, PointerValSet> &ValMap = IPGraph.getValueMap();
  vector<pair<Value*, AllocDSNode*> > Scalars;

  for (map<Value*, PointerValSet>::iterator I = ValMap.begin(),
         E = ValMap.end(); I != E; ++I) {
    const PointerValSet &PVS = I->second;  // Set of things pointed to by scalar
    // Check to see if the scalar points to anything that is an allocation...
    for (unsigned i = 0, e = PVS.size(); i != e; ++i)
      if (AllocDSNode *Alloc = dyn_cast<AllocDSNode>(PVS[i].Node)) {
        assert(PVS[i].Index == 0 && "Nonzero not handled yet!");
        
        // If the allocation is in the nonescaping set...
        if (find(Allocs.begin(), Allocs.end(), Alloc) != Allocs.end())
          // Add it to the list of scalars we have
          Scalars.push_back(make_pair(I->first, Alloc));
      }
  }

  cerr << "In '" << F->getName()
       << "': Found the following values that point to poolable nodes:\n";

  for (unsigned i = 0, e = Scalars.size(); i != e; ++i)
    Scalars[i].first->dump();
  return false;
}


// Prototypes that we add to support pool allocation...
Function *PoolInit, *PoolDestroy, *PoolAlloc, *PoolFree;

// addPoolPrototypes - Add prototypes for the pool methods to the specified
// module and update the Pool* instance variables to point to them.
//
void PoolAllocate::addPoolPrototypes(Module *M) {
  //M->getOrCreate...

}


bool PoolAllocate::run(Module *M) {
  addPoolPrototypes(M);
  CurModule = M;
  
  DS = &getAnalysis<DataStructure>();
  bool Changed = false;
  for (Module::iterator I = M->begin(); I != M->end(); ++I)
    if (!(*I)->isExternal())
      Changed |= processFunction(*I);

  CurModule = 0;
  DS = 0;
  return false;
}


// createPoolAllocatePass - Global function to access the functionality of this
// pass...
//
Pass *createPoolAllocatePass() { return new PoolAllocate(); }
