//===-- PoolAllocate.h - Pool allocation pass -------------------*- C++ -*-===//
//
// This transform changes programs so that disjoint data structures are
// allocated out of different pools of memory, increasing locality.  This header
// file exposes information about the pool allocation itself so that follow-on
// passes may extend or use the pool allocation for analysis.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_POOLALLOCATE_H
#define LLVM_TRANSFORMS_POOLALLOCATE_H

#include "llvm/Pass.h"
#include "Support/hash_set"
class BUDataStructures;
class DSNode;
class DSGraph;

namespace PA {
  /// FuncInfo - Represent the pool allocation information for one function in
  /// the program.  Note that many functions must actually be cloned in order
  /// for pool allocation to add arguments to the function signature.  In this
  /// case, the Clone and NewToOldValueMap information identify how the clone
  /// maps to the original function...
  ///
  struct FuncInfo {
    /// MarkedNodes - The set of nodes which are not locally pool allocatable in
    /// the current function.
    ///
    hash_set<DSNode*> MarkedNodes;

    /// Clone - The cloned version of the function, if applicable.
    Function *Clone;

    /// ArgNodes - The list of DSNodes which have pools passed in as arguments.
    ///
    std::vector<DSNode*> ArgNodes;

    /// PoolDescriptors - The Value* (either an argument or an alloca) which
    /// defines the pool descriptor for this DSNode.  Pools are mapped one to
    /// one with nodes in the DSGraph, so this contains a pointer to the node it
    /// corresponds to.  In addition, the pool is initialized by calling the
    /// "poolinit" library function with a chunk of memory allocated with an
    /// alloca instruction.  This entry contains a pointer to that alloca if the
    /// pool is locally allocated or the argument it is passed in through if
    /// not.
    ///
    std::map<DSNode*, Value*> PoolDescriptors;

    /// NewToOldValueMap - When and if a function needs to be cloned, this map
    /// contains a mapping from all of the values in the new function back to
    /// the values they correspond to in the old function.
    ///
    std::map<Value*, const Value*> NewToOldValueMap;
  };
}

/// PoolAllocate - The main pool allocation pass
///
class PoolAllocate : public Pass {
  Module *CurModule;
  BUDataStructures *BU;
  
  std::map<Function*, PA::FuncInfo> FunctionInfo;
 public:
  Function *PoolInit, *PoolDestroy, *PoolAlloc, *PoolFree;
 public:
  bool run(Module &M);
  
  virtual void getAnalysisUsage(AnalysisUsage &AU) const;
  
  BUDataStructures &getBUDataStructures() const { return *BU; }
  
  PA::FuncInfo *getFuncInfo(Function &F) {
    std::map<Function*, PA::FuncInfo>::iterator I = FunctionInfo.find(&F);
    return I != FunctionInfo.end() ? &I->second : 0;
  }
  
 private:
  
  /// AddPoolPrototypes - Add prototypes for the pool functions to the
  /// specified module and update the Pool* instance variables to point to
  /// them.
  ///
  void AddPoolPrototypes();
  
  /// MakeFunctionClone - If the specified function needs to be modified for
  /// pool allocation support, make a clone of it, adding additional arguments
  /// as neccesary, and return it.  If not, just return null.
  ///
  Function *MakeFunctionClone(Function &F);
  
  /// ProcessFunctionBody - Rewrite the body of a transformed function to use
  /// pool allocation where appropriate.
  ///
  void ProcessFunctionBody(Function &Old, Function &New);
  
  /// CreatePools - This creates the pool initialization and destruction code
  /// for the DSNodes specified by the NodesToPA list.  This adds an entry to
  /// the PoolDescriptors map for each DSNode.
  ///
  void CreatePools(Function &F, const std::vector<DSNode*> &NodesToPA,
                   std::map<DSNode*, Value*> &PoolDescriptors);
  
  void TransformFunctionBody(Function &F, DSGraph &G, PA::FuncInfo &FI);
};

#endif
