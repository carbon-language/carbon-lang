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
#include "Support/EquivalenceClasses.h"
class BUDataStructures;
class TDDataStructures;
class DSNode;
class DSGraph;
class CallInst;

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

    /// In order to handle indirect functions, the start and end of the 
    /// arguments that are useful to this function. 
    /// The pool arguments useful to this function are PoolArgFirst to 
    /// PoolArgLast not inclusive.
    int PoolArgFirst, PoolArgLast;
    
    /// PoolDescriptors - The Value* (either an argument or an alloca) which
    /// defines the pool descriptor for this DSNode.  Pools are mapped one to
    /// one with nodes in the DSGraph, so this contains a pointer to the node it
    /// corresponds to.  In addition, the pool is initialized by calling the
    /// "poolinit" library function with a chunk of memory allocated with an
    /// alloca instruction.  This entry contains a pointer to that alloca if the
    /// pool is locally allocated or the argument it is passed in through if
    /// not.
    /// Note: Does not include pool arguments that are passed in because of
    /// indirect function calls that are not used in the function.
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

  TDDataStructures *TDDS;
  
  std::map<Function*, PA::FuncInfo> FunctionInfo;

  void buildIndirectFunctionSets(Module &M);   

  void FindFunctionPoolArgs(Function &F);   
  
  // Debug function to print the FuncECs
  void printFuncECs();
  
 public:
  Function *PoolInit, *PoolDestroy, *PoolAlloc, *PoolAllocArray, *PoolFree;

  // Equivalence class where functions that can potentially be called via
  // the same function pointer are in the same class.
  EquivalenceClasses<Function *> FuncECs;

  // Map from an Indirect CallInst to the set of Functions that it can point to
  std::multimap<CallInst *, Function *> CallInstTargets;

  // This maps an equivalence class to the last pool argument number for that 
  // class. This is used because the pool arguments for all functions within
  // an equivalence class is passed to all the functions in that class.
  // If an equivalence class does not require pool arguments, it is not
  // on this map.
  std::map<Function *, int> EqClass2LastPoolArg;

 public:
  bool run(Module &M);
  
  virtual void getAnalysisUsage(AnalysisUsage &AU) const;
  
  BUDataStructures &getBUDataStructures() const { return *BU; }
  
  PA::FuncInfo *getFuncInfo(Function &F) {
    std::map<Function*, PA::FuncInfo>::iterator I = FunctionInfo.find(&F);
    return I != FunctionInfo.end() ? &I->second : 0;
  }

  Module *getCurModule() { return CurModule; }

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
