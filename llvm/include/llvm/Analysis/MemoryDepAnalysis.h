//===- MemoryDepAnalysis.h - Compute dep graph for memory ops ---*- C++ -*-===//
//
// This file provides a pass (MemoryDepAnalysis) that computes memory-based
// data dependences between instructions for each function in a module.  
// Memory-based dependences occur due to load and store operations, but
// also the side-effects of call instructions.
//
// The result of this pass is a DependenceGraph for each function
// representing the memory-based data dependences between instructions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_MEMORYDEPANALYSIS_H
#define LLVM_ANALYSIS_MEMORYDEPANALYSIS_H

#include "llvm/Analysis/DependenceGraph.h"
#include "llvm/Analysis/IPModRef.h"
#include "llvm/Analysis/DataStructure.h"
#include "llvm/Pass.h"
#include "Support/TarjanSCCIterator.h"
#include "Support/hash_map"

class Instruction;
class ModRefTable;

///---------------------------------------------------------------------------
/// class MemoryDepGraph:
///   Dependence analysis for load/store/call instructions using IPModRef info
///   computed at the granularity of individual DSGraph nodes.
///
/// This pass computes memory dependences for each function in a module.
/// It can be made a FunctionPass once a Pass (such as Parallelize) is
/// allowed to use a FunctionPass such as this one.
///---------------------------------------------------------------------------

class MemoryDepAnalysis: /* Use if FunctionPass: public DependenceGraph, */
                         public Pass {
  /// The following map and depGraph pointer are temporary until this class
  /// becomes a FunctionPass instead of a module Pass. */
  hash_map<Function*, DependenceGraph*> funcMap;
  DependenceGraph* funcDepGraph;

  /// Information about one function being analyzed.
  const DSGraph*  funcGraph;
  const FunctionModRefInfo* funcModRef;

  /// Internal routine that processes each SCC of the CFG.
  void MemoryDepAnalysis::ProcessSCC(SCC<Function*>& S,
                                     ModRefTable& ModRefAfter);

  friend class PgmDependenceGraph;

public:
  MemoryDepAnalysis()
    : /*DependenceGraph(),*/ funcDepGraph(NULL),
      funcGraph(NULL), funcModRef(NULL) { }
  ~MemoryDepAnalysis();

  ///------------------------------------------------------------------------
  /// TEMPORARY FUNCTIONS TO MAKE THIS A MODULE PASS ---
  /// These functions will go away once this class becomes a FunctionPass.
  
  /// Driver function to compute dependence graphs for every function.
  bool run(Module& M);

  /// getGraph() -- Retrieve the dependence graph for a function.
  /// This is temporary and will go away once this is a FunctionPass.
  /// At that point, this class should directly inherit from DependenceGraph.
  /// 
  DependenceGraph& getGraph(Function& F) {
    hash_map<Function*, DependenceGraph*>::iterator I = funcMap.find(&F);
    assert(I != funcMap.end());
    return *I->second;
  }
  const DependenceGraph& getGraph(Function& F) const {
    hash_map<Function*, DependenceGraph*>::const_iterator
      I = funcMap.find(&F);
    assert(I != funcMap.end());
    return *I->second;
  }

  /// Release depGraphs held in the Function -> DepGraph map.
  /// 
  virtual void releaseMemory();

  ///----END TEMPORARY FUNCTIONS---------------------------------------------


  /// Driver functions to compute the Load/Store Dep. Graph per function.
  /// 
  bool runOnFunction(Function& _func);

  /// getAnalysisUsage - This does not modify anything.
  /// It uses the Top-Down DS Graph and IPModRef.
  ///
  void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addRequired<TDDataStructures>();
    AU.addRequired<IPModRef>();
  }

  /// Debugging support methods
  /// 
  void print(std::ostream &O) const;
  void dump() const;
};


//===----------------------------------------------------------------------===//

#endif
