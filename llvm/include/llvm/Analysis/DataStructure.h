//===- DataStructure.h - Build data structure graphs ------------*- C++ -*-===//
//
// Implement the LLVM data structure analysis library.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_DATA_STRUCTURE_H
#define LLVM_ANALYSIS_DATA_STRUCTURE_H

#include <assert.h>

#include "llvm/Pass.h"
#include "Support/HashExtras.h"
#include "Support/hash_set"

class Type;
class DSGraph;
class DSNode;
class DSCallSite;

// FIXME: move this stuff to a private header
namespace DataStructureAnalysis {
  // isPointerType - Return true if this first class type is big enough to hold
  // a pointer.
  //
  bool isPointerType(const Type *Ty);
}


// LocalDataStructures - The analysis that computes the local data structure
// graphs for all of the functions in the program.
//
// FIXME: This should be a Function pass that can be USED by a Pass, and would
// be automatically preserved.  Until we can do that, this is a Pass.
//
class LocalDataStructures : public Pass {
  // DSInfo, one graph for each function
  hash_map<const Function*, DSGraph*> DSInfo;
  DSGraph *GlobalsGraph;
public:
  ~LocalDataStructures() { releaseMemory(); }

  virtual bool run(Module &M);

  bool hasGraph(const Function &F) const {
    return DSInfo.find(&F) != DSInfo.end();
  }

  // getDSGraph - Return the data structure graph for the specified function.
  DSGraph &getDSGraph(const Function &F) const {
    hash_map<const Function*, DSGraph*>::const_iterator I = DSInfo.find(&F);
    assert(I != DSInfo.end() && "Function not in module!");
    return *I->second;
  }

  DSGraph &getGlobalsGraph() const { return *GlobalsGraph; }

  // print - Print out the analysis results...
  void print(std::ostream &O, const Module *M) const;

  // If the pass pipeline is done with this pass, we can release our memory...
  virtual void releaseMemory();

  // getAnalysisUsage - This obviously provides a data structure graph.
  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
  }
};


// BUDataStructures - The analysis that computes the interprocedurally closed
// data structure graphs for all of the functions in the program.  This pass
// only performs a "Bottom Up" propagation (hence the name).
//
class BUDataStructures : public Pass {
  // DSInfo, one graph for each function
  hash_map<const Function*, DSGraph*> DSInfo;
  DSGraph *GlobalsGraph;
public:
  ~BUDataStructures() { releaseMemory(); }

  virtual bool run(Module &M);

  bool hasGraph(const Function &F) const {
    return DSInfo.find(&F) != DSInfo.end();
  }

  // getDSGraph - Return the data structure graph for the specified function.
  DSGraph &getDSGraph(const Function &F) const {
    hash_map<const Function*, DSGraph*>::const_iterator I = DSInfo.find(&F);
    assert(I != DSInfo.end() && "Function not in module!");
    return *I->second;
  }

  DSGraph &getGlobalsGraph() const { return *GlobalsGraph; }

  // print - Print out the analysis results...
  void print(std::ostream &O, const Module *M) const;

  // If the pass pipeline is done with this pass, we can release our memory...
  virtual void releaseMemory();

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addRequired<LocalDataStructures>();
  }
private:
  DSGraph &calculateGraph(Function &F);

  // inlineNonSCCGraphs - This method is almost like the other two calculate
  // graph methods.  This one is used to inline function graphs (from functions
  // outside of the SCC) into functions in the SCC.  It is not supposed to touch
  // functions IN the SCC at all.
  //
  DSGraph &inlineNonSCCGraphs(Function &F,
                              hash_set<Function*> &SCCFunctions);
 
  DSGraph &calculateSCCGraph(Function &F,
                             hash_set<Function*> &InlinedSCCFunctions);
  void calculateReachableGraphs(Function *F);


  DSGraph &getOrCreateGraph(Function *F);

  unsigned calculateGraphs(Function *F, std::vector<Function*> &Stack,
                           unsigned &NextID, 
                           hash_map<Function*, unsigned> &ValMap);
};


// TDDataStructures - Analysis that computes new data structure graphs
// for each function using the closed graphs for the callers computed
// by the bottom-up pass.
//
class TDDataStructures : public Pass {
  // DSInfo, one graph for each function
  hash_map<const Function*, DSGraph*> DSInfo;
  hash_set<const Function*> GraphDone;
  DSGraph *GlobalsGraph;
public:
  ~TDDataStructures() { releaseMyMemory(); }

  virtual bool run(Module &M);

  bool hasGraph(const Function &F) const {
    return DSInfo.find(&F) != DSInfo.end();
  }

  // getDSGraph - Return the data structure graph for the specified function.
  DSGraph &getDSGraph(const Function &F) const {
    hash_map<const Function*, DSGraph*>::const_iterator I = DSInfo.find(&F);
    assert(I != DSInfo.end() && "Function not in module!");
    return *I->second;
  }

  DSGraph &getGlobalsGraph() const { return *GlobalsGraph; }

  // print - Print out the analysis results...
  void print(std::ostream &O, const Module *M) const;

  // If the pass pipeline is done with this pass, we can release our memory...
  virtual void releaseMyMemory();

  // getAnalysisUsage - This obviously provides a data structure graph.
  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addRequired<BUDataStructures>();
  }

private:
  void calculateGraph(Function &F);
  DSGraph &getOrCreateDSGraph(Function &F);

  void ResolveCallSite(DSGraph &Graph, const DSCallSite &CallSite);
};

#endif
