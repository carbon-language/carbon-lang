//===- DataStructure.h - Build data structure graphs ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implement the LLVM data structure analysis library.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_DATA_STRUCTURE_H
#define LLVM_ANALYSIS_DATA_STRUCTURE_H

#include "llvm/Pass.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Support/CallSite.h"
#include "llvm/ADT/hash_map"
#include "llvm/ADT/hash_set"
#include "llvm/ADT/EquivalenceClasses.h"

namespace llvm {

class Type;
class Instruction;
class GlobalValue;
class DSGraph;
class DSCallSite;
class DSNode;
class DSNodeHandle;

FunctionPass *createDataStructureStatsPass();
FunctionPass *createDataStructureGraphCheckerPass();


// FIXME: move this stuff to a private header
namespace DataStructureAnalysis {
  /// isPointerType - Return true if this first class type is big enough to hold
  /// a pointer.
  ///
  bool isPointerType(const Type *Ty);
}


// LocalDataStructures - The analysis that computes the local data structure
// graphs for all of the functions in the program.
//
// FIXME: This should be a Function pass that can be USED by a Pass, and would
// be automatically preserved.  Until we can do that, this is a Pass.
//
class LocalDataStructures : public ModulePass {
  // DSInfo, one graph for each function
  hash_map<Function*, DSGraph*> DSInfo;
  DSGraph *GlobalsGraph;

  /// GlobalECs - The equivalence classes for each global value that is merged
  /// with other global values in the DSGraphs.
  EquivalenceClasses<GlobalValue*> GlobalECs;
public:
  ~LocalDataStructures() { releaseMemory(); }

  virtual bool runOnModule(Module &M);

  bool hasGraph(const Function &F) const {
    return DSInfo.find(const_cast<Function*>(&F)) != DSInfo.end();
  }

  /// getDSGraph - Return the data structure graph for the specified function.
  ///
  DSGraph &getDSGraph(const Function &F) const {
    hash_map<Function*, DSGraph*>::const_iterator I =
      DSInfo.find(const_cast<Function*>(&F));
    assert(I != DSInfo.end() && "Function not in module!");
    return *I->second;
  }

  DSGraph &getGlobalsGraph() const { return *GlobalsGraph; }

  EquivalenceClasses<GlobalValue*> &getGlobalECs() { return GlobalECs; }

  /// print - Print out the analysis results...
  ///
  void print(std::ostream &O, const Module *M) const;

  /// releaseMemory - if the pass pipeline is done with this pass, we can
  /// release our memory...
  ///
  virtual void releaseMemory();

  /// getAnalysisUsage - This obviously provides a data structure graph.
  ///
  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addRequired<TargetData>();
  }
};


/// BUDataStructures - The analysis that computes the interprocedurally closed
/// data structure graphs for all of the functions in the program.  This pass
/// only performs a "Bottom Up" propagation (hence the name).
///
class BUDataStructures : public ModulePass {
protected:
  // DSInfo, one graph for each function
  hash_map<Function*, DSGraph*> DSInfo;
  DSGraph *GlobalsGraph;
  std::set<std::pair<Instruction*, Function*> > ActualCallees;

  // This map is only maintained during construction of BU Graphs
  std::map<std::vector<Function*>,
           std::pair<DSGraph*, std::vector<DSNodeHandle> > > *IndCallGraphMap;

  /// GlobalECs - The equivalence classes for each global value that is merged
  /// with other global values in the DSGraphs.
  EquivalenceClasses<GlobalValue*> GlobalECs;

  std::map<CallSite, std::vector<Function*> > AlreadyInlined;

public:
  ~BUDataStructures() { releaseMyMemory(); }

  virtual bool runOnModule(Module &M);

  bool hasGraph(const Function &F) const {
    return DSInfo.find(const_cast<Function*>(&F)) != DSInfo.end();
  }

  /// getDSGraph - Return the data structure graph for the specified function.
  ///
  DSGraph &getDSGraph(const Function &F) const {
    hash_map<Function*, DSGraph*>::const_iterator I =
      DSInfo.find(const_cast<Function*>(&F));
    if (I != DSInfo.end())
      return *I->second;
    return const_cast<BUDataStructures*>(this)->
                   CreateGraphForExternalFunction(F);
  }
  
  /// DSGraphExists - Is the DSGraph computed for this function?
  ///
  bool doneDSGraph(const Function *F) const {
    return (DSInfo.find(const_cast<Function*>(F)) != DSInfo.end());
  }

  DSGraph &getGlobalsGraph() const { return *GlobalsGraph; }

  EquivalenceClasses<GlobalValue*> &getGlobalECs() { return GlobalECs; }

  DSGraph &CreateGraphForExternalFunction(const Function &F);

  /// deleteValue/copyValue - Interfaces to update the DSGraphs in the program.
  /// These correspond to the interfaces defined in the AliasAnalysis class.
  void deleteValue(Value *V);
  void copyValue(Value *From, Value *To);

  /// print - Print out the analysis results...
  ///
  void print(std::ostream &O, const Module *M) const;

  // FIXME: Once the pass manager is straightened out, rename this to
  // releaseMemory.
  void releaseMyMemory();

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addRequired<LocalDataStructures>();
  }

  typedef std::set<std::pair<Instruction*, Function*> > ActualCalleesTy;
  const ActualCalleesTy &getActualCallees() const {
    return ActualCallees;
  }

  typedef ActualCalleesTy::const_iterator callee_iterator;
  callee_iterator callee_begin(Instruction *I) const {
    return ActualCallees.lower_bound(std::pair<Instruction*,Function*>(I, 0));
  }

  callee_iterator callee_end(Instruction *I) const {
    I = (Instruction*)((char*)I + 1);
    return ActualCallees.lower_bound(std::pair<Instruction*,Function*>(I, 0));
  }

private:
  bool calculateGraph(DSGraph &G);

  DSGraph &getOrCreateGraph(Function *F);

  unsigned calculateGraphs(Function *F, std::vector<Function*> &Stack,
                           unsigned &NextID,
                           hash_map<Function*, unsigned> &ValMap);
};


/// TDDataStructures - Analysis that computes new data structure graphs
/// for each function using the closed graphs for the callers computed
/// by the bottom-up pass.
///
class TDDataStructures : public ModulePass {
  // DSInfo, one graph for each function
  hash_map<Function*, DSGraph*> DSInfo;
  hash_set<Function*> ArgsRemainIncomplete;
  DSGraph *GlobalsGraph;
  BUDataStructures *BUInfo;

  /// GlobalECs - The equivalence classes for each global value that is merged
  /// with other global values in the DSGraphs.
  EquivalenceClasses<GlobalValue*> GlobalECs;

  /// CallerCallEdges - For a particular graph, we keep a list of these records
  /// which indicates which graphs call this function and from where.
  struct CallerCallEdge {
    DSGraph *CallerGraph;        // The graph of the caller function.
    const DSCallSite *CS;        // The actual call site.
    Function *CalledFunction;    // The actual function being called.

    CallerCallEdge(DSGraph *G, const DSCallSite *cs, Function *CF)
      : CallerGraph(G), CS(cs), CalledFunction(CF) {}

    bool operator<(const CallerCallEdge &RHS) const {
      return CallerGraph < RHS.CallerGraph ||
            (CallerGraph == RHS.CallerGraph && CS < RHS.CS);
    }
  };

  std::map<DSGraph*, std::vector<CallerCallEdge> > CallerEdges;


  // IndCallMap - We memoize the results of indirect call inlining operations
  // that have multiple targets here to avoid N*M inlining.  The key to the map
  // is a sorted set of callee functions, the value is the DSGraph that holds
  // all of the caller graphs merged together, and the DSCallSite to merge with
  // the arguments for each function.
  std::map<std::vector<Function*>, DSGraph*> IndCallMap;

public:
  ~TDDataStructures() { releaseMyMemory(); }

  virtual bool runOnModule(Module &M);

  bool hasGraph(const Function &F) const {
    return DSInfo.find(const_cast<Function*>(&F)) != DSInfo.end();
  }

  /// getDSGraph - Return the data structure graph for the specified function.
  ///
  DSGraph &getDSGraph(const Function &F) const {
    hash_map<Function*, DSGraph*>::const_iterator I =
      DSInfo.find(const_cast<Function*>(&F));
    if (I != DSInfo.end()) return *I->second;
    return const_cast<TDDataStructures*>(this)->
        getOrCreateDSGraph(const_cast<Function&>(F));
  }

  DSGraph &getGlobalsGraph() const { return *GlobalsGraph; }
  EquivalenceClasses<GlobalValue*> &getGlobalECs() { return GlobalECs; }


  /// deleteValue/copyValue - Interfaces to update the DSGraphs in the program.
  /// These correspond to the interfaces defined in the AliasAnalysis class.
  void deleteValue(Value *V);
  void copyValue(Value *From, Value *To);

  /// print - Print out the analysis results...
  ///
  void print(std::ostream &O, const Module *M) const;

  /// If the pass pipeline is done with this pass, we can release our memory...
  ///
  virtual void releaseMyMemory();

  /// getAnalysisUsage - This obviously provides a data structure graph.
  ///
  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addRequired<BUDataStructures>();
  }

private:
  void markReachableFunctionsExternallyAccessible(DSNode *N,
                                                  hash_set<DSNode*> &Visited);

  void InlineCallersIntoGraph(DSGraph &G);
  DSGraph &getOrCreateDSGraph(Function &F);
  void ComputePostOrder(Function &F, hash_set<DSGraph*> &Visited,
                        std::vector<DSGraph*> &PostOrder);
};


/// CompleteBUDataStructures - This is the exact same as the bottom-up graphs,
/// but we use take a completed call graph and inline all indirect callees into
/// their callers graphs, making the result more useful for things like pool
/// allocation.
///
struct CompleteBUDataStructures : public BUDataStructures {
  virtual bool runOnModule(Module &M);

  bool hasGraph(const Function &F) const {
    return DSInfo.find(const_cast<Function*>(&F)) != DSInfo.end();
  }

  /// getDSGraph - Return the data structure graph for the specified function.
  ///
  DSGraph &getDSGraph(const Function &F) const {
    hash_map<Function*, DSGraph*>::const_iterator I =
      DSInfo.find(const_cast<Function*>(&F));
    assert(I != DSInfo.end() && "Function not in module!");
    return *I->second;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addRequired<BUDataStructures>();

    // FIXME: TEMPORARY (remove once finalization of indirect call sites in the
    // globals graph has been implemented in the BU pass)
    AU.addRequired<TDDataStructures>();
  }

  /// print - Print out the analysis results...
  ///
  void print(std::ostream &O, const Module *M) const;

private:
  unsigned calculateSCCGraphs(DSGraph &FG, std::vector<DSGraph*> &Stack,
                              unsigned &NextID,
                              hash_map<DSGraph*, unsigned> &ValMap);
  DSGraph &getOrCreateGraph(Function &F);
  void processGraph(DSGraph &G);
};


/// EquivClassGraphs - This is the same as the complete bottom-up graphs, but
/// with functions partitioned into equivalence classes and a single merged
/// DS graph for all functions in an equivalence class.  After this merging,
/// graphs are inlined bottom-up on the SCCs of the final (CBU) call graph.
///
struct EquivClassGraphs : public ModulePass {
  CompleteBUDataStructures *CBU;

  DSGraph *GlobalsGraph;

  // DSInfo - one graph for each function.
  hash_map<const Function*, DSGraph*> DSInfo;

  /// ActualCallees - The actual functions callable from indirect call sites.
  ///
  std::set<std::pair<Instruction*, Function*> > ActualCallees;

  // Equivalence class where functions that can potentially be called via the
  // same function pointer are in the same class.
  EquivalenceClasses<Function*> FuncECs;

  /// OneCalledFunction - For each indirect call, we keep track of one
  /// target of the call.  This is used to find equivalence class called by
  /// a call site.
  std::map<DSNode*, Function *> OneCalledFunction;

  /// GlobalECs - The equivalence classes for each global value that is merged
  /// with other global values in the DSGraphs.
  EquivalenceClasses<GlobalValue*> GlobalECs;

public:
  /// EquivClassGraphs - Computes the equivalence classes and then the
  /// folded DS graphs for each class.
  ///
  virtual bool runOnModule(Module &M);

  /// print - Print out the analysis results...
  ///
  void print(std::ostream &O, const Module *M) const;

  EquivalenceClasses<GlobalValue*> &getGlobalECs() { return GlobalECs; }

  /// getDSGraph - Return the data structure graph for the specified function.
  /// This returns the folded graph.  The folded graph is the same as the CBU
  /// graph iff the function is in a singleton equivalence class AND all its
  /// callees also have the same folded graph as the CBU graph.
  ///
  DSGraph &getDSGraph(const Function &F) const {
    hash_map<const Function*, DSGraph*>::const_iterator I = DSInfo.find(&F);
    assert(I != DSInfo.end() && "No graph computed for that function!");
    return *I->second;
  }

  bool hasGraph(const Function &F) const {
    return DSInfo.find(&F) != DSInfo.end();
  }

  /// ContainsDSGraphFor - Return true if we have a graph for the specified
  /// function.
  bool ContainsDSGraphFor(const Function &F) const {
    return DSInfo.find(&F) != DSInfo.end();
  }

  /// getSomeCalleeForCallSite - Return any one callee function at
  /// a call site.
  ///
  Function *getSomeCalleeForCallSite(const CallSite &CS) const;

  DSGraph &getGlobalsGraph() const {
    return *GlobalsGraph;
  }

  typedef std::set<std::pair<Instruction*, Function*> > ActualCalleesTy;
  const ActualCalleesTy &getActualCallees() const {
    return ActualCallees;
  }

  typedef ActualCalleesTy::const_iterator callee_iterator;
  callee_iterator callee_begin(Instruction *I) const {
    return ActualCallees.lower_bound(std::pair<Instruction*,Function*>(I, 0));
  }

  callee_iterator callee_end(Instruction *I) const {
    I = (Instruction*)((char*)I + 1);
    return ActualCallees.lower_bound(std::pair<Instruction*,Function*>(I, 0));
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addRequired<CompleteBUDataStructures>();
  }

private:
  void buildIndirectFunctionSets(Module &M);

  unsigned processSCC(DSGraph &FG, std::vector<DSGraph*> &Stack,
                      unsigned &NextID,
                      std::map<DSGraph*, unsigned> &ValMap);
  void processGraph(DSGraph &FG);

  DSGraph &getOrCreateGraph(Function &F);
};

} // End llvm namespace

#endif
