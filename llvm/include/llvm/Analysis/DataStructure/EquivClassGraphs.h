//===-- EquivClassGraphs.h - Merge equiv-class graphs -----------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This pass is the same as the complete bottom-up graphs, but with functions
// partitioned into equivalence classes and a single merged DS graph for all
// functions in an equivalence class.  After this merging, graphs are inlined
// bottom-up on the SCCs of the final (CBU) call graph.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DataStructure/DataStructure.h"
#include "llvm/Analysis/DataStructure/DSGraph.h"
#include "llvm/ADT/STLExtras.h"
#include <vector>
#include <map>

namespace llvm {
  class Module;
  class Function;

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
    
    ActualCalleesTy::iterator callee_begin(Instruction *I) const {
      return ActualCallees.lower_bound(std::pair<Instruction*,Function*>(I, 0));
    }
    
    ActualCalleesTy::iterator callee_end(Instruction *I) const {
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
}; // end llvm namespace
