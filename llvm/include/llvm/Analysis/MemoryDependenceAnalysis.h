//===- llvm/Analysis/MemoryDependenceAnalysis.h - Memory Deps  --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines an analysis that determines, for a given memory operation,
// what preceding memory operations it depends on.  It builds on alias analysis
// information, and tries to provide a lazy, caching interface to a common kind
// of alias information query.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_MEMORY_DEPENDENCE_H
#define LLVM_ANALYSIS_MEMORY_DEPENDENCE_H

#include "llvm/Pass.h"
#include "llvm/Support/CallSite.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class Function;
class FunctionPass;
class Instruction;

class MemoryDependenceAnalysis : public FunctionPass {
  private:
    // A map from instructions to their dependency, with a boolean
    // flags for whether this mapping is confirmed or not
    typedef DenseMap<Instruction*, std::pair<Instruction*, bool> > 
            depMapType;
    depMapType depGraphLocal;

    // A map from instructions to their non-local dependencies.
    typedef DenseMap<Instruction*, DenseMap<BasicBlock*, Value*> >
            nonLocalDepMapType;
    nonLocalDepMapType depGraphNonLocal;
    
    // A reverse mapping form dependencies to the dependees.  This is
    // used when removing instructions to keep the cache coherent.
    typedef DenseMap<Value*, SmallPtrSet<Instruction*, 4> >
            reverseDepMapType;
    reverseDepMapType reverseDep;
    
    // A reverse mapping form dependencies to the non-local dependees.
    reverseDepMapType reverseDepNonLocal;
    
  public:
    void ping(Instruction* D);

    // Special marker indicating that the query has no dependency
    // in the specified block.
    static Instruction* const NonLocal;
    
    // Special marker indicating that the query has no dependency at all
    static Instruction* const None;
    
    
    // Special marker indicating a dirty cache entry
    static Instruction* const Dirty;
    
    static char ID; // Class identification, replacement for typeinfo
    MemoryDependenceAnalysis() : FunctionPass((intptr_t)&ID) {}

    /// Pass Implementation stuff.  This doesn't do any analysis.
    ///
    bool runOnFunction(Function &) {return false; }
    
    /// Clean up memory in between runs
    void releaseMemory() {
      depGraphLocal.clear();
      depGraphNonLocal.clear();
      reverseDep.clear();
      reverseDepNonLocal.clear();
    }

    /// getAnalysisUsage - Does not modify anything.  It uses Value Numbering
    /// and Alias Analysis.
    ///
    virtual void getAnalysisUsage(AnalysisUsage &AU) const;
    
    /// getDependency - Return the instruction on which a memory operation
    /// depends, starting with start.
    Instruction* getDependency(Instruction* query, Instruction* start = 0,
                               BasicBlock* block = 0);
    
    /// getNonLocalDependency - Fills the passed-in map with the non-local 
    /// dependencies of the queries.  The map will contain NonLocal for
    /// blocks between the query and its dependencies.
    void getNonLocalDependency(Instruction* query,
                               DenseMap<BasicBlock*, Value*>& resp);
    
    /// removeInstruction - Remove an instruction from the dependence analysis,
    /// updating the dependence of instructions that previously depended on it.
    void removeInstruction(Instruction* rem);
    
  private:
    Instruction* getCallSiteDependency(CallSite C, Instruction* start,
                                       BasicBlock* block);
    void nonLocalHelper(Instruction* query, BasicBlock* block,
                        DenseMap<BasicBlock*, Value*>& resp);
  };

} // End llvm namespace

#endif
