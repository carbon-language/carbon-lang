//===- llvm/Analysis/MemoryDependenceAnalysis.h - Memory Deps  --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the MemoryDependenceAnalysis analysis pass.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_MEMORY_DEPENDENCE_H
#define LLVM_ANALYSIS_MEMORY_DEPENDENCE_H

#include "llvm/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/PointerIntPair.h"

namespace llvm {
  class Function;
  class FunctionPass;
  class Instruction;
  class CallSite;
  
  /// MemDepResult - A memory dependence query can return one of three different
  /// answers:
  ///   Normal  : The query is dependent on a specific instruction.
  ///   NonLocal: The query does not depend on anything inside this block, but
  ///             we haven't scanned beyond the block to find out what.
  ///   None    : The query does not depend on anything: we found the entry
  ///             block or the allocation site of the memory.
  class MemDepResult {
    enum DepType {
      Invalid = 0, Normal, NonLocal, None
    };
    typedef PointerIntPair<Instruction*, 2, DepType> PairTy;
    PairTy Value;
    explicit MemDepResult(PairTy V) : Value(V) {}
  public:
    MemDepResult() : Value(0, Invalid) {}
    
    /// get methods: These are static ctor methods for creating various
    /// MemDepResult kinds.
    static MemDepResult get(Instruction *Inst) {
      return MemDepResult(PairTy(Inst, Normal));
    }
    static MemDepResult getNonLocal() {
      return MemDepResult(PairTy(0, NonLocal));
    }
    static MemDepResult getNone() {
      return MemDepResult(PairTy(0, None));
    }

    /// isNormal - Return true if this MemDepResult represents a query that is
    /// a normal instruction dependency.
    bool isNormal()          const { return Value.getInt() == Normal; }
    
    /// isNonLocal - Return true if this MemDepResult represents an query that
    /// is transparent to the start of the block, but where a non-local hasn't
    /// been done.
    bool isNonLocal()      const { return Value.getInt() == NonLocal; }
    
    /// isNone - Return true if this MemDepResult represents a query that
    /// doesn't depend on any instruction.
    bool isNone()          const { return Value.getInt() == None; }

    /// getInst() - If this is a normal dependency, return the instruction that
    /// is depended on.  Otherwise, return null.
    Instruction *getInst() const { return isNormal() ? Value.getPointer() : 0; }
    
    bool operator==(const MemDepResult &M) { return M.Value == Value; }
    bool operator!=(const MemDepResult &M) { return M.Value != Value; }
  };

  /// MemoryDependenceAnalysis - This is an analysis that determines, for a
  /// given memory operation, what preceding memory operations it depends on.
  /// It builds on alias analysis information, and tries to provide a lazy,
  /// caching interface to a common kind of alias information query.
  class MemoryDependenceAnalysis : public FunctionPass {
    /// DepType - This enum is used to indicate what flavor of dependence this
    /// is.  If the type is Normal, there is an associated instruction pointer.
    enum DepType {
      /// Normal - This is a normal instruction dependence.  The pointer member
      /// of the DepResultTy pair holds the instruction.
      Normal = 0,

      /// None - This dependence type indicates that the query does not depend
      /// on any instructions, either because it scanned to the start of the
      /// function or it scanned to the definition of the memory
      /// (alloca/malloc).
      None,
      
      /// NonLocal - This marker indicates that the query has no dependency in
      /// the specified block.  To find out more, the client should query other
      /// predecessor blocks.
      NonLocal,
      
      /// Dirty - This is an internal marker indicating that that a cache entry
      /// is dirty.
      Dirty
    };
    typedef PointerIntPair<Instruction*, 2, DepType> DepResultTy;

    // A map from instructions to their dependency, with a boolean
    // flags for whether this mapping is confirmed or not.
    typedef DenseMap<Instruction*,
                     std::pair<DepResultTy, bool> > LocalDepMapType;
    LocalDepMapType LocalDeps;

    // A map from instructions to their non-local dependencies.
    // FIXME: DENSEMAP of DENSEMAP not a great idea.
    typedef DenseMap<Instruction*,
                     DenseMap<BasicBlock*, DepResultTy> > nonLocalDepMapType;
    nonLocalDepMapType depGraphNonLocal;
    
    // A reverse mapping from dependencies to the dependees.  This is
    // used when removing instructions to keep the cache coherent.
    typedef DenseMap<DepResultTy,
                     SmallPtrSet<Instruction*, 4> > reverseDepMapType;
    reverseDepMapType reverseDep;
    
    // A reverse mapping form dependencies to the non-local dependees.
    reverseDepMapType reverseDepNonLocal;
    
  public:
    MemoryDependenceAnalysis() : FunctionPass(&ID) {}
    static char ID;

    /// Pass Implementation stuff.  This doesn't do any analysis.
    ///
    bool runOnFunction(Function &) {return false; }
    
    /// Clean up memory in between runs
    void releaseMemory() {
      LocalDeps.clear();
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
    MemDepResult getDependency(Instruction *query, Instruction *start = 0,
                               BasicBlock *block = 0);
    
    /// getNonLocalDependency - Fills the passed-in map with the non-local 
    /// dependencies of the queries.  The map will contain NonLocal for
    /// blocks between the query and its dependencies.
    void getNonLocalDependency(Instruction* query,
                               DenseMap<BasicBlock*, MemDepResult> &resp);
    
    /// removeInstruction - Remove an instruction from the dependence analysis,
    /// updating the dependence of instructions that previously depended on it.
    void removeInstruction(Instruction *InstToRemove);
    
    /// dropInstruction - Remove an instruction from the analysis, making 
    /// absolutely conservative assumptions when updating the cache.  This is
    /// useful, for example when an instruction is changed rather than removed.
    void dropInstruction(Instruction *InstToDrop);
    
  private:
    DepResultTy ConvFromResult(MemDepResult R) {
      if (Instruction *I = R.getInst())
        return DepResultTy(I, Normal);
      if (R.isNonLocal())
        return DepResultTy(0, NonLocal);
      assert(R.isNone() && "Unknown MemDepResult!");
      return DepResultTy(0, None);
    }
    
    MemDepResult ConvToResult(DepResultTy R) {
      if (R.getInt() == Normal)
        return MemDepResult::get(R.getPointer());
      if (R.getInt() == NonLocal)
        return MemDepResult::getNonLocal();
      assert(R.getInt() == None && "Unknown MemDepResult!");
      return MemDepResult::getNone();
    }
    
    
    /// verifyRemoved - Verify that the specified instruction does not occur
    /// in our internal data structures.
    void verifyRemoved(Instruction *Inst) const;
    
    MemDepResult getCallSiteDependency(CallSite C, Instruction* start,
                                       BasicBlock* block);
    void nonLocalHelper(Instruction* query, BasicBlock* block,
                        DenseMap<BasicBlock*, DepResultTy> &resp);
  };

} // End llvm namespace

#endif
