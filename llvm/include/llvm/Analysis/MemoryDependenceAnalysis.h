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

#include "llvm/BasicBlock.h"
#include "llvm/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/PointerIntPair.h"

namespace llvm {
  class Function;
  class FunctionPass;
  class Instruction;
  class CallSite;
  class AliasAnalysis;
  class TargetData;
  
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
    bool isNormal() const { return Value.getInt() == Normal; }
    
    /// isNonLocal - Return true if this MemDepResult represents an query that
    /// is transparent to the start of the block, but where a non-local hasn't
    /// been done.
    bool isNonLocal() const { return Value.getInt() == NonLocal; }
    
    /// isNone - Return true if this MemDepResult represents a query that
    /// doesn't depend on any instruction.
    bool isNone() const { return Value.getInt() == None; }

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
  ///
  /// The dependency information returned is somewhat unusual, but is pragmatic.
  /// If queried about a store or call that might modify memory, the analysis
  /// will return the instruction[s] that may either load from that memory or
  /// store to it.  If queried with a load or call that can never modify memory,
  /// the analysis will return calls and stores that might modify the pointer,
  /// but generally does not return loads unless a) they are volatile, or
  /// b) they load from *must-aliased* pointers.  Returning a dependence on
  /// must-alias'd pointers instead of all pointers interacts well with the
  /// internal caching mechanism.
  ///
  class MemoryDependenceAnalysis : public FunctionPass {
    /// DepType - This enum is used to indicate what flavor of dependence this
    /// is.  If the type is Normal, there is an associated instruction pointer.
    enum DepType {
      /// Dirty - Entries with this marker occur in a LocalDeps map or
      /// NonLocalDeps map when the instruction they previously referenced was
      /// removed from MemDep.  In either case, the entry may include an
      /// instruction pointer.  If so, the pointer is an instruction in the
      /// block where scanning can start from, saving some work.
      ///
      /// In a default-constructed DepResultTy object, the type will be Dirty
      /// and the instruction pointer will be null.
      ///
      Dirty = 0,
      
      /// Normal - This is a normal instruction dependence.  The pointer member
      /// of the DepResultTy pair holds the instruction.
      Normal,

      /// None - This dependence type indicates that the query does not depend
      /// on any instructions, either because it is not a memory instruction or
      /// because it scanned to the definition of the memory (alloca/malloc)
      /// being accessed.
      None,
      
      /// NonLocal - This marker indicates that the query has no dependency in
      /// the specified block.  To find out more, the client should query other
      /// predecessor blocks.
      NonLocal
    };
    typedef PointerIntPair<Instruction*, 2, DepType> DepResultTy;

    // A map from instructions to their dependency.
    typedef DenseMap<Instruction*, DepResultTy> LocalDepMapType;
    LocalDepMapType LocalDeps;

    typedef DenseMap<BasicBlock*, DepResultTy> NonLocalDepInfo;
    
    /// PerInstNLInfo - This is the instruction we keep for each cached access
    /// that we have for an instruction.  The pointer is an owning pointer and
    /// the bool indicates whether we have any dirty bits in the set.
    typedef PointerIntPair<NonLocalDepInfo*, 1, bool> PerInstNLInfo;
    
    // A map from instructions to their non-local dependencies.
    typedef DenseMap<Instruction*, PerInstNLInfo> NonLocalDepMapType;
      
    NonLocalDepMapType NonLocalDeps;
    
    // A reverse mapping from dependencies to the dependees.  This is
    // used when removing instructions to keep the cache coherent.
    typedef DenseMap<Instruction*,
                     SmallPtrSet<Instruction*, 4> > ReverseDepMapType;
    ReverseDepMapType ReverseLocalDeps;
    
    // A reverse mapping form dependencies to the non-local dependees.
    ReverseDepMapType ReverseNonLocalDeps;
    
    /// Current AA implementation, just a cache.
    AliasAnalysis *AA;
    TargetData *TD;
  public:
    MemoryDependenceAnalysis() : FunctionPass(&ID) {}
    static char ID;

    /// Pass Implementation stuff.  This doesn't do any analysis eagerly.
    bool runOnFunction(Function &);
    
    /// Clean up memory in between runs
    void releaseMemory() {
      LocalDeps.clear();
      for (NonLocalDepMapType::iterator I = NonLocalDeps.begin(),
           E = NonLocalDeps.end(); I != E; ++I)
        delete I->second.getPointer();
      NonLocalDeps.clear();
      ReverseLocalDeps.clear();
      ReverseNonLocalDeps.clear();
    }

    /// getAnalysisUsage - Does not modify anything.  It uses Value Numbering
    /// and Alias Analysis.
    ///
    virtual void getAnalysisUsage(AnalysisUsage &AU) const;
    
    /// getDependency - Return the instruction on which a memory operation
    /// depends.  See the class comment for more details.
    MemDepResult getDependency(Instruction *QueryInst);

    /// getDependencyFrom - Return the instruction on which the memory operation
    /// 'QueryInst' depends.  This starts scanning from the instruction before
    /// the position indicated by ScanIt.
    ///
    /// Note that this method does no caching at all.  You should use
    /// getDependency where possible.
    MemDepResult getDependencyFrom(Instruction *QueryInst,
                                   BasicBlock::iterator ScanIt, BasicBlock *BB){
      return ConvToResult(getDependencyFromInternal(QueryInst, ScanIt, BB));
    }

    
    /// getNonLocalDependency - Perform a full dependency query for the
    /// specified instruction, returning the set of blocks that the value is
    /// potentially live across.  The returned set of results will include a
    /// "NonLocal" result for all blocks where the value is live across.
    ///
    /// This method assumes the instruction returns a "nonlocal" dependency
    /// within its own block.
    void getNonLocalDependency(Instruction *QueryInst,
                               SmallVectorImpl<std::pair<BasicBlock*, 
                                                       MemDepResult> > &Result);
    
    /// removeInstruction - Remove an instruction from the dependence analysis,
    /// updating the dependence of instructions that previously depended on it.
    void removeInstruction(Instruction *InstToRemove);
    
  private:
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
    
    /// getDependencyFromInternal - Return the instruction on which the memory
    /// operation 'QueryInst' depends.  This starts scanning from the
    /// instruction before the position indicated by ScanIt.
    DepResultTy getDependencyFromInternal(Instruction *QueryInst,
                                   BasicBlock::iterator ScanIt, BasicBlock *BB);
    DepResultTy getCallSiteDependency(CallSite C, BasicBlock::iterator ScanIt,
                                      BasicBlock *BB);
  };

} // End llvm namespace

#endif
