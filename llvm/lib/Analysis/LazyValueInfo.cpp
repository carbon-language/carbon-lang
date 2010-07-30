//===- LazyValueInfo.cpp - Value constraint analysis ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interface for lazy computation of value constraint
// information.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "lazy-value-info"
#include "llvm/Analysis/LazyValueInfo.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ValueHandle.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/STLExtras.h"
using namespace llvm;

char LazyValueInfo::ID = 0;
INITIALIZE_PASS(LazyValueInfo, "lazy-value-info",
                "Lazy Value Information Analysis", false, true);

namespace llvm {
  FunctionPass *createLazyValueInfoPass() { return new LazyValueInfo(); }
}


//===----------------------------------------------------------------------===//
//                               LVILatticeVal
//===----------------------------------------------------------------------===//

/// LVILatticeVal - This is the information tracked by LazyValueInfo for each
/// value.
///
/// FIXME: This is basically just for bringup, this can be made a lot more rich
/// in the future.
///
namespace {
class LVILatticeVal {
  enum LatticeValueTy {
    /// undefined - This LLVM Value has no known value yet.
    undefined,
    /// constant - This LLVM Value has a specific constant value.
    constant,
    
    /// notconstant - This LLVM value is known to not have the specified value.
    notconstant,
    
    /// overdefined - This instruction is not known to be constant, and we know
    /// it has a value.
    overdefined
  };
  
  /// Val: This stores the current lattice value along with the Constant* for
  /// the constant if this is a 'constant' or 'notconstant' value.
  PointerIntPair<Constant *, 2, LatticeValueTy> Val;
  
public:
  LVILatticeVal() : Val(0, undefined) {}

  static LVILatticeVal get(Constant *C) {
    LVILatticeVal Res;
    Res.markConstant(C);
    return Res;
  }
  static LVILatticeVal getNot(Constant *C) {
    LVILatticeVal Res;
    Res.markNotConstant(C);
    return Res;
  }
  
  bool isUndefined() const   { return Val.getInt() == undefined; }
  bool isConstant() const    { return Val.getInt() == constant; }
  bool isNotConstant() const { return Val.getInt() == notconstant; }
  bool isOverdefined() const { return Val.getInt() == overdefined; }
  
  Constant *getConstant() const {
    assert(isConstant() && "Cannot get the constant of a non-constant!");
    return Val.getPointer();
  }
  
  Constant *getNotConstant() const {
    assert(isNotConstant() && "Cannot get the constant of a non-notconstant!");
    return Val.getPointer();
  }
  
  /// markOverdefined - Return true if this is a change in status.
  bool markOverdefined() {
    if (isOverdefined())
      return false;
    Val.setInt(overdefined);
    return true;
  }

  /// markConstant - Return true if this is a change in status.
  bool markConstant(Constant *V) {
    if (isConstant()) {
      assert(getConstant() == V && "Marking constant with different value");
      return false;
    }
    
    assert(isUndefined());
    Val.setInt(constant);
    assert(V && "Marking constant with NULL");
    Val.setPointer(V);
    return true;
  }
  
  /// markNotConstant - Return true if this is a change in status.
  bool markNotConstant(Constant *V) {
    if (isNotConstant()) {
      assert(getNotConstant() == V && "Marking !constant with different value");
      return false;
    }
    
    if (isConstant())
      assert(getConstant() != V && "Marking not constant with different value");
    else
      assert(isUndefined());

    Val.setInt(notconstant);
    assert(V && "Marking constant with NULL");
    Val.setPointer(V);
    return true;
  }
  
  /// mergeIn - Merge the specified lattice value into this one, updating this
  /// one and returning true if anything changed.
  bool mergeIn(const LVILatticeVal &RHS) {
    if (RHS.isUndefined() || isOverdefined()) return false;
    if (RHS.isOverdefined()) return markOverdefined();

    if (RHS.isNotConstant()) {
      if (isNotConstant()) {
        if (getNotConstant() != RHS.getNotConstant() ||
            isa<ConstantExpr>(getNotConstant()) ||
            isa<ConstantExpr>(RHS.getNotConstant()))
          return markOverdefined();
        return false;
      }
      if (isConstant()) {
        if (getConstant() == RHS.getNotConstant() ||
            isa<ConstantExpr>(RHS.getNotConstant()) ||
            isa<ConstantExpr>(getConstant()))
          return markOverdefined();
        return markNotConstant(RHS.getNotConstant());
      }
      
      assert(isUndefined() && "Unexpected lattice");
      return markNotConstant(RHS.getNotConstant());
    }
    
    // RHS must be a constant, we must be undef, constant, or notconstant.
    if (isUndefined())
      return markConstant(RHS.getConstant());
    
    if (isConstant()) {
      if (getConstant() != RHS.getConstant())
        return markOverdefined();
      return false;
    }

    // If we are known "!=4" and RHS is "==5", stay at "!=4".
    if (getNotConstant() == RHS.getConstant() ||
        isa<ConstantExpr>(getNotConstant()) ||
        isa<ConstantExpr>(RHS.getConstant()))
      return markOverdefined();
    return false;
  }
  
};
  
} // end anonymous namespace.

namespace llvm {
raw_ostream &operator<<(raw_ostream &OS, const LVILatticeVal &Val) {
  if (Val.isUndefined())
    return OS << "undefined";
  if (Val.isOverdefined())
    return OS << "overdefined";

  if (Val.isNotConstant())
    return OS << "notconstant<" << *Val.getNotConstant() << '>';
  return OS << "constant<" << *Val.getConstant() << '>';
}
}

//===----------------------------------------------------------------------===//
//                          LazyValueInfoCache Decl
//===----------------------------------------------------------------------===//

namespace {
  /// LazyValueInfoCache - This is the cache kept by LazyValueInfo which
  /// maintains information about queries across the clients' queries.
  class LazyValueInfoCache {
  public:
    /// BlockCacheEntryTy - This is a computed lattice value at the end of the
    /// specified basic block for a Value* that depends on context.
    typedef std::pair<BasicBlock*, LVILatticeVal> BlockCacheEntryTy;
    
    /// ValueCacheEntryTy - This is all of the cached block information for
    /// exactly one Value*.  The entries are sorted by the BasicBlock* of the
    /// entries, allowing us to do a lookup with a binary search.
    typedef std::map<BasicBlock*, LVILatticeVal> ValueCacheEntryTy;

  private:
     /// LVIValueHandle - A callback value handle update the cache when
     /// values are erased.
    struct LVIValueHandle : public CallbackVH {
      LazyValueInfoCache *Parent;
      
      LVIValueHandle(Value *V, LazyValueInfoCache *P)
        : CallbackVH(V), Parent(P) { }
      
      void deleted();
      void allUsesReplacedWith(Value* V) {
        deleted();
      }

      LVIValueHandle &operator=(Value *V) {
        return *this = LVIValueHandle(V, Parent);
      }
    };

    /// ValueCache - This is all of the cached information for all values,
    /// mapped from Value* to key information.
    std::map<LVIValueHandle, ValueCacheEntryTy> ValueCache;
    
    /// OverDefinedCache - This tracks, on a per-block basis, the set of 
    /// values that are over-defined at the end of that block.  This is required
    /// for cache updating.
    std::set<std::pair<BasicBlock*, Value*> > OverDefinedCache;

  public:
    
    /// getValueInBlock - This is the query interface to determine the lattice
    /// value for the specified Value* at the end of the specified block.
    LVILatticeVal getValueInBlock(Value *V, BasicBlock *BB);

    /// getValueOnEdge - This is the query interface to determine the lattice
    /// value for the specified Value* that is true on the specified edge.
    LVILatticeVal getValueOnEdge(Value *V, BasicBlock *FromBB,BasicBlock *ToBB);
    
    /// threadEdge - This is the update interface to inform the cache that an
    /// edge from PredBB to OldSucc has been threaded to be from PredBB to
    /// NewSucc.
    void threadEdge(BasicBlock *PredBB,BasicBlock *OldSucc,BasicBlock *NewSucc);
  };
} // end anonymous namespace

//===----------------------------------------------------------------------===//
//                              LVIQuery Impl
//===----------------------------------------------------------------------===//

namespace {
  /// LVIQuery - This is a transient object that exists while a query is
  /// being performed.
  ///
  /// TODO: Reuse LVIQuery instead of recreating it for every query, this avoids
  /// reallocation of the densemap on every query.
  class LVIQuery {
    typedef LazyValueInfoCache::BlockCacheEntryTy BlockCacheEntryTy;
    typedef LazyValueInfoCache::ValueCacheEntryTy ValueCacheEntryTy;
    
    /// This is the current value being queried for.
    Value *Val;
    
    /// This is a pointer to the owning cache, for recursive queries.
    LazyValueInfoCache &Parent;

    /// This is all of the cached information about this value.
    ValueCacheEntryTy &Cache;
    
    /// This tracks, for each block, what values are overdefined.
    std::set<std::pair<BasicBlock*, Value*> > &OverDefinedCache;
    
    ///  NewBlocks - This is a mapping of the new BasicBlocks which have been
    /// added to cache but that are not in sorted order.
    DenseSet<BasicBlock*> NewBlockInfo;
  public:
    
    LVIQuery(Value *V, LazyValueInfoCache &P,
             ValueCacheEntryTy &VC,
             std::set<std::pair<BasicBlock*, Value*> > &ODC)
      : Val(V), Parent(P), Cache(VC), OverDefinedCache(ODC) {
    }

    ~LVIQuery() {
      // When the query is done, insert the newly discovered facts into the
      // cache in sorted order.
      if (NewBlockInfo.empty()) return;
      
      for (DenseSet<BasicBlock*>::iterator I = NewBlockInfo.begin(),
           E = NewBlockInfo.end(); I != E; ++I) {
        if (Cache[*I].isOverdefined())
          OverDefinedCache.insert(std::make_pair(*I, Val));
      }
    }

    LVILatticeVal getBlockValue(BasicBlock *BB);
    LVILatticeVal getEdgeValue(BasicBlock *FromBB, BasicBlock *ToBB);

  private:
    LVILatticeVal &getCachedEntryForBlock(BasicBlock *BB);
  };
} // end anonymous namespace

void LazyValueInfoCache::LVIValueHandle::deleted() {
  Parent->ValueCache.erase(*this);
  for (std::set<std::pair<BasicBlock*, Value*> >::iterator
       I = Parent->OverDefinedCache.begin(),
       E = Parent->OverDefinedCache.end();
       I != E; ) {
    std::set<std::pair<BasicBlock*, Value*> >::iterator tmp = I;
    ++I;
    if (tmp->second == getValPtr())
      Parent->OverDefinedCache.erase(tmp);
  }
}


/// getCachedEntryForBlock - See if we already have a value for this block.  If
/// so, return it, otherwise create a new entry in the Cache map to use.
LVILatticeVal &LVIQuery::getCachedEntryForBlock(BasicBlock *BB) {
  NewBlockInfo.insert(BB);
  return Cache[BB];
}

LVILatticeVal LVIQuery::getBlockValue(BasicBlock *BB) {
  // See if we already have a value for this block.
  LVILatticeVal &BBLV = getCachedEntryForBlock(BB);
  
  // If we've already computed this block's value, return it.
  if (!BBLV.isUndefined()) {
    DEBUG(dbgs() << "  reuse BB '" << BB->getName() << "' val=" << BBLV <<'\n');
    return BBLV;
  }

  // Otherwise, this is the first time we're seeing this block.  Reset the
  // lattice value to overdefined, so that cycles will terminate and be
  // conservatively correct.
  BBLV.markOverdefined();
  
  // If V is live into BB, see if our predecessors know anything about it.
  Instruction *BBI = dyn_cast<Instruction>(Val);
  if (BBI == 0 || BBI->getParent() != BB) {
    LVILatticeVal Result;  // Start Undefined.
    unsigned NumPreds = 0;
    
    // Loop over all of our predecessors, merging what we know from them into
    // result.
    for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI) {
      Result.mergeIn(getEdgeValue(*PI, BB));
      
      // If we hit overdefined, exit early.  The BlockVals entry is already set
      // to overdefined.
      if (Result.isOverdefined()) {
        DEBUG(dbgs() << " compute BB '" << BB->getName()
                     << "' - overdefined because of pred.\n");
        return Result;
      }
      ++NumPreds;
    }
    
    // If this is the entry block, we must be asking about an argument.  The
    // value is overdefined.
    if (NumPreds == 0 && BB == &BB->getParent()->front()) {
      assert(isa<Argument>(Val) && "Unknown live-in to the entry block");
      Result.markOverdefined();
      return Result;
    }
    
    // Return the merged value, which is more precise than 'overdefined'.
    assert(!Result.isOverdefined());
    return getCachedEntryForBlock(BB) = Result;
  }
  
  // If this value is defined by an instruction in this block, we have to
  // process it here somehow or return overdefined.
  if (PHINode *PN = dyn_cast<PHINode>(BBI)) {
    LVILatticeVal Result;  // Start Undefined.
    
    // Loop over all of our predecessors, merging what we know from them into
    // result.
    for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI) {
      Value* PhiVal = PN->getIncomingValueForBlock(*PI);
      Result.mergeIn(Parent.getValueOnEdge(PhiVal, *PI, BB));
      
      // If we hit overdefined, exit early.  The BlockVals entry is already set
      // to overdefined.
      if (Result.isOverdefined()) {
        DEBUG(dbgs() << " compute BB '" << BB->getName()
                     << "' - overdefined because of pred.\n");
        return Result;
      }
    }
    
    // Return the merged value, which is more precise than 'overdefined'.
    assert(!Result.isOverdefined());
    return getCachedEntryForBlock(BB) = Result;

  } else {
    
  }
  
  DEBUG(dbgs() << " compute BB '" << BB->getName()
               << "' - overdefined because inst def found.\n");

  LVILatticeVal Result;
  Result.markOverdefined();
  return getCachedEntryForBlock(BB) = Result;
}


/// getEdgeValue - This method attempts to infer more complex 
LVILatticeVal LVIQuery::getEdgeValue(BasicBlock *BBFrom, BasicBlock *BBTo) {
  // TODO: Handle more complex conditionals.  If (v == 0 || v2 < 1) is false, we
  // know that v != 0.
  if (BranchInst *BI = dyn_cast<BranchInst>(BBFrom->getTerminator())) {
    // If this is a conditional branch and only one successor goes to BBTo, then
    // we maybe able to infer something from the condition. 
    if (BI->isConditional() &&
        BI->getSuccessor(0) != BI->getSuccessor(1)) {
      bool isTrueDest = BI->getSuccessor(0) == BBTo;
      assert(BI->getSuccessor(!isTrueDest) == BBTo &&
             "BBTo isn't a successor of BBFrom");
      
      // If V is the condition of the branch itself, then we know exactly what
      // it is.
      if (BI->getCondition() == Val)
        return LVILatticeVal::get(ConstantInt::get(
                               Type::getInt1Ty(Val->getContext()), isTrueDest));
      
      // If the condition of the branch is an equality comparison, we may be
      // able to infer the value.
      if (ICmpInst *ICI = dyn_cast<ICmpInst>(BI->getCondition()))
        if (ICI->isEquality() && ICI->getOperand(0) == Val &&
            isa<Constant>(ICI->getOperand(1))) {
          // We know that V has the RHS constant if this is a true SETEQ or
          // false SETNE. 
          if (isTrueDest == (ICI->getPredicate() == ICmpInst::ICMP_EQ))
            return LVILatticeVal::get(cast<Constant>(ICI->getOperand(1)));
          return LVILatticeVal::getNot(cast<Constant>(ICI->getOperand(1)));
        }
    }
  }

  // If the edge was formed by a switch on the value, then we may know exactly
  // what it is.
  if (SwitchInst *SI = dyn_cast<SwitchInst>(BBFrom->getTerminator())) {
    // If BBTo is the default destination of the switch, we don't know anything.
    // Given a more powerful range analysis we could know stuff.
    if (SI->getCondition() == Val && SI->getDefaultDest() != BBTo) {
      // We only know something if there is exactly one value that goes from
      // BBFrom to BBTo.
      unsigned NumEdges = 0;
      ConstantInt *EdgeVal = 0;
      for (unsigned i = 1, e = SI->getNumSuccessors(); i != e; ++i) {
        if (SI->getSuccessor(i) != BBTo) continue;
        if (NumEdges++) break;
        EdgeVal = SI->getCaseValue(i);
      }
      assert(EdgeVal && "Missing successor?");
      if (NumEdges == 1)
        return LVILatticeVal::get(EdgeVal);
    }
  }
  
  // Otherwise see if the value is known in the block.
  return getBlockValue(BBFrom);
}


//===----------------------------------------------------------------------===//
//                         LazyValueInfoCache Impl
//===----------------------------------------------------------------------===//

LVILatticeVal LazyValueInfoCache::getValueInBlock(Value *V, BasicBlock *BB) {
  // If already a constant, there is nothing to compute.
  if (Constant *VC = dyn_cast<Constant>(V))
    return LVILatticeVal::get(VC);
  
  DEBUG(dbgs() << "LVI Getting block end value " << *V << " at '"
        << BB->getName() << "'\n");
  
  LVILatticeVal Result = LVIQuery(V, *this,
                                  ValueCache[LVIValueHandle(V, this)], 
                                  OverDefinedCache).getBlockValue(BB);
  
  DEBUG(dbgs() << "  Result = " << Result << "\n");
  return Result;
}

LVILatticeVal LazyValueInfoCache::
getValueOnEdge(Value *V, BasicBlock *FromBB, BasicBlock *ToBB) {
  // If already a constant, there is nothing to compute.
  if (Constant *VC = dyn_cast<Constant>(V))
    return LVILatticeVal::get(VC);
  
  DEBUG(dbgs() << "LVI Getting edge value " << *V << " from '"
        << FromBB->getName() << "' to '" << ToBB->getName() << "'\n");
  
  LVILatticeVal Result =
    LVIQuery(V, *this, ValueCache[LVIValueHandle(V, this)],
             OverDefinedCache).getEdgeValue(FromBB, ToBB);
  
  DEBUG(dbgs() << "  Result = " << Result << "\n");
  
  return Result;
}

void LazyValueInfoCache::threadEdge(BasicBlock *PredBB, BasicBlock *OldSucc,
                                    BasicBlock *NewSucc) {
  // When an edge in the graph has been threaded, values that we could not 
  // determine a value for before (i.e. were marked overdefined) may be possible
  // to solve now.  We do NOT try to proactively update these values.  Instead,
  // we clear their entries from the cache, and allow lazy updating to recompute
  // them when needed.
  
  // The updating process is fairly simple: we need to dropped cached info
  // for all values that were marked overdefined in OldSucc, and for those same
  // values in any successor of OldSucc (except NewSucc) in which they were
  // also marked overdefined.
  std::vector<BasicBlock*> worklist;
  worklist.push_back(OldSucc);
  
  DenseSet<Value*> ClearSet;
  for (std::set<std::pair<BasicBlock*, Value*> >::iterator
       I = OverDefinedCache.begin(), E = OverDefinedCache.end(); I != E; ++I) {
    if (I->first == OldSucc)
      ClearSet.insert(I->second);
  }
  
  // Use a worklist to perform a depth-first search of OldSucc's successors.
  // NOTE: We do not need a visited list since any blocks we have already
  // visited will have had their overdefined markers cleared already, and we
  // thus won't loop to their successors.
  while (!worklist.empty()) {
    BasicBlock *ToUpdate = worklist.back();
    worklist.pop_back();
    
    // Skip blocks only accessible through NewSucc.
    if (ToUpdate == NewSucc) continue;
    
    bool changed = false;
    for (DenseSet<Value*>::iterator I = ClearSet.begin(),E = ClearSet.end();
         I != E; ++I) {
      // If a value was marked overdefined in OldSucc, and is here too...
      std::set<std::pair<BasicBlock*, Value*> >::iterator OI =
        OverDefinedCache.find(std::make_pair(ToUpdate, *I));
      if (OI == OverDefinedCache.end()) continue;

      // Remove it from the caches.
      ValueCacheEntryTy &Entry = ValueCache[LVIValueHandle(*I, this)];
      ValueCacheEntryTy::iterator CI = Entry.find(ToUpdate);
        
      assert(CI != Entry.end() && "Couldn't find entry to update?");
      Entry.erase(CI);
      OverDefinedCache.erase(OI);

      // If we removed anything, then we potentially need to update 
      // blocks successors too.
      changed = true;
    }
        
    if (!changed) continue;
    
    worklist.insert(worklist.end(), succ_begin(ToUpdate), succ_end(ToUpdate));
  }
}

//===----------------------------------------------------------------------===//
//                            LazyValueInfo Impl
//===----------------------------------------------------------------------===//

bool LazyValueInfo::runOnFunction(Function &F) {
  TD = getAnalysisIfAvailable<TargetData>();
  // Fully lazy.
  return false;
}

/// getCache - This lazily constructs the LazyValueInfoCache.
static LazyValueInfoCache &getCache(void *&PImpl) {
  if (!PImpl)
    PImpl = new LazyValueInfoCache();
  return *static_cast<LazyValueInfoCache*>(PImpl);
}

void LazyValueInfo::releaseMemory() {
  // If the cache was allocated, free it.
  if (PImpl) {
    delete &getCache(PImpl);
    PImpl = 0;
  }
}

Constant *LazyValueInfo::getConstant(Value *V, BasicBlock *BB) {
  LVILatticeVal Result = getCache(PImpl).getValueInBlock(V, BB);
  
  if (Result.isConstant())
    return Result.getConstant();
  return 0;
}

/// getConstantOnEdge - Determine whether the specified value is known to be a
/// constant on the specified edge.  Return null if not.
Constant *LazyValueInfo::getConstantOnEdge(Value *V, BasicBlock *FromBB,
                                           BasicBlock *ToBB) {
  LVILatticeVal Result = getCache(PImpl).getValueOnEdge(V, FromBB, ToBB);
  
  if (Result.isConstant())
    return Result.getConstant();
  return 0;
}

/// getPredicateOnEdge - Determine whether the specified value comparison
/// with a constant is known to be true or false on the specified CFG edge.
/// Pred is a CmpInst predicate.
LazyValueInfo::Tristate
LazyValueInfo::getPredicateOnEdge(unsigned Pred, Value *V, Constant *C,
                                  BasicBlock *FromBB, BasicBlock *ToBB) {
  LVILatticeVal Result = getCache(PImpl).getValueOnEdge(V, FromBB, ToBB);
  
  // If we know the value is a constant, evaluate the conditional.
  Constant *Res = 0;
  if (Result.isConstant()) {
    Res = ConstantFoldCompareInstOperands(Pred, Result.getConstant(), C, TD);
    if (ConstantInt *ResCI = dyn_cast_or_null<ConstantInt>(Res))
      return ResCI->isZero() ? False : True;
    return Unknown;
  }
  
  if (Result.isNotConstant()) {
    // If this is an equality comparison, we can try to fold it knowing that
    // "V != C1".
    if (Pred == ICmpInst::ICMP_EQ) {
      // !C1 == C -> false iff C1 == C.
      Res = ConstantFoldCompareInstOperands(ICmpInst::ICMP_NE,
                                            Result.getNotConstant(), C, TD);
      if (Res->isNullValue())
        return False;
    } else if (Pred == ICmpInst::ICMP_NE) {
      // !C1 != C -> true iff C1 == C.
      Res = ConstantFoldCompareInstOperands(ICmpInst::ICMP_NE,
                                            Result.getNotConstant(), C, TD);
      if (Res->isNullValue())
        return True;
    }
    return Unknown;
  }
  
  return Unknown;
}

void LazyValueInfo::threadEdge(BasicBlock *PredBB, BasicBlock *OldSucc,
                               BasicBlock* NewSucc) {
  getCache(PImpl).threadEdge(PredBB, OldSucc, NewSucc);
}
