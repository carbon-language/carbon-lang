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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerIntPair.h"
using namespace llvm;

char LazyValueInfo::ID = 0;
static RegisterPass<LazyValueInfo>
X("lazy-value-info", "Lazy Value Information Analysis", false, true);

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
        if (getNotConstant() != RHS.getNotConstant())
          return markOverdefined();
        return false;
      }
      if (isConstant() && getConstant() != RHS.getNotConstant())
        return markOverdefined();
      return markNotConstant(RHS.getNotConstant());
    }
    
    // RHS must be a constant, we must be undef or constant.
    if (isConstant() && getConstant() != RHS.getConstant())
      return markOverdefined();
    return markConstant(RHS.getConstant());
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
//                            LazyValueInfo Impl
//===----------------------------------------------------------------------===//

bool LazyValueInfo::runOnFunction(Function &F) {
  TD = getAnalysisIfAvailable<TargetData>();
  // Fully lazy.
  return false;
}

void LazyValueInfo::releaseMemory() {
  // No caching yet.
}

static LVILatticeVal GetValueInBlock(Value *V, BasicBlock *BB,
                                     DenseMap<BasicBlock*, LVILatticeVal> &);

static LVILatticeVal GetValueOnEdge(Value *V, BasicBlock *BBFrom,
                                    BasicBlock *BBTo,
                              DenseMap<BasicBlock*, LVILatticeVal> &BlockVals) {
  // FIXME: Pull edge logic out of jump threading.
  
  
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
      if (BI->getCondition() == V)
        return LVILatticeVal::get(ConstantInt::get(
                                 Type::getInt1Ty(V->getContext()), isTrueDest));
      
      // If the condition of the branch is an equality comparison, we may be
      // able to infer the value.
      if (ICmpInst *ICI = dyn_cast<ICmpInst>(BI->getCondition()))
        if (ICI->isEquality() && ICI->getOperand(0) == V &&
            isa<Constant>(ICI->getOperand(1))) {
          // We know that V has the RHS constant if this is a true SETEQ or
          // false SETNE. 
          if (isTrueDest == (ICI->getPredicate() == ICmpInst::ICMP_EQ))
            return LVILatticeVal::get(cast<Constant>(ICI->getOperand(1)));
          return LVILatticeVal::getNot(cast<Constant>(ICI->getOperand(1)));
        }
    }
  }
  
  // TODO: Info from switch.
  
  
  // Otherwise see if the value is known in the block.
  return GetValueInBlock(V, BBFrom, BlockVals);
}

static LVILatticeVal GetValueInBlock(Value *V, BasicBlock *BB,
                              DenseMap<BasicBlock*, LVILatticeVal> &BlockVals) {
  // See if we already have a value for this block.
  LVILatticeVal &BBLV = BlockVals[BB];

  // If we've already computed this block's value, return it.
  if (!BBLV.isUndefined())
    return BBLV;
  
  // Otherwise, this is the first time we're seeing this block.  Reset the
  // lattice value to overdefined, so that cycles will terminate and be
  // conservatively correct.
  BBLV.markOverdefined();

  LVILatticeVal Result;  // Start Undefined.
  
  // If V is live in to BB, see if our predecessors know anything about it.
  Instruction *BBI = dyn_cast<Instruction>(V);
  if (BBI == 0 || BBI->getParent() != BB) {
    unsigned NumPreds = 0;
    
    // Loop over all of our predecessors, merging what we know from them into
    // result.
    for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI) {
      Result.mergeIn(GetValueOnEdge(V, *PI, BB, BlockVals));
      
      // If we hit overdefined, exit early.  The BlockVals entry is already set
      // to overdefined.
      if (Result.isOverdefined())
        return Result;
      ++NumPreds;
    }
    
    // If this is the entry block, we must be asking about an argument.  The
    // value is overdefined.
    if (NumPreds == 0 && BB == &BB->getParent()->front()) {
      assert(isa<Argument>(V) && "Unknown live-in to the entry block");
      Result.markOverdefined();
      return Result;
    }

    // Return the merged value, which is more precise than 'overdefined'.
    assert(!Result.isOverdefined());
    return BlockVals[BB] = Result;
  }

  // If this value is defined by an instruction in this block, we have to
  // process it here somehow or return overdefined.
  if (PHINode *PN = dyn_cast<PHINode>(BBI)) {
    (void)PN;
    // TODO: PHI Translation in preds.
  } else {
    
  }
  
  Result.markOverdefined();
  return BlockVals[BB] = Result;
}


Constant *LazyValueInfo::getConstant(Value *V, BasicBlock *BB) {
  // If already a constant, return it.
  if (Constant *VC = dyn_cast<Constant>(V))
    return VC;
  
  DenseMap<BasicBlock*, LVILatticeVal> BlockValues;
  
  DEBUG(errs() << "Getting value " << *V << " at end of block '"
               << BB->getName() << "'\n");
  LVILatticeVal Result = GetValueInBlock(V, BB, BlockValues);
  
  DEBUG(errs() << "  Result = " << Result << "\n");

  if (Result.isConstant())
    return Result.getConstant();
  return 0;
}

/// getConstantOnEdge - Determine whether the specified value is known to be a
/// constant on the specified edge.  Return null if not.
Constant *LazyValueInfo::getConstantOnEdge(Value *V, BasicBlock *FromBB,
                                           BasicBlock *ToBB) {
  // If already a constant, return it.
  if (Constant *VC = dyn_cast<Constant>(V))
    return VC;
  
  DenseMap<BasicBlock*, LVILatticeVal> BlockValues;
  
  DEBUG(errs() << "Getting value " << *V << " on edge from '"
               << FromBB->getName() << "' to '" << ToBB->getName() << "'\n");
  LVILatticeVal Result = GetValueOnEdge(V, FromBB, ToBB, BlockValues);
  
  DEBUG(errs() << "  Result = " << Result << "\n");
  
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
  LVILatticeVal Result;
  
  // If already a constant, we can use constant folding.
  if (Constant *VC = dyn_cast<Constant>(V)) {
    Result = LVILatticeVal::get(VC);
  } else {
    DenseMap<BasicBlock*, LVILatticeVal> BlockValues;
    
    DEBUG(errs() << "Getting value " << *V << " on edge from '"
          << FromBB->getName() << "' to '" << ToBB->getName() << "'\n");
    Result = GetValueOnEdge(V, FromBB, ToBB, BlockValues);
    DEBUG(errs() << "  Result = " << Result << "\n");
  }
  
  // If we know the value is a constant, evaluate the conditional.
  Constant *Res = 0;
  if (Result.isConstant()) {
    Res = ConstantFoldCompareInstOperands(Pred, Result.getConstant(), C, TD);
    if (ConstantInt *ResCI = dyn_cast_or_null<ConstantInt>(Res))
      return ResCI->isZero() ? False : True;
  } else if (Result.isNotConstant()) {
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
      Res = ConstantFoldCompareInstOperands(ICmpInst::ICMP_EQ,
                                            Result.getNotConstant(), C, TD);
      if (Res->isNullValue())
        return True;
    }
  }
  
  return Unknown;
}


