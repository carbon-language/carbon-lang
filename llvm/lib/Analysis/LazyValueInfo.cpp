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
    /// overdefined - This instruction is not known to be constant, and we know
    /// it has a value.
    overdefined
  };
  
  /// Val: This stores the current lattice value along with the Constant* for
  /// the constant if this is a 'constant' value.
  PointerIntPair<Constant *, 2, LatticeValueTy> Val;
  
public:
  LVILatticeVal() : Val(0, undefined) {}

  static LVILatticeVal get(Constant *C) {
    LVILatticeVal Res;
    Res.markConstant(C);
    return Res;
  }
  
  bool isUndefined() const   { return Val.getInt() == undefined; }
  bool isConstant() const    { return Val.getInt() == constant; }
  bool isOverdefined() const { return Val.getInt() == overdefined; }
  
  Constant *getConstant() const {
    assert(isConstant() && "Cannot get the constant of a non-constant!");
    return Val.getPointer();
  }
  
  /// getConstantInt - If this is a constant with a ConstantInt value, return it
  /// otherwise return null.
  ConstantInt *getConstantInt() const {
    if (isConstant())
      return dyn_cast<ConstantInt>(getConstant());
    return 0;
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
  
  /// mergeIn - Merge the specified lattice value into this one, updating this
  /// one and returning true if anything changed.
  bool mergeIn(const LVILatticeVal &RHS) {
    if (RHS.isUndefined() || isOverdefined()) return false;
    if (RHS.isOverdefined()) return markOverdefined();

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

/// isEqual - Determine whether the specified value is known to be equal or
/// not-equal to the specified constant at the end of the specified block.
LazyValueInfo::Tristate
LazyValueInfo::isEqual(Value *V, Constant *C, BasicBlock *BB) {
  // If already a constant, we can use constant folding.
  if (Constant *VC = dyn_cast<Constant>(V)) {
    // Ignore FP for now.  TODO, consider what form of equality we want.
    if (C->getType()->isFPOrFPVector())
      return Unknown;
    
    Constant *Res = ConstantFoldCompareInstOperands(ICmpInst::ICMP_EQ, VC,C,TD);
    if (ConstantInt *ResCI = dyn_cast<ConstantInt>(Res))
      return ResCI->isZero() ? No : Yes;
  }
  
  // Not a very good implementation.
  return Unknown;
}


