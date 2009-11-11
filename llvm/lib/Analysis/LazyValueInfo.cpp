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

#include "llvm/Analysis/LazyValueInfo.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Target/TargetData.h"
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
  }
  
};
  
} // end anonymous namespace.


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

Constant *LazyValueInfo::getConstant(Value *V, BasicBlock *BB) {
  // If already a constant, return it.
  if (Constant *VC = dyn_cast<Constant>(V))
    return VC;
    
  // Not a very good implementation.
  return 0;
}

