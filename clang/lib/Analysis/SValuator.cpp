// SValuator.cpp - Basic class for all SValuator implementations --*- C++ -*--//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines SValuator, the base class for all (complete) SValuator
//  implementations.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/PathSensitive/SValuator.h"
#include "clang/Analysis/PathSensitive/GRState.h"

using namespace clang;


SVal SValuator::EvalBinOp(const GRState *ST, BinaryOperator::Opcode Op,
                          SVal L, SVal R, QualType T) {

  if (L.isUndef() || R.isUndef())
    return UndefinedVal();
  
  if (L.isUnknown() || R.isUnknown())
    return UnknownVal();
  
  if (isa<Loc>(L)) {
    if (isa<Loc>(R))
      return EvalBinOpLL(Op, cast<Loc>(L), cast<Loc>(R), T);

    return EvalBinOpLN(ST, Op, cast<Loc>(L), cast<NonLoc>(R), T);
  }
  
  if (isa<Loc>(R)) {
    // Support pointer arithmetic where the increment/decrement operand
    // is on the left and the pointer on the right.    
    assert(Op == BinaryOperator::Add || Op == BinaryOperator::Sub);
    
    // Commute the operands.
    return EvalBinOpLN(ST, Op, cast<Loc>(R), cast<NonLoc>(L), T);
  }

  return EvalBinOpNN(Op, cast<NonLoc>(L), cast<NonLoc>(R), T);  
}

SValuator::CastResult SValuator::EvalCast(SVal val, const GRState *state, 
                                          QualType castTy, QualType originalTy){
  
  if (val.isUnknownOrUndef() || castTy == originalTy)
    return CastResult(state, val);
  
  ASTContext &C = ValMgr.getContext();
  
  // For const casts, just propagate the value.
  if (C.getCanonicalType(castTy).getUnqualifiedType() == 
      C.getCanonicalType(originalTy).getUnqualifiedType())
    return CastResult(state, val);
  
  // Check for casts from pointers to integers.
  if (castTy->isIntegerType() && Loc::IsLocType(originalTy))
    return CastResult(state, EvalCastL(cast<Loc>(val), castTy));
  
  // Check for casts from integers to pointers.
  if (Loc::IsLocType(castTy) && originalTy->isIntegerType()) {
    if (nonloc::LocAsInteger *LV = dyn_cast<nonloc::LocAsInteger>(&val)) {
      // Just unpackage the lval and return it.
      return CastResult(state, LV->getLoc());
    }
    
    goto DispatchCast;
  }
  
  // Just pass through function and block pointers.
  if (originalTy->isBlockPointerType() || originalTy->isFunctionPointerType()) {
    assert(Loc::IsLocType(castTy));
    return CastResult(state, val);
  }
  
  // Check for casts from array type to another type.
  if (originalTy->isArrayType()) {
    // We will always decay to a pointer.
    val = ValMgr.getStateManager().ArrayToPointer(cast<Loc>(val));
    
    // Are we casting from an array to a pointer?  If so just pass on
    // the decayed value.
    if (castTy->isPointerType())
      return CastResult(state, val);
    
    // Are we casting from an array to an integer?  If so, cast the decayed
    // pointer value to an integer.
    assert(castTy->isIntegerType());
    
    // FIXME: Keep these here for now in case we decide soon that we
    // need the original decayed type.
    //    QualType elemTy = cast<ArrayType>(originalTy)->getElementType();
    //    QualType pointerTy = C.getPointerType(elemTy);
    return CastResult(state, EvalCastL(cast<Loc>(val), castTy));
  }
  
  // Check for casts from a region to a specific type.
  if (const MemRegion *R = val.getAsRegion()) {
    // FIXME: We should handle the case where we strip off view layers to get
    //  to a desugared type.
    
    assert(Loc::IsLocType(castTy));
    // We get a symbolic function pointer for a dereference of a function
    // pointer, but it is of function type. Example:
    
    //  struct FPRec {
    //    void (*my_func)(int * x);  
    //  };
    //
    //  int bar(int x);
    //
    //  int f1_a(struct FPRec* foo) {
    //    int x;
    //    (*foo->my_func)(&x);
    //    return bar(x)+1; // no-warning
    //  }
    
    assert(Loc::IsLocType(originalTy) || originalTy->isFunctionType() ||
           originalTy->isBlockPointerType());
    
    StoreManager &storeMgr = ValMgr.getStateManager().getStoreManager();
    
    // Delegate to store manager to get the result of casting a region
    // to a different type.
    const StoreManager::CastResult& Res = storeMgr.CastRegion(state, R, castTy);
    
    // Inspect the result.  If the MemRegion* returned is NULL, this
    // expression evaluates to UnknownVal.
    R = Res.getRegion();
    
    if (R)
      return CastResult(Res.getState(), loc::MemRegionVal(R));
    
    return CastResult(Res.getState(), UnknownVal());
  }
    
  // All other cases.  
DispatchCast:
  return CastResult(state,
                    isa<Loc>(val) ? EvalCastL(cast<Loc>(val), castTy) 
                                  : EvalCastNL(cast<NonLoc>(val), castTy));
}
