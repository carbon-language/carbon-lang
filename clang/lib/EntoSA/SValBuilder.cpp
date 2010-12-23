// SValBuilder.cpp - Basic class for all SValBuilder implementations -*- C++ -*-
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines SValBuilder, the base class for all (complete) SValBuilder
//  implementations.
//
//===----------------------------------------------------------------------===//

#include "clang/EntoSA/PathSensitive/MemRegion.h"
#include "clang/EntoSA/PathSensitive/SVals.h"
#include "clang/EntoSA/PathSensitive/SValBuilder.h"
#include "clang/EntoSA/PathSensitive/GRState.h"
#include "clang/EntoSA/PathSensitive/BasicValueFactory.h"

using namespace clang;
using namespace ento;

//===----------------------------------------------------------------------===//
// Basic SVal creation.
//===----------------------------------------------------------------------===//

DefinedOrUnknownSVal SValBuilder::makeZeroVal(QualType T) {
  if (Loc::IsLocType(T))
    return makeNull();

  if (T->isIntegerType())
    return makeIntVal(0, T);

  // FIXME: Handle floats.
  // FIXME: Handle structs.
  return UnknownVal();
}


NonLoc SValBuilder::makeNonLoc(const SymExpr *lhs, BinaryOperator::Opcode op,
                                const llvm::APSInt& v, QualType T) {
  // The Environment ensures we always get a persistent APSInt in
  // BasicValueFactory, so we don't need to get the APSInt from
  // BasicValueFactory again.
  assert(!Loc::IsLocType(T));
  return nonloc::SymExprVal(SymMgr.getSymIntExpr(lhs, op, v, T));
}

NonLoc SValBuilder::makeNonLoc(const SymExpr *lhs, BinaryOperator::Opcode op,
                                const SymExpr *rhs, QualType T) {
  assert(SymMgr.getType(lhs) == SymMgr.getType(rhs));
  assert(!Loc::IsLocType(T));
  return nonloc::SymExprVal(SymMgr.getSymSymExpr(lhs, op, rhs, T));
}


SVal SValBuilder::convertToArrayIndex(SVal V) {
  if (V.isUnknownOrUndef())
    return V;

  // Common case: we have an appropriately sized integer.
  if (nonloc::ConcreteInt* CI = dyn_cast<nonloc::ConcreteInt>(&V)) {
    const llvm::APSInt& I = CI->getValue();
    if (I.getBitWidth() == ArrayIndexWidth && I.isSigned())
      return V;
  }

  return evalCastNL(cast<NonLoc>(V), ArrayIndexTy);
}

DefinedOrUnknownSVal 
SValBuilder::getRegionValueSymbolVal(const TypedRegion* R) {
  QualType T = R->getValueType();

  if (!SymbolManager::canSymbolicate(T))
    return UnknownVal();

  SymbolRef sym = SymMgr.getRegionValueSymbol(R);

  if (Loc::IsLocType(T))
    return loc::MemRegionVal(MemMgr.getSymbolicRegion(sym));

  return nonloc::SymbolVal(sym);
}

DefinedOrUnknownSVal SValBuilder::getConjuredSymbolVal(const void *SymbolTag,
                                                        const Expr *E,
                                                        unsigned Count) {
  QualType T = E->getType();

  if (!SymbolManager::canSymbolicate(T))
    return UnknownVal();

  SymbolRef sym = SymMgr.getConjuredSymbol(E, Count, SymbolTag);

  if (Loc::IsLocType(T))
    return loc::MemRegionVal(MemMgr.getSymbolicRegion(sym));

  return nonloc::SymbolVal(sym);
}

DefinedOrUnknownSVal SValBuilder::getConjuredSymbolVal(const void *SymbolTag,
                                                        const Expr *E,
                                                        QualType T,
                                                        unsigned Count) {
  
  if (!SymbolManager::canSymbolicate(T))
    return UnknownVal();

  SymbolRef sym = SymMgr.getConjuredSymbol(E, T, Count, SymbolTag);

  if (Loc::IsLocType(T))
    return loc::MemRegionVal(MemMgr.getSymbolicRegion(sym));

  return nonloc::SymbolVal(sym);
}

DefinedSVal SValBuilder::getMetadataSymbolVal(const void *SymbolTag,
                                               const MemRegion *MR,
                                               const Expr *E, QualType T,
                                               unsigned Count) {
  assert(SymbolManager::canSymbolicate(T) && "Invalid metadata symbol type");

  SymbolRef sym = SymMgr.getMetadataSymbol(MR, E, T, Count, SymbolTag);

  if (Loc::IsLocType(T))
    return loc::MemRegionVal(MemMgr.getSymbolicRegion(sym));

  return nonloc::SymbolVal(sym);
}

DefinedOrUnknownSVal
SValBuilder::getDerivedRegionValueSymbolVal(SymbolRef parentSymbol,
                                             const TypedRegion *R) {
  QualType T = R->getValueType();

  if (!SymbolManager::canSymbolicate(T))
    return UnknownVal();

  SymbolRef sym = SymMgr.getDerivedSymbol(parentSymbol, R);

  if (Loc::IsLocType(T))
    return loc::MemRegionVal(MemMgr.getSymbolicRegion(sym));

  return nonloc::SymbolVal(sym);
}

DefinedSVal SValBuilder::getFunctionPointer(const FunctionDecl* FD) {
  return loc::MemRegionVal(MemMgr.getFunctionTextRegion(FD));
}

DefinedSVal SValBuilder::getBlockPointer(const BlockDecl *D,
                                          CanQualType locTy,
                                          const LocationContext *LC) {
  const BlockTextRegion *BC =
    MemMgr.getBlockTextRegion(D, locTy, LC->getAnalysisContext());
  const BlockDataRegion *BD = MemMgr.getBlockDataRegion(BC, LC);
  return loc::MemRegionVal(BD);
}

//===----------------------------------------------------------------------===//

SVal SValBuilder::evalBinOp(const GRState *ST, BinaryOperator::Opcode Op,
                          SVal L, SVal R, QualType T) {

  if (L.isUndef() || R.isUndef())
    return UndefinedVal();

  if (L.isUnknown() || R.isUnknown())
    return UnknownVal();

  if (isa<Loc>(L)) {
    if (isa<Loc>(R))
      return evalBinOpLL(ST, Op, cast<Loc>(L), cast<Loc>(R), T);

    return evalBinOpLN(ST, Op, cast<Loc>(L), cast<NonLoc>(R), T);
  }

  if (isa<Loc>(R)) {
    // Support pointer arithmetic where the addend is on the left
    // and the pointer on the right.
    assert(Op == BO_Add);

    // Commute the operands.
    return evalBinOpLN(ST, Op, cast<Loc>(R), cast<NonLoc>(L), T);
  }

  return evalBinOpNN(ST, Op, cast<NonLoc>(L), cast<NonLoc>(R), T);
}

DefinedOrUnknownSVal SValBuilder::evalEQ(const GRState *ST,
                                       DefinedOrUnknownSVal L,
                                       DefinedOrUnknownSVal R) {
  return cast<DefinedOrUnknownSVal>(evalBinOp(ST, BO_EQ, L, R,
                                              Context.IntTy));
}

// FIXME: should rewrite according to the cast kind.
SVal SValBuilder::evalCast(SVal val, QualType castTy, QualType originalTy) {
  if (val.isUnknownOrUndef() || castTy == originalTy)
    return val;

  // For const casts, just propagate the value.
  if (!castTy->isVariableArrayType() && !originalTy->isVariableArrayType())
    if (Context.hasSameUnqualifiedType(castTy, originalTy))
      return val;

  // Check for casts to real or complex numbers.  We don't handle these at all
  // right now.
  if (castTy->isFloatingType() || castTy->isAnyComplexType())
    return UnknownVal();
  
  // Check for casts from integers to integers.
  if (castTy->isIntegerType() && originalTy->isIntegerType())
    return evalCastNL(cast<NonLoc>(val), castTy);

  // Check for casts from pointers to integers.
  if (castTy->isIntegerType() && Loc::IsLocType(originalTy))
    return evalCastL(cast<Loc>(val), castTy);

  // Check for casts from integers to pointers.
  if (Loc::IsLocType(castTy) && originalTy->isIntegerType()) {
    if (nonloc::LocAsInteger *LV = dyn_cast<nonloc::LocAsInteger>(&val)) {
      if (const MemRegion *R = LV->getLoc().getAsRegion()) {
        StoreManager &storeMgr = StateMgr.getStoreManager();
        R = storeMgr.CastRegion(R, castTy);
        return R ? SVal(loc::MemRegionVal(R)) : UnknownVal();
      }
      return LV->getLoc();
    }
    goto DispatchCast;
  }

  // Just pass through function and block pointers.
  if (originalTy->isBlockPointerType() || originalTy->isFunctionPointerType()) {
    assert(Loc::IsLocType(castTy));
    return val;
  }

  // Check for casts from array type to another type.
  if (originalTy->isArrayType()) {
    // We will always decay to a pointer.
    val = StateMgr.ArrayToPointer(cast<Loc>(val));

    // Are we casting from an array to a pointer?  If so just pass on
    // the decayed value.
    if (castTy->isPointerType())
      return val;

    // Are we casting from an array to an integer?  If so, cast the decayed
    // pointer value to an integer.
    assert(castTy->isIntegerType());

    // FIXME: Keep these here for now in case we decide soon that we
    // need the original decayed type.
    //    QualType elemTy = cast<ArrayType>(originalTy)->getElementType();
    //    QualType pointerTy = C.getPointerType(elemTy);
    return evalCastL(cast<Loc>(val), castTy);
  }

  // Check for casts from a region to a specific type.
  if (const MemRegion *R = val.getAsRegion()) {
    // FIXME: We should handle the case where we strip off view layers to get
    //  to a desugared type.

    if (!Loc::IsLocType(castTy)) {
      // FIXME: There can be gross cases where one casts the result of a function
      // (that returns a pointer) to some other value that happens to fit
      // within that pointer value.  We currently have no good way to
      // model such operations.  When this happens, the underlying operation
      // is that the caller is reasoning about bits.  Conceptually we are
      // layering a "view" of a location on top of those bits.  Perhaps
      // we need to be more lazy about mutual possible views, even on an
      // SVal?  This may be necessary for bit-level reasoning as well.
      return UnknownVal();
    }

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

    StoreManager &storeMgr = StateMgr.getStoreManager();

    // Delegate to store manager to get the result of casting a region to a
    // different type.  If the MemRegion* returned is NULL, this expression
    // Evaluates to UnknownVal.
    R = storeMgr.CastRegion(R, castTy);
    return R ? SVal(loc::MemRegionVal(R)) : UnknownVal();
  }

DispatchCast:
  // All other cases.
  return isa<Loc>(val) ? evalCastL(cast<Loc>(val), castTy)
                       : evalCastNL(cast<NonLoc>(val), castTy);
}
