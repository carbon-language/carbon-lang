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

#include "clang/StaticAnalyzer/Core/PathSensitive/MemRegion.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SVals.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SValBuilder.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/GRState.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/BasicValueFactory.h"

using namespace clang;
using namespace ento;

//===----------------------------------------------------------------------===//
// Basic SVal creation.
//===----------------------------------------------------------------------===//

DefinedOrUnknownSVal SValBuilder::makeZeroVal(QualType type) {
  if (Loc::isLocType(type))
    return makeNull();

  if (type->isIntegerType())
    return makeIntVal(0, type);

  // FIXME: Handle floats.
  // FIXME: Handle structs.
  return UnknownVal();
}


NonLoc SValBuilder::makeNonLoc(const SymExpr *lhs, BinaryOperator::Opcode op,
                                const llvm::APSInt& rhs, QualType type) {
  // The Environment ensures we always get a persistent APSInt in
  // BasicValueFactory, so we don't need to get the APSInt from
  // BasicValueFactory again.
  assert(!Loc::isLocType(type));
  return nonloc::SymExprVal(SymMgr.getSymIntExpr(lhs, op, rhs, type));
}

NonLoc SValBuilder::makeNonLoc(const SymExpr *lhs, BinaryOperator::Opcode op,
                               const SymExpr *rhs, QualType type) {
  assert(SymMgr.getType(lhs) == SymMgr.getType(rhs));
  assert(!Loc::isLocType(type));
  return nonloc::SymExprVal(SymMgr.getSymSymExpr(lhs, op, rhs, type));
}


SVal SValBuilder::convertToArrayIndex(SVal val) {
  if (val.isUnknownOrUndef())
    return val;

  // Common case: we have an appropriately sized integer.
  if (nonloc::ConcreteInt* CI = dyn_cast<nonloc::ConcreteInt>(&val)) {
    const llvm::APSInt& I = CI->getValue();
    if (I.getBitWidth() == ArrayIndexWidth && I.isSigned())
      return val;
  }

  return evalCastFromNonLoc(cast<NonLoc>(val), ArrayIndexTy);
}

DefinedOrUnknownSVal 
SValBuilder::getRegionValueSymbolVal(const TypedRegion* region) {
  QualType T = region->getValueType();

  if (!SymbolManager::canSymbolicate(T))
    return UnknownVal();

  SymbolRef sym = SymMgr.getRegionValueSymbol(region);

  if (Loc::isLocType(T))
    return loc::MemRegionVal(MemMgr.getSymbolicRegion(sym));

  return nonloc::SymbolVal(sym);
}

DefinedOrUnknownSVal SValBuilder::getConjuredSymbolVal(const void *symbolTag,
                                                       const Expr *expr,
                                                       unsigned count) {
  QualType T = expr->getType();

  if (!SymbolManager::canSymbolicate(T))
    return UnknownVal();

  SymbolRef sym = SymMgr.getConjuredSymbol(expr, count, symbolTag);

  if (Loc::isLocType(T))
    return loc::MemRegionVal(MemMgr.getSymbolicRegion(sym));

  return nonloc::SymbolVal(sym);
}

DefinedOrUnknownSVal SValBuilder::getConjuredSymbolVal(const void *symbolTag,
                                                       const Expr *expr,
                                                       QualType type,
                                                       unsigned count) {
  
  if (!SymbolManager::canSymbolicate(type))
    return UnknownVal();

  SymbolRef sym = SymMgr.getConjuredSymbol(expr, type, count, symbolTag);

  if (Loc::isLocType(type))
    return loc::MemRegionVal(MemMgr.getSymbolicRegion(sym));

  return nonloc::SymbolVal(sym);
}

DefinedSVal SValBuilder::getMetadataSymbolVal(const void *symbolTag,
                                              const MemRegion *region,
                                              const Expr *expr, QualType type,
                                              unsigned count) {
  assert(SymbolManager::canSymbolicate(type) && "Invalid metadata symbol type");

  SymbolRef sym =
      SymMgr.getMetadataSymbol(region, expr, type, count, symbolTag);

  if (Loc::isLocType(type))
    return loc::MemRegionVal(MemMgr.getSymbolicRegion(sym));

  return nonloc::SymbolVal(sym);
}

DefinedOrUnknownSVal
SValBuilder::getDerivedRegionValueSymbolVal(SymbolRef parentSymbol,
                                             const TypedRegion *region) {
  QualType T = region->getValueType();

  if (!SymbolManager::canSymbolicate(T))
    return UnknownVal();

  SymbolRef sym = SymMgr.getDerivedSymbol(parentSymbol, region);

  if (Loc::isLocType(T))
    return loc::MemRegionVal(MemMgr.getSymbolicRegion(sym));

  return nonloc::SymbolVal(sym);
}

DefinedSVal SValBuilder::getFunctionPointer(const FunctionDecl* func) {
  return loc::MemRegionVal(MemMgr.getFunctionTextRegion(func));
}

DefinedSVal SValBuilder::getBlockPointer(const BlockDecl *block,
                                         CanQualType locTy,
                                         const LocationContext *locContext) {
  const BlockTextRegion *BC =
    MemMgr.getBlockTextRegion(block, locTy, locContext->getAnalysisContext());
  const BlockDataRegion *BD = MemMgr.getBlockDataRegion(BC, locContext);
  return loc::MemRegionVal(BD);
}

//===----------------------------------------------------------------------===//

SVal SValBuilder::evalBinOp(const GRState *state, BinaryOperator::Opcode op,
                            SVal lhs, SVal rhs, QualType type) {

  if (lhs.isUndef() || rhs.isUndef())
    return UndefinedVal();

  if (lhs.isUnknown() || rhs.isUnknown())
    return UnknownVal();

  if (isa<Loc>(lhs)) {
    if (isa<Loc>(rhs))
      return evalBinOpLL(state, op, cast<Loc>(lhs), cast<Loc>(rhs), type);

    return evalBinOpLN(state, op, cast<Loc>(lhs), cast<NonLoc>(rhs), type);
  }

  if (isa<Loc>(rhs)) {
    // Support pointer arithmetic where the addend is on the left
    // and the pointer on the right.
    assert(op == BO_Add);

    // Commute the operands.
    return evalBinOpLN(state, op, cast<Loc>(rhs), cast<NonLoc>(lhs), type);
  }

  return evalBinOpNN(state, op, cast<NonLoc>(lhs), cast<NonLoc>(rhs), type);
}

DefinedOrUnknownSVal SValBuilder::evalEQ(const GRState *state,
                                         DefinedOrUnknownSVal lhs,
                                         DefinedOrUnknownSVal rhs) {
  return cast<DefinedOrUnknownSVal>(evalBinOp(state, BO_EQ, lhs, rhs,
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
    return evalCastFromNonLoc(cast<NonLoc>(val), castTy);

  // Check for casts from pointers to integers.
  if (castTy->isIntegerType() && Loc::isLocType(originalTy))
    return evalCastFromLoc(cast<Loc>(val), castTy);

  // Check for casts from integers to pointers.
  if (Loc::isLocType(castTy) && originalTy->isIntegerType()) {
    if (nonloc::LocAsInteger *LV = dyn_cast<nonloc::LocAsInteger>(&val)) {
      if (const MemRegion *R = LV->getLoc().getAsRegion()) {
        StoreManager &storeMgr = StateMgr.getStoreManager();
        R = storeMgr.castRegion(R, castTy);
        return R ? SVal(loc::MemRegionVal(R)) : UnknownVal();
      }
      return LV->getLoc();
    }
    goto DispatchCast;
  }

  // Just pass through function and block pointers.
  if (originalTy->isBlockPointerType() || originalTy->isFunctionPointerType()) {
    assert(Loc::isLocType(castTy));
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
    return evalCastFromLoc(cast<Loc>(val), castTy);
  }

  // Check for casts from a region to a specific type.
  if (const MemRegion *R = val.getAsRegion()) {
    // FIXME: We should handle the case where we strip off view layers to get
    //  to a desugared type.

    if (!Loc::isLocType(castTy)) {
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

    assert(Loc::isLocType(originalTy) || originalTy->isFunctionType() ||
           originalTy->isBlockPointerType() || castTy->isReferenceType());

    StoreManager &storeMgr = StateMgr.getStoreManager();

    // Delegate to store manager to get the result of casting a region to a
    // different type.  If the MemRegion* returned is NULL, this expression
    // Evaluates to UnknownVal.
    R = storeMgr.castRegion(R, castTy);
    return R ? SVal(loc::MemRegionVal(R)) : UnknownVal();
  }

DispatchCast:
  // All other cases.
  return isa<Loc>(val) ? evalCastFromLoc(cast<Loc>(val), castTy)
                       : evalCastFromNonLoc(cast<NonLoc>(val), castTy);
}
