//== ValueManager.cpp - Aggregate manager of symbols and SVals --*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines ValueManager, a class that manages symbolic values
//  and SVals created for use by GRExprEngine and related classes.  It
//  wraps and owns SymbolManager, MemRegionManager, and BasicValueFactory.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/PathSensitive/ValueManager.h"

using namespace clang;
using namespace llvm;

//===----------------------------------------------------------------------===//
// Utility methods for constructing SVals.
//===----------------------------------------------------------------------===//

DefinedOrUnknownSVal ValueManager::makeZeroVal(QualType T) {
  if (Loc::IsLocType(T))
    return makeNull();

  if (T->isIntegerType())
    return makeIntVal(0, T);

  // FIXME: Handle floats.
  // FIXME: Handle structs.
  return UnknownVal();
}

//===----------------------------------------------------------------------===//
// Utility methods for constructing Non-Locs.
//===----------------------------------------------------------------------===//

NonLoc ValueManager::makeNonLoc(const SymExpr *lhs, BinaryOperator::Opcode op,
                                const APSInt& v, QualType T) {
  // The Environment ensures we always get a persistent APSInt in
  // BasicValueFactory, so we don't need to get the APSInt from
  // BasicValueFactory again.
  assert(!Loc::IsLocType(T));
  return nonloc::SymExprVal(SymMgr.getSymIntExpr(lhs, op, v, T));
}

NonLoc ValueManager::makeNonLoc(const SymExpr *lhs, BinaryOperator::Opcode op,
                                const SymExpr *rhs, QualType T) {
  assert(SymMgr.getType(lhs) == SymMgr.getType(rhs));
  assert(!Loc::IsLocType(T));
  return nonloc::SymExprVal(SymMgr.getSymSymExpr(lhs, op, rhs, T));
}


SVal ValueManager::convertToArrayIndex(SVal V) {
  if (V.isUnknownOrUndef())
    return V;

  // Common case: we have an appropriately sized integer.
  if (nonloc::ConcreteInt* CI = dyn_cast<nonloc::ConcreteInt>(&V)) {
    const llvm::APSInt& I = CI->getValue();
    if (I.getBitWidth() == ArrayIndexWidth && I.isSigned())
      return V;
  }

  return SVator->EvalCastNL(cast<NonLoc>(V), ArrayIndexTy);
}

DefinedOrUnknownSVal ValueManager::getRegionValueSymbolVal(const MemRegion* R,
                                                           QualType T) {

  if (T.isNull()) {
    const TypedRegion* TR = cast<TypedRegion>(R);
    T = TR->getValueType(SymMgr.getContext());
  }

  if (!SymbolManager::canSymbolicate(T))
    return UnknownVal();

  SymbolRef sym = SymMgr.getRegionValueSymbol(R, T);

  if (Loc::IsLocType(T))
    return loc::MemRegionVal(MemMgr.getSymbolicRegion(sym));

  return nonloc::SymbolVal(sym);
}

DefinedOrUnknownSVal ValueManager::getConjuredSymbolVal(const void *SymbolTag,
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

DefinedOrUnknownSVal ValueManager::getConjuredSymbolVal(const void *SymbolTag,
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


DefinedOrUnknownSVal
ValueManager::getDerivedRegionValueSymbolVal(SymbolRef parentSymbol,
                                             const TypedRegion *R) {
  QualType T = R->getValueType(R->getContext());

  if (!SymbolManager::canSymbolicate(T))
    return UnknownVal();

  SymbolRef sym = SymMgr.getDerivedSymbol(parentSymbol, R);

  if (Loc::IsLocType(T))
    return loc::MemRegionVal(MemMgr.getSymbolicRegion(sym));

  return nonloc::SymbolVal(sym);
}

DefinedSVal ValueManager::getFunctionPointer(const FunctionDecl* FD) {
  CodeTextRegion *R  = MemMgr.getCodeTextRegion(FD);
  return loc::MemRegionVal(R);
}
