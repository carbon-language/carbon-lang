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

SVal ValueManager::makeZeroVal(QualType T) {
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


SVal ValueManager::getRegionValueSymbolVal(const MemRegion* R, QualType T) {
  SymbolRef sym = SymMgr.getRegionValueSymbol(R, T);
                                
  if (const TypedRegion* TR = dyn_cast<TypedRegion>(R)) {
    if (T.isNull())
      T = TR->getValueType(SymMgr.getContext());

    // If T is of function pointer type, create a CodeTextRegion wrapping a
    // symbol.
    if (T->isFunctionPointerType()) {
      return loc::MemRegionVal(MemMgr.getCodeTextRegion(sym, T));
    }
    
    if (Loc::IsLocType(T))
      return loc::MemRegionVal(MemMgr.getSymbolicRegion(sym));
  
    // Only handle integers for now.
    if (T->isIntegerType() && T->isScalarType())
      return nonloc::SymbolVal(sym);
  }

  return UnknownVal();
}

SVal ValueManager::getConjuredSymbolVal(const Expr* E, unsigned Count) {
  QualType T = E->getType();
  SymbolRef sym = SymMgr.getConjuredSymbol(E, Count);

  // If T is of function pointer type, create a CodeTextRegion wrapping a
  // symbol.
  if (T->isFunctionPointerType()) {
    return loc::MemRegionVal(MemMgr.getCodeTextRegion(sym, T));
  }

  if (Loc::IsLocType(T))
    return loc::MemRegionVal(MemMgr.getSymbolicRegion(sym));

  if (T->isIntegerType() && T->isScalarType())
    return nonloc::SymbolVal(sym);

  return UnknownVal();
}

SVal ValueManager::getConjuredSymbolVal(const Expr* E, QualType T,
                                        unsigned Count) {

  SymbolRef sym = SymMgr.getConjuredSymbol(E, T, Count);

  // If T is of function pointer type, create a CodeTextRegion wrapping a
  // symbol.
  if (T->isFunctionPointerType()) {
    return loc::MemRegionVal(MemMgr.getCodeTextRegion(sym, T));
  }

  if (Loc::IsLocType(T))
    return loc::MemRegionVal(MemMgr.getSymbolicRegion(sym));

  if (T->isIntegerType() && T->isScalarType())
    return nonloc::SymbolVal(sym);

  return UnknownVal();
}

SVal ValueManager::getFunctionPointer(const FunctionDecl* FD) {
  CodeTextRegion* R 
    = MemMgr.getCodeTextRegion(FD, Context.getPointerType(FD->getType()));
  return loc::MemRegionVal(R);
}
