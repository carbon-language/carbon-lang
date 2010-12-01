//== ValueManager.h - Aggregate manager of symbols and SVals ----*- C++ -*--==//
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

#ifndef LLVM_CLANG_ANALYSIS_AGGREGATE_VALUE_MANAGER_H
#define LLVM_CLANG_ANALYSIS_AGGREGATE_VALUE_MANAGER_H

#include "llvm/ADT/OwningPtr.h"
#include "clang/Checker/PathSensitive/MemRegion.h"
#include "clang/Checker/PathSensitive/SVals.h"
#include "clang/Checker/PathSensitive/BasicValueFactory.h"
#include "clang/Checker/PathSensitive/SymbolManager.h"
#include "clang/Checker/PathSensitive/SValBuilder.h"
#include "clang/AST/ExprCXX.h"

namespace llvm { class BumpPtrAllocator; }

namespace clang {

class GRStateManager;

class ValueManager {

  ASTContext &Context;
  BasicValueFactory BasicVals;

  /// SymMgr - Object that manages the symbol information.
  SymbolManager SymMgr;

  /// svalBuilder - SValBuilder object that creates SVals from expressions.
  llvm::OwningPtr<SValBuilder> svalBuilder;

  MemRegionManager MemMgr;

  GRStateManager &StateMgr;

  const QualType ArrayIndexTy;
  const unsigned ArrayIndexWidth;

public:
  ValueManager(llvm::BumpPtrAllocator &alloc, ASTContext &context,
               GRStateManager &stateMgr)
               : Context(context), BasicVals(context, alloc),
                 SymMgr(context, BasicVals, alloc),
                 MemMgr(context, alloc), StateMgr(stateMgr),
                 ArrayIndexTy(context.IntTy),
                 ArrayIndexWidth(context.getTypeSize(ArrayIndexTy)) {
    // FIXME: Generalize later.
    svalBuilder.reset(clang::createSimpleSValBuilder(*this));
  }

  // Accessors to submanagers.

  ASTContext &getContext() { return Context; }
  const ASTContext &getContext() const { return Context; }

  GRStateManager &getStateManager() { return StateMgr; }

  BasicValueFactory &getBasicValueFactory() { return BasicVals; }
  const BasicValueFactory &getBasicValueFactory() const { return BasicVals; }

  SymbolManager &getSymbolManager() { return SymMgr; }
  const SymbolManager &getSymbolManager() const { return SymMgr; }

  SValBuilder &getSValBuilder() { return *svalBuilder.get(); }

  MemRegionManager &getRegionManager() { return MemMgr; }
  const MemRegionManager &getRegionManager() const { return MemMgr; }

  // Forwarding methods to SymbolManager.

  const SymbolConjured* getConjuredSymbol(const Stmt* E, QualType T,
                                          unsigned VisitCount,
                                          const void* SymbolTag = 0) {
    return SymMgr.getConjuredSymbol(E, T, VisitCount, SymbolTag);
  }

  const SymbolConjured* getConjuredSymbol(const Expr* E, unsigned VisitCount,
                                          const void* SymbolTag = 0) {
    return SymMgr.getConjuredSymbol(E, VisitCount, SymbolTag);
  }

  /// makeZeroVal - Construct an SVal representing '0' for the specified type.
  DefinedOrUnknownSVal makeZeroVal(QualType T);

  /// getRegionValueSymbolVal - make a unique symbol for value of R.
  DefinedOrUnknownSVal getRegionValueSymbolVal(const TypedRegion *R);

  DefinedOrUnknownSVal getConjuredSymbolVal(const void *SymbolTag,
                                            const Expr *E, unsigned Count);
  DefinedOrUnknownSVal getConjuredSymbolVal(const void *SymbolTag,
                                            const Expr *E, QualType T,
                                            unsigned Count);

  DefinedOrUnknownSVal getDerivedRegionValueSymbolVal(SymbolRef parentSymbol,
                                                      const TypedRegion *R);

  DefinedSVal getMetadataSymbolVal(const void *SymbolTag, const MemRegion *MR,
                                   const Expr *E, QualType T, unsigned Count);

  DefinedSVal getFunctionPointer(const FunctionDecl *FD);
  
  DefinedSVal getBlockPointer(const BlockDecl *BD, CanQualType locTy,
                              const LocationContext *LC);

  NonLoc makeCompoundVal(QualType T, llvm::ImmutableList<SVal> Vals) {
    return nonloc::CompoundVal(BasicVals.getCompoundValData(T, Vals));
  }

  NonLoc makeLazyCompoundVal(const void *store, const TypedRegion *R) {
    return nonloc::LazyCompoundVal(BasicVals.getLazyCompoundValData(store, R));
  }

  NonLoc makeZeroArrayIndex() {
    return nonloc::ConcreteInt(BasicVals.getValue(0, ArrayIndexTy));
  }

  NonLoc makeArrayIndex(uint64_t idx) {
    return nonloc::ConcreteInt(BasicVals.getValue(idx, ArrayIndexTy));
  }

  SVal convertToArrayIndex(SVal V);

  nonloc::ConcreteInt makeIntVal(const IntegerLiteral* I) {
    return nonloc::ConcreteInt(BasicVals.getValue(I->getValue(),
                                        I->getType()->isUnsignedIntegerType()));
  }

  nonloc::ConcreteInt makeIntVal(const CXXBoolLiteralExpr *E) {
    return E->getValue() ? nonloc::ConcreteInt(BasicVals.getValue(1, 1, true))
                         : nonloc::ConcreteInt(BasicVals.getValue(0, 1, true));
  }

  nonloc::ConcreteInt makeIntVal(const llvm::APSInt& V) {
    return nonloc::ConcreteInt(BasicVals.getValue(V));
  }

  loc::ConcreteInt makeIntLocVal(const llvm::APSInt &v) {
    return loc::ConcreteInt(BasicVals.getValue(v));
  }

  NonLoc makeIntVal(const llvm::APInt& V, bool isUnsigned) {
    return nonloc::ConcreteInt(BasicVals.getValue(V, isUnsigned));
  }

  DefinedSVal makeIntVal(uint64_t X, QualType T) {
    if (Loc::IsLocType(T))
      return loc::ConcreteInt(BasicVals.getValue(X, T));

    return nonloc::ConcreteInt(BasicVals.getValue(X, T));
  }

  NonLoc makeIntVal(uint64_t X, bool isUnsigned) {
    return nonloc::ConcreteInt(BasicVals.getIntValue(X, isUnsigned));
  }

  NonLoc makeIntValWithPtrWidth(uint64_t X, bool isUnsigned) {
    return nonloc::ConcreteInt(BasicVals.getIntWithPtrWidth(X, isUnsigned));
  }

  NonLoc makeIntVal(uint64_t X, unsigned BitWidth, bool isUnsigned) {
    return nonloc::ConcreteInt(BasicVals.getValue(X, BitWidth, isUnsigned));
  }

  NonLoc makeLocAsInteger(Loc V, unsigned Bits) {
    return nonloc::LocAsInteger(BasicVals.getPersistentSValWithData(V, Bits));
  }

  NonLoc makeNonLoc(const SymExpr *lhs, BinaryOperator::Opcode op,
                    const llvm::APSInt& rhs, QualType T);

  NonLoc makeNonLoc(const SymExpr *lhs, BinaryOperator::Opcode op,
                    const SymExpr *rhs, QualType T);

  NonLoc makeTruthVal(bool b, QualType T) {
    return nonloc::ConcreteInt(BasicVals.getTruthValue(b, T));
  }

  NonLoc makeTruthVal(bool b) {
    return nonloc::ConcreteInt(BasicVals.getTruthValue(b));
  }

  Loc makeNull() {
    return loc::ConcreteInt(BasicVals.getZeroWithPtrWidth());
  }

  Loc makeLoc(SymbolRef Sym) {
    return loc::MemRegionVal(MemMgr.getSymbolicRegion(Sym));
  }

  Loc makeLoc(const MemRegion* R) {
    return loc::MemRegionVal(R);
  }

  Loc makeLoc(const AddrLabelExpr* E) {
    return loc::GotoLabel(E->getLabel());
  }

  Loc makeLoc(const llvm::APSInt& V) {
    return loc::ConcreteInt(BasicVals.getValue(V));
  }
};
} // end clang namespace
#endif

