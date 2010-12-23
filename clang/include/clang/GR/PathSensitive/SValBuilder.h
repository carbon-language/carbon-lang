// SValBuilder.h - Construction of SVals from evaluating expressions -*- C++ -*-
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines SValBuilder, a class that defines the interface for
//  "symbolical evaluators" which construct an SVal from an expression.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_GR_SVALBUILDER
#define LLVM_CLANG_GR_SVALBUILDER

#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/GR/PathSensitive/SVals.h"
#include "clang/GR/PathSensitive/BasicValueFactory.h"
#include "clang/GR/PathSensitive/MemRegion.h"

namespace clang {

namespace ento {

class GRState;

class SValBuilder {
protected:
  ASTContext &Context;
  
  /// Manager of APSInt values.
  BasicValueFactory BasicVals;

  /// Manages the creation of symbols.
  SymbolManager SymMgr;

  /// Manages the creation of memory regions.
  MemRegionManager MemMgr;

  GRStateManager &StateMgr;

  /// The scalar type to use for array indices.
  const QualType ArrayIndexTy;
  
  /// The width of the scalar type used for array indices.
  const unsigned ArrayIndexWidth;

public:
  // FIXME: Make these protected again one RegionStoreManager correctly
  // handles loads from differening bound value types.
  virtual SVal evalCastNL(NonLoc val, QualType castTy) = 0;
  virtual SVal evalCastL(Loc val, QualType castTy) = 0;

public:
  SValBuilder(llvm::BumpPtrAllocator &alloc, ASTContext &context,
              GRStateManager &stateMgr)
    : Context(context), BasicVals(context, alloc),
      SymMgr(context, BasicVals, alloc),
      MemMgr(context, alloc),
      StateMgr(stateMgr),
      ArrayIndexTy(context.IntTy),
      ArrayIndexWidth(context.getTypeSize(ArrayIndexTy)) {}

  virtual ~SValBuilder() {}

  SVal evalCast(SVal V, QualType castTy, QualType originalType);
  
  virtual SVal evalMinus(NonLoc val) = 0;

  virtual SVal evalComplement(NonLoc val) = 0;

  virtual SVal evalBinOpNN(const GRState *state, BinaryOperator::Opcode Op,
                           NonLoc lhs, NonLoc rhs, QualType resultTy) = 0;

  virtual SVal evalBinOpLL(const GRState *state, BinaryOperator::Opcode Op,
                           Loc lhs, Loc rhs, QualType resultTy) = 0;

  virtual SVal evalBinOpLN(const GRState *state, BinaryOperator::Opcode Op,
                           Loc lhs, NonLoc rhs, QualType resultTy) = 0;

  /// getKnownValue - evaluates a given SVal. If the SVal has only one possible
  ///  (integer) value, that value is returned. Otherwise, returns NULL.
  virtual const llvm::APSInt *getKnownValue(const GRState *state, SVal V) = 0;
  
  SVal evalBinOp(const GRState *ST, BinaryOperator::Opcode Op,
                 SVal L, SVal R, QualType T);
  
  DefinedOrUnknownSVal evalEQ(const GRState *ST, DefinedOrUnknownSVal L,
                              DefinedOrUnknownSVal R);

  ASTContext &getContext() { return Context; }
  const ASTContext &getContext() const { return Context; }

  GRStateManager &getStateManager() { return StateMgr; }
  
  QualType getConditionType() const {
    return  getContext().IntTy;
  }
  
  QualType getArrayIndexType() const {
    return ArrayIndexTy;
  }

  BasicValueFactory &getBasicValueFactory() { return BasicVals; }
  const BasicValueFactory &getBasicValueFactory() const { return BasicVals; }

  SymbolManager &getSymbolManager() { return SymMgr; }
  const SymbolManager &getSymbolManager() const { return SymMgr; }

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

SValBuilder* createSimpleSValBuilder(llvm::BumpPtrAllocator &alloc,
                                     ASTContext &context,
                                     GRStateManager &stateMgr);

} // end GR namespace

} // end clang namespace

#endif
