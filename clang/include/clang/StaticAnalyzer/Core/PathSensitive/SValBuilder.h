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
#include "clang/StaticAnalyzer/Core/PathSensitive/SVals.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/BasicValueFactory.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/MemRegion.h"

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
  // FIXME: Make these protected again once RegionStoreManager correctly
  // handles loads from different bound value types.
  virtual SVal evalCastFromNonLoc(NonLoc val, QualType castTy) = 0;
  virtual SVal evalCastFromLoc(Loc val, QualType castTy) = 0;

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

  SVal evalCast(SVal val, QualType castTy, QualType originalType);
  
  virtual SVal evalMinus(NonLoc val) = 0;

  virtual SVal evalComplement(NonLoc val) = 0;

  virtual SVal evalBinOpNN(const GRState *state, BinaryOperator::Opcode op,
                           NonLoc lhs, NonLoc rhs, QualType resultTy) = 0;

  virtual SVal evalBinOpLL(const GRState *state, BinaryOperator::Opcode op,
                           Loc lhs, Loc rhs, QualType resultTy) = 0;

  virtual SVal evalBinOpLN(const GRState *state, BinaryOperator::Opcode op,
                           Loc lhs, NonLoc rhs, QualType resultTy) = 0;

  /// getKnownValue - evaluates a given SVal. If the SVal has only one possible
  ///  (integer) value, that value is returned. Otherwise, returns NULL.
  virtual const llvm::APSInt *getKnownValue(const GRState *state, SVal val) = 0;
  
  SVal evalBinOp(const GRState *state, BinaryOperator::Opcode op,
                 SVal lhs, SVal rhs, QualType type);
  
  DefinedOrUnknownSVal evalEQ(const GRState *state, DefinedOrUnknownSVal lhs,
                              DefinedOrUnknownSVal rhs);

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

  const SymbolConjured* getConjuredSymbol(const Stmt *stmt, QualType type,
                                          unsigned visitCount,
                                          const void *symbolTag = 0) {
    return SymMgr.getConjuredSymbol(stmt, type, visitCount, symbolTag);
  }

  const SymbolConjured* getConjuredSymbol(const Expr *expr, unsigned visitCount,
                                          const void *symbolTag = 0) {
    return SymMgr.getConjuredSymbol(expr, visitCount, symbolTag);
  }

  /// makeZeroVal - Construct an SVal representing '0' for the specified type.
  DefinedOrUnknownSVal makeZeroVal(QualType type);

  /// getRegionValueSymbolVal - make a unique symbol for value of region.
  DefinedOrUnknownSVal getRegionValueSymbolVal(const TypedValueRegion *region);

  DefinedOrUnknownSVal getConjuredSymbolVal(const void *symbolTag,
                                            const Expr *expr, unsigned count);
  DefinedOrUnknownSVal getConjuredSymbolVal(const void *symbolTag,
                                            const Expr *expr, QualType type,
                                            unsigned count);

  DefinedOrUnknownSVal getDerivedRegionValueSymbolVal(
      SymbolRef parentSymbol, const TypedValueRegion *region);

  DefinedSVal getMetadataSymbolVal(
      const void *symbolTag, const MemRegion *region,
      const Expr *expr, QualType type, unsigned count);

  DefinedSVal getFunctionPointer(const FunctionDecl *func);
  
  DefinedSVal getBlockPointer(const BlockDecl *block, CanQualType locTy,
                              const LocationContext *locContext);

  NonLoc makeCompoundVal(QualType type, llvm::ImmutableList<SVal> vals) {
    return nonloc::CompoundVal(BasicVals.getCompoundValData(type, vals));
  }

  NonLoc makeLazyCompoundVal(const StoreRef &store, 
                             const TypedValueRegion *region) {
    return nonloc::LazyCompoundVal(
        BasicVals.getLazyCompoundValData(store, region));
  }

  NonLoc makeZeroArrayIndex() {
    return nonloc::ConcreteInt(BasicVals.getValue(0, ArrayIndexTy));
  }

  NonLoc makeArrayIndex(uint64_t idx) {
    return nonloc::ConcreteInt(BasicVals.getValue(idx, ArrayIndexTy));
  }

  SVal convertToArrayIndex(SVal val);

  nonloc::ConcreteInt makeIntVal(const IntegerLiteral* integer) {
    return nonloc::ConcreteInt(
        BasicVals.getValue(integer->getValue(),
                     integer->getType()->isUnsignedIntegerOrEnumerationType()));
  }

  nonloc::ConcreteInt makeBoolVal(const CXXBoolLiteralExpr *boolean) {
    return makeTruthVal(boolean->getValue());
  }

  nonloc::ConcreteInt makeIntVal(const llvm::APSInt& integer) {
    return nonloc::ConcreteInt(BasicVals.getValue(integer));
  }

  loc::ConcreteInt makeIntLocVal(const llvm::APSInt &integer) {
    return loc::ConcreteInt(BasicVals.getValue(integer));
  }

  NonLoc makeIntVal(const llvm::APInt& integer, bool isUnsigned) {
    return nonloc::ConcreteInt(BasicVals.getValue(integer, isUnsigned));
  }

  DefinedSVal makeIntVal(uint64_t integer, QualType type) {
    if (Loc::isLocType(type))
      return loc::ConcreteInt(BasicVals.getValue(integer, type));

    return nonloc::ConcreteInt(BasicVals.getValue(integer, type));
  }

  NonLoc makeIntVal(uint64_t integer, bool isUnsigned) {
    return nonloc::ConcreteInt(BasicVals.getIntValue(integer, isUnsigned));
  }

  NonLoc makeIntValWithPtrWidth(uint64_t integer, bool isUnsigned) {
    return nonloc::ConcreteInt(
        BasicVals.getIntWithPtrWidth(integer, isUnsigned));
  }

  NonLoc makeIntVal(uint64_t integer, unsigned bitWidth, bool isUnsigned) {
    return nonloc::ConcreteInt(
        BasicVals.getValue(integer, bitWidth, isUnsigned));
  }

  NonLoc makeLocAsInteger(Loc loc, unsigned bits) {
    return nonloc::LocAsInteger(BasicVals.getPersistentSValWithData(loc, bits));
  }

  NonLoc makeNonLoc(const SymExpr *lhs, BinaryOperator::Opcode op,
                    const llvm::APSInt& rhs, QualType type);

  NonLoc makeNonLoc(const SymExpr *lhs, BinaryOperator::Opcode op,
                    const SymExpr *rhs, QualType type);

  nonloc::ConcreteInt makeTruthVal(bool b, QualType type) {
    return nonloc::ConcreteInt(BasicVals.getTruthValue(b, type));
  }

  nonloc::ConcreteInt makeTruthVal(bool b) {
    return nonloc::ConcreteInt(BasicVals.getTruthValue(b));
  }

  Loc makeNull() {
    return loc::ConcreteInt(BasicVals.getZeroWithPtrWidth());
  }

  Loc makeLoc(SymbolRef sym) {
    return loc::MemRegionVal(MemMgr.getSymbolicRegion(sym));
  }

  Loc makeLoc(const MemRegion* region) {
    return loc::MemRegionVal(region);
  }

  Loc makeLoc(const AddrLabelExpr *expr) {
    return loc::GotoLabel(expr->getLabel());
  }

  Loc makeLoc(const llvm::APSInt& integer) {
    return loc::ConcreteInt(BasicVals.getValue(integer));
  }

};

SValBuilder* createSimpleSValBuilder(llvm::BumpPtrAllocator &alloc,
                                     ASTContext &context,
                                     GRStateManager &stateMgr);

} // end GR namespace

} // end clang namespace

#endif
