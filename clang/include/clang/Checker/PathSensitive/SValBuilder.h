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

#ifndef LLVM_CLANG_ANALYSIS_SVALBUILDER
#define LLVM_CLANG_ANALYSIS_SVALBUILDER

#include "clang/AST/Expr.h"
#include "clang/Checker/PathSensitive/SVals.h"

namespace clang {

class GRState;
class ValueManager;

class SValBuilder {
  friend class ValueManager;
protected:
  ValueManager &ValMgr;

public:
  // FIXME: Make these protected again one RegionStoreManager correctly
  // handles loads from differening bound value types.
  virtual SVal evalCastNL(NonLoc val, QualType castTy) = 0;
  virtual SVal evalCastL(Loc val, QualType castTy) = 0;

public:
  SValBuilder(ValueManager &valMgr) : ValMgr(valMgr) {}
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
};

SValBuilder* createSimpleSValBuilder(ValueManager &valMgr);

} // end clang namespace
#endif
