// SValuator.h - Construction of SVals from evaluating expressions -*- C++ -*---
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines SValuator, a class that defines the interface for
//  "symbolical evaluators" which construct an SVal from an expression.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_SVALUATOR
#define LLVM_CLANG_ANALYSIS_SVALUATOR

#include "clang/AST/Expr.h"
#include "clang/Checker/PathSensitive/SVals.h"

namespace clang {

class GRState;
class ValueManager;

class SValuator {
  friend class ValueManager;
protected:
  ValueManager &ValMgr;

public:
  // FIXME: Make these protected again one RegionStoreManager correctly
  // handles loads from differening bound value types.
  virtual SVal EvalCastNL(NonLoc val, QualType castTy) = 0;
  virtual SVal EvalCastL(Loc val, QualType castTy) = 0;

public:
  SValuator(ValueManager &valMgr) : ValMgr(valMgr) {}
  virtual ~SValuator() {}

  SVal EvalCast(SVal V, QualType castTy, QualType originalType);
  
  virtual SVal EvalMinus(NonLoc val) = 0;

  virtual SVal EvalComplement(NonLoc val) = 0;

  virtual SVal EvalBinOpNN(const GRState *state, BinaryOperator::Opcode Op,
                           NonLoc lhs, NonLoc rhs, QualType resultTy) = 0;

  virtual SVal EvalBinOpLL(BinaryOperator::Opcode Op, Loc lhs, Loc rhs,
                           QualType resultTy) = 0;

  virtual SVal EvalBinOpLN(const GRState *state, BinaryOperator::Opcode Op,
                           Loc lhs, NonLoc rhs, QualType resultTy) = 0;
  
  SVal EvalBinOp(const GRState *ST, BinaryOperator::Opcode Op,
                 SVal L, SVal R, QualType T);
  
  DefinedOrUnknownSVal EvalEQ(const GRState *ST, DefinedOrUnknownSVal L,
                              DefinedOrUnknownSVal R);
};

SValuator* CreateSimpleSValuator(ValueManager &valMgr);

} // end clang namespace
#endif
