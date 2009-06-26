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
#include "clang/Analysis/PathSensitive/SVals.h"

namespace clang {

class GRState;
class ValueManager;

class SValuator {
protected:
  ValueManager &ValMgr;
  
public:
  SValuator(ValueManager &valMgr) : ValMgr(valMgr) {}
  virtual ~SValuator() {}
  
  virtual SVal EvalCast(NonLoc val, QualType castTy) = 0;  

  virtual SVal EvalCast(Loc val, QualType castTy) = 0;
  
  virtual SVal EvalMinus(NonLoc val) = 0;
  
  virtual SVal EvalComplement(NonLoc val) = 0;

  virtual SVal EvalBinOpNN(BinaryOperator::Opcode Op, NonLoc lhs,
                           NonLoc rhs, QualType resultTy) = 0;

  virtual SVal EvalBinOpLL(BinaryOperator::Opcode Op, Loc lhs, Loc rhs,
                           QualType resultTy) = 0;

  virtual SVal EvalBinOpLN(const GRState *state, BinaryOperator::Opcode Op,
                           Loc lhs, NonLoc rhs, QualType resultTy) = 0;  
};
  
SValuator* CreateSimpleSValuator(ValueManager &valMgr);
  
} // end clang namespace
#endif
