//===- llvm/Analysis/Expressions.h - Expression Analysis Utils --*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines a package of expression analysis utilties:
//
// ClassifyExpr: Analyze an expression to determine the complexity of the
// expression, and which other variables it depends on.
// 
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_EXPRESSIONS_H
#define LLVM_ANALYSIS_EXPRESSIONS_H

namespace llvm {

class Type;
class Value;
class ConstantInt;

struct ExprType;

/// ClassifyExpr - Analyze an expression to determine the complexity of the
/// expression, and which other values it depends on.
///
ExprType ClassifyExpr(Value *Expr);

/// ExprType Class - Represent an expression of the form CONST*VAR+CONST
/// or simpler.  The expression form that yields the least information about the
/// expression is just the Linear form with no offset.
///
struct ExprType {
  enum ExpressionType {
    Constant,            // Expr is a simple constant, Offset is value
    Linear,              // Expr is linear expr, Value is Var+Offset
    ScaledLinear,        // Expr is scaled linear exp, Value is Scale*Var+Offset
  } ExprTy;

  const ConstantInt *Offset;  // Offset of expr, or null if 0
  Value             *Var;     // Var referenced, if Linear or above (null if 0)
  const ConstantInt *Scale;   // Scale of var if ScaledLinear expr (null if 1)

  inline ExprType(const ConstantInt *CPV = 0) {
    Offset = CPV; Var = 0; Scale = 0;
    ExprTy = Constant;
  }
  ExprType(Value *Val);        // Create a linear or constant expression
  ExprType(const ConstantInt *scale, Value *var, const ConstantInt *offset);

  /// If this expression has an intrinsic type, return it.  If it is zero,
  /// return the specified type.
  ///
  const Type *getExprType(const Type *Default) const;
};

} // End llvm namespace

#endif
