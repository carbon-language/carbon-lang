//===- llvm/Analysis/Expressions.h - Expression Analysis Utils ---*- C++ -*--=//
//
// This file defines a package of expression analysis utilties:
//
// ClassifyExpression: Analyze an expression to determine the complexity of the
//   expression, and which other variables it depends on.  
// 
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_EXPRESSIONS_H
#define LLVM_ANALYSIS_EXPRESSIONS_H

#include <assert.h>
class Value;
class ConstPoolInt;
struct ExprAnalysisResult;

// ClassifyExpression: Analyze an expression to determine the complexity of the
// expression, and which other values it depends on.  
//
ExprAnalysisResult ClassifyExpression(Value *Expr);

// ExprAnalysisResult - Represent an expression of the form CONST*VAR+CONST
// or simpler.  The expression form that yields the least information about the
// expression is just the Linear form with no offset.
//
struct ExprAnalysisResult {
  enum ExpressionType {
    Constant,            // Expr is a simple constant, Offset is value
    Linear,              // Expr is linear expr, Value is Var+Offset
    ScaledLinear,        // Expr is scaled linear exp, Value is Scale*Var+Offset
  } ExprType;

  const ConstPoolInt *Offset;  // Offset of expr, or null if 0
  Value              *Var;     // Var referenced, if Linear or above (null if 0)
  const ConstPoolInt *Scale;   // Scale of var if ScaledLinear expr (null if 1)

  inline ExprAnalysisResult(const ConstPoolInt *CPV = 0) {
    Offset = CPV; Var = 0; Scale = 0;
    ExprType = Constant;
  }
  inline ExprAnalysisResult(Value *Val) {
    Var = Val; Offset = Scale = 0;
    ExprType = Var ? Linear : Constant;
  }
  inline ExprAnalysisResult(const ConstPoolInt *scale, Value *var, 
			    const ConstPoolInt *offset) {
    assert(!(Scale && !Var) && "Can't have scaled nonvariable!");
    Scale = scale; Var = var; Offset = offset;
    ExprType = Scale ? ScaledLinear : (Var ? Linear : Constant);
  }


private:
  friend ExprAnalysisResult ClassifyExpression(Value *);
  inline ExprAnalysisResult operator+(const ConstPoolInt *Offset);
  
};

#endif
