//===--- CGExpr.cpp - Emit LLVM Code from Expressions ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Expr nodes as LLVM code.
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "clang/AST/AST.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
using namespace llvm;
using namespace clang;
using namespace CodeGen;


//===--------------------------------------------------------------------===//
//                             Expression Emission
//===--------------------------------------------------------------------===//

ExprResult CodeGenFunction::EmitExpr(const Expr *E) {
  assert(E && "Null expression?");
  
  switch (E->getStmtClass()) {
  default:
    printf("Unimplemented expr!\n");
    E->dump();
    return ExprResult::get(UndefValue::get(llvm::Type::Int32Ty));
  case Stmt::ParenExprClass:
    return EmitExpr(cast<ParenExpr>(E)->getSubExpr());
  case Stmt::IntegerLiteralClass:
    return EmitIntegerLiteral(cast<IntegerLiteral>(E)); 
    
  case Stmt::BinaryOperatorClass:
    return EmitBinaryOperator(cast<BinaryOperator>(E));
  }
  
}

ExprResult CodeGenFunction::EmitIntegerLiteral(const IntegerLiteral *E) {
  return ExprResult::get(ConstantInt::get(E->getValue()));
}


//===--------------------------------------------------------------------===//
//                         Binary Operator Emission
//===--------------------------------------------------------------------===//

// FIXME describe.
void CodeGenFunction::EmitUsualArithmeticConversions(const BinaryOperator *E,
                                                     ExprResult &LHS, 
                                                     ExprResult &RHS) {
  // FIXME: implement right.
  LHS = EmitExpr(E->getLHS());
  RHS = EmitExpr(E->getRHS());
}


ExprResult CodeGenFunction::EmitBinaryOperator(const BinaryOperator *E) {
  switch (E->getOpcode()) {
  default:
    printf("Unimplemented expr!\n");
    E->dump();
    return ExprResult::get(UndefValue::get(llvm::Type::Int32Ty));
  case BinaryOperator::Add: return EmitBinaryAdd(E);
  }
}


ExprResult CodeGenFunction::EmitBinaryAdd(const BinaryOperator *E) {
  ExprResult LHS, RHS;
  
  EmitUsualArithmeticConversions(E, LHS, RHS);

  
  return ExprResult::get(Builder.CreateAdd(LHS.getVal(), RHS.getVal(), "tmp"));
}