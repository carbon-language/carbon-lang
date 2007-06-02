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
//                         LValue Expression Emission
//===--------------------------------------------------------------------===//

LValue CodeGenFunction::EmitLValue(const Expr *E) {
  switch (E->getStmtClass()) {
  default:
    printf("Unimplemented lvalue expr!\n");
    E->dump();
    return LValue::getAddr(UndefValue::get(
                              llvm::PointerType::get(llvm::Type::Int32Ty)));

  case Expr::DeclRefExprClass: return EmitDeclRefLValue(cast<DeclRefExpr>(E));
  }
}


LValue CodeGenFunction::EmitDeclRefLValue(const DeclRefExpr *E) {
  const Decl *D = E->getDecl();
  if (isa<BlockVarDecl>(D)) {
    Value *V = LocalDeclMap[D];
    assert(V && "BlockVarDecl not entered in LocalDeclMap?");
    return LValue::getAddr(V);
  }
  assert(0 && "Unimp declref");
}

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
    
  // l-values.
  case Expr::DeclRefExprClass: {
    // FIXME: EnumConstantDecl's are not lvalues.
    LValue LV = EmitLValue(E);
    // FIXME: this is silly.
    assert(!LV.isBitfield());
    return ExprResult::get(Builder.CreateLoad(LV.getAddress(), "tmp"));
  }
    
  // Leaf expressions.
  case Expr::IntegerLiteralClass:
    return EmitIntegerLiteral(cast<IntegerLiteral>(E)); 
    
  // Operators.  
  case Expr::ParenExprClass:
    return EmitExpr(cast<ParenExpr>(E)->getSubExpr());
  case Expr::BinaryOperatorClass:
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