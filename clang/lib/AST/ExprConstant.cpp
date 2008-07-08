//===--- Expr.cpp - Expression Constant Evaluator -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Expr constant evaluator.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/APValue.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/STmtVisitor.h"
#include "llvm/Support/Compiler.h"

using namespace clang;

#define USE_NEW_EVALUATOR 0

static bool CalcFakeICEVal(const Expr* Expr,
                           llvm::APSInt& Result,
                           ASTContext& Context) {
  // Calculate the value of an expression that has a calculatable
  // value, but isn't an ICE. Currently, this only supports
  // a very narrow set of extensions, but it can be expanded if needed.
  if (const ParenExpr *PE = dyn_cast<ParenExpr>(Expr))
    return CalcFakeICEVal(PE->getSubExpr(), Result, Context);
  
  if (const CastExpr *CE = dyn_cast<CastExpr>(Expr)) {
    QualType CETy = CE->getType();
    if ((CETy->isIntegralType() && !CETy->isBooleanType()) ||
        CETy->isPointerType()) {
      if (CalcFakeICEVal(CE->getSubExpr(), Result, Context)) {
        Result.extOrTrunc(Context.getTypeSize(CETy));
        // FIXME: This assumes pointers are signed.
        Result.setIsSigned(CETy->isSignedIntegerType() ||
                           CETy->isPointerType());
        return true;
      }
    }
  }
  
  if (Expr->getType()->isIntegralType())
    return Expr->isIntegerConstantExpr(Result, Context);
  
  return false;
}

namespace {
class VISIBILITY_HIDDEN IntExprEvaluator
  : public StmtVisitor<IntExprEvaluator, APValue> {
  ASTContext &Ctx;

  IntExprEvaluator(ASTContext &ctx)
    : Ctx(ctx) {}

public:
  static bool Evaluate(const Expr* E, APValue& Result, ASTContext &Ctx) {
    Result = IntExprEvaluator(Ctx).Visit(const_cast<Expr*>(E));
    return Result.isSInt();
  }
    
  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//
  APValue VisitStmt(Stmt *S) {
    // FIXME: Remove this when we support more expressions.
    printf("Unhandled statement\n");
    S->dump();  
    return APValue();
  }
  
  APValue VisitParenExpr(ParenExpr *PE) { return Visit(PE->getSubExpr()); }

};    
}
  
bool Expr::tryEvaluate(APValue& Result, ASTContext &Ctx) const
{
  llvm::APSInt sInt(1);
  
#if USE_NEW_EVALUATOR
  if (getType()->isIntegerType())
    return IntExprEvaluator::Evaluate(this, Result, Ctx);
  else
    return false;
    
#else
  if (CalcFakeICEVal(this, sInt, Ctx)) {
    Result = APValue(sInt);
    return true;
  }
#endif
  
  return false;
}
