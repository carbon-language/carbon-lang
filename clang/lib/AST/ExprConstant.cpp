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

using namespace clang;


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

bool Expr::tryEvaluate(APValue& Result, ASTContext &Ctx) const
{
  llvm::APSInt sInt(1);
  
  if (CalcFakeICEVal(this, sInt, Ctx)) {
    Result = APValue(sInt);
    return true;
  }
  
  return false;
}
