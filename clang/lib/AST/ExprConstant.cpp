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
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/TargetInfo.h"
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
class VISIBILITY_HIDDEN PointerExprEvaluator
  : public StmtVisitor<PointerExprEvaluator, APValue> {
  ASTContext &Ctx;

  PointerExprEvaluator(ASTContext &ctx)
    : Ctx(ctx) {}

public:
  static bool Evaluate(const Expr* E, APValue& Result, ASTContext &Ctx) {
    if (!E->getType()->isPointerType())
      return false;
    Result = PointerExprEvaluator(Ctx).Visit(const_cast<Expr*>(E));
    return Result.isLValue();
  }
    
  APValue VisitStmt(Stmt *S) {
    // FIXME: Remove this when we support more expressions.
    printf("Unhandled pointer statement\n");
    S->dump();  
    return APValue();
  }

  APValue VisitParenExpr(ParenExpr *E) { return Visit(E->getSubExpr()); }

  APValue VisitBinaryOperator(const BinaryOperator *E);
  APValue VisitCastExpr(const CastExpr* E);

};

class VISIBILITY_HIDDEN IntExprEvaluator
  : public StmtVisitor<IntExprEvaluator, APValue> {
  ASTContext &Ctx;

  IntExprEvaluator(ASTContext &ctx)
    : Ctx(ctx) {}

public:
  static bool Evaluate(const Expr* E, llvm::APSInt& Result, ASTContext &Ctx) {
    APValue Value = IntExprEvaluator(Ctx).Visit(const_cast<Expr*>(E));
    if (!Value.isSInt())
      return false;
    
    Result = Value.getSInt();
    return true;
  }
    
  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//
  APValue VisitStmt(Stmt *S) {
    // FIXME: Remove this when we support more expressions.
    printf("unhandled int expression");
    S->dump();  
    return APValue();
  }
  
  APValue VisitParenExpr(ParenExpr *E) { return Visit(E->getSubExpr()); }

  APValue VisitBinaryOperator(const BinaryOperator *E);
  APValue VisitUnaryOperator(const UnaryOperator *E);

  APValue HandleCast(const Expr* SubExpr, QualType DestType);
  APValue VisitCastExpr(const CastExpr* E) {
    return HandleCast(E->getSubExpr(), E->getType());
  }
  APValue VisitImplicitCastExpr(const ImplicitCastExpr* E) {
    return HandleCast(E->getSubExpr(), E->getType());
  }
  APValue VisitSizeOfAlignOfTypeExpr(const SizeOfAlignOfTypeExpr *E);
 
  APValue VisitIntegerLiteral(const IntegerLiteral *E) {
    llvm::APSInt Result(Ctx.getTypeSize(E->getType()));

    Result = E->getValue();
    return APValue(Result);
  }

};
  
APValue PointerExprEvaluator::VisitBinaryOperator(const BinaryOperator *E)
{
  if (E->getOpcode() != BinaryOperator::Add &&
      E->getOpcode() != BinaryOperator::Sub)
    return APValue();
  
  const Expr *PExp = E->getLHS();
  const Expr *IExp = E->getRHS();
  if (IExp->getType()->isPointerType())
  std::swap(PExp, IExp);
  
  APValue ResultLValue;
  if (!PointerExprEvaluator::Evaluate(PExp, ResultLValue, Ctx))
    return APValue();
  llvm::APSInt AdditionalOffset(32);
  if (!IntExprEvaluator::Evaluate(IExp, AdditionalOffset, Ctx))
    return APValue();

  uint64_t Offset = ResultLValue.getLValueOffset();
  if (E->getOpcode() == BinaryOperator::Add)
    Offset += AdditionalOffset.getZExtValue();
  else
    Offset -= AdditionalOffset.getZExtValue();
    
  return APValue(ResultLValue.getLValueBase(), Offset);
}
  

APValue PointerExprEvaluator::VisitCastExpr(const CastExpr* E)
{
  const Expr* SubExpr = E->getSubExpr();

   // Check for pointer->pointer cast
  if (SubExpr->getType()->isPointerType()) {
    APValue Result;
    if (PointerExprEvaluator::Evaluate(SubExpr, Result, Ctx))
      return Result;
    else
      return APValue();
  }
  
  if (SubExpr->getType()->isArithmeticType()) {
    llvm::APSInt Result(32);
    if (IntExprEvaluator::Evaluate(SubExpr, Result, Ctx)) {
      Result.extOrTrunc(static_cast<uint32_t>(Ctx.getTypeSize(E->getType())));
      return APValue(0, Result.getZExtValue());
    }
  }
  
  assert(0 && "Unhandled cast");
  return APValue();
}  

APValue IntExprEvaluator::VisitBinaryOperator(const BinaryOperator *E) {
  // The LHS of a constant expr is always evaluated and needed.
  llvm::APSInt Result(32);
  if (!Evaluate(E->getLHS(), Result, Ctx))
    return APValue(); 

  llvm::APSInt RHS(32);
  if (!Evaluate(E->getRHS(), RHS, Ctx))
    return APValue();
  
  switch (E->getOpcode()) {
  default:
    return APValue();
  case BinaryOperator::Mul:
    Result *= RHS;
    break;
  case BinaryOperator::Div:
    if (RHS == 0)
      return APValue();
   Result /= RHS;
     break;
  case BinaryOperator::Rem:
    if (RHS == 0)
      return APValue();
    Result %= RHS;
    break;
  case BinaryOperator::Add: Result += RHS; break;
  case BinaryOperator::Sub: Result -= RHS; break;
  case BinaryOperator::Shl:
    Result <<= 
      static_cast<uint32_t>(RHS.getLimitedValue(Result.getBitWidth()-1));
    break;
  case BinaryOperator::Shr:
    Result >>= 
      static_cast<uint32_t>(RHS.getLimitedValue(Result.getBitWidth()-1));
    break;
  case BinaryOperator::LT:  Result = Result < RHS; break;
  case BinaryOperator::GT:  Result = Result > RHS; break;
  case BinaryOperator::LE:  Result = Result <= RHS; break;
  case BinaryOperator::GE:  Result = Result >= RHS; break;
  case BinaryOperator::EQ:  Result = Result == RHS; break;
  case BinaryOperator::NE:  Result = Result != RHS; break;
  case BinaryOperator::And: Result &= RHS; break;
  case BinaryOperator::Xor: Result ^= RHS; break;
  case BinaryOperator::Or:  Result |= RHS; break;
    
  case BinaryOperator::Comma:
    // C99 6.6p3: "shall not contain assignment, ..., or comma operators,
    // *except* when they are contained within a subexpression that is not
    // evaluated".  Note that Assignment can never happen due to constraints
    // on the LHS subexpr, so we don't need to check it here.
    // FIXME: Need to come up with an efficient way to deal with the C99
    // rules on evaluation while still evaluating this.  Maybe a
    // "evaluated comma" out parameter?
    return APValue();
  }

  Result.setIsUnsigned(E->getType()->isUnsignedIntegerType());

  return APValue(Result);
}

APValue IntExprEvaluator::VisitUnaryOperator(const UnaryOperator *E) {
  llvm::APSInt Result(32);
  
  if (E->isOffsetOfOp())
    Result = E->evaluateOffsetOf(Ctx);
  else if (E->isSizeOfAlignOfOp()) {
    // Return the result in the right width.
    Result.zextOrTrunc(static_cast<uint32_t>(Ctx.getTypeSize(E->getType())));

    // sizeof(void) and __alignof__(void) = 1 as a gcc extension.
    if (E->getSubExpr()->getType()->isVoidType())
      Result = 1;

    // sizeof(vla) is not a constantexpr: C99 6.5.3.4p2.
    if (!E->getSubExpr()->getType()->isConstantSizeType()) {
      // FIXME: Should we attempt to evaluate this?
      return APValue();
    }

    // Get information about the size or align.
    if (E->getSubExpr()->getType()->isFunctionType()) {
      // GCC extension: sizeof(function) = 1.
      // FIXME: AlignOf shouldn't be unconditionally 4!
      Result = E->getOpcode() == UnaryOperator::AlignOf ? 4 : 1;
    } else {
      unsigned CharSize = Ctx.Target.getCharWidth();
      if (E->getOpcode() == UnaryOperator::AlignOf)
        Result = Ctx.getTypeAlign(E->getSubExpr()->getType()) / CharSize;
      else
        Result = Ctx.getTypeSize(E->getSubExpr()->getType()) / CharSize;
    }
  } else {
    // Get the operand value.  If this is sizeof/alignof, do not evalute the
    // operand.  This affects C99 6.6p3.
    if (!Evaluate(E->getSubExpr(), Result, Ctx))
      return APValue();

    switch (E->getOpcode()) {
      // Address, indirect, pre/post inc/dec, etc are not valid constant exprs.
      // See C99 6.6p3.
    default:
      return APValue();
    case UnaryOperator::Extension:
      assert(0 && "Handle UnaryOperator::Extension");
      return APValue();  
    case UnaryOperator::LNot: {
      bool Val = Result == 0;
      uint32_t typeSize = Ctx.getTypeSize(E->getType());
      Result.zextOrTrunc(typeSize);
      Result = Val;
      break;
    }
    case UnaryOperator::Plus:
      break;
    case UnaryOperator::Minus:
      Result = -Result;
      break;
    case UnaryOperator::Not:
      Result = ~Result;
      break;
    }
  }

  Result.setIsUnsigned(E->getType()->isUnsignedIntegerType());
  return APValue(Result);    
}
  
APValue IntExprEvaluator::HandleCast(const Expr* SubExpr, QualType DestType) {
  llvm::APSInt Result(32);

  uint32_t DestWidth = static_cast<uint32_t>(Ctx.getTypeSize(DestType));

  // Handle simple integer->integer casts.
  if (SubExpr->getType()->isIntegerType()) {
    if (!Evaluate(SubExpr, Result, Ctx))
      return APValue();
    
    // Figure out if this is a truncate, extend or noop cast.
    // If the input is signed, do a sign extend, noop, or truncate.
    if (DestType->isBooleanType()) {
      // Conversion to bool compares against zero.
      Result = Result != 0;
      Result.zextOrTrunc(DestWidth);
    }
    else
      Result.extOrTrunc(DestWidth);
  } else if (SubExpr->getType()->isPointerType()) {
    APValue LV;
    if (!PointerExprEvaluator::Evaluate(SubExpr, LV, Ctx))
      return APValue();
    if (LV.getLValueBase())
      return APValue();
    
    Result = llvm::APSInt(DestWidth, LV.getLValueOffset());
  } else {
    assert(0 && "Unhandled cast!");
  }
  
  Result.setIsUnsigned(DestType->isUnsignedIntegerType());
  return APValue(Result); 
}

APValue IntExprEvaluator::VisitSizeOfAlignOfTypeExpr
  (const SizeOfAlignOfTypeExpr *E)
{
  llvm::APSInt Result(32);

  // Return the result in the right width.
  Result.zextOrTrunc(static_cast<uint32_t>(Ctx.getTypeSize(E->getType())));

  // sizeof(void) and __alignof__(void) = 1 as a gcc extension.
  if (E->getArgumentType()->isVoidType()) {
    Result = 1;
    Result.setIsUnsigned(E->getType()->isUnsignedIntegerType());
    return APValue(Result);
  }

  // alignof always evaluates to a constant, sizeof does if arg is not VLA.
  if (E->isSizeOf() && !E->getArgumentType()->isConstantSizeType()) 
    return APValue();

  // Get information about the size or align.
  if (E->getArgumentType()->isFunctionType()) {
    // GCC extension: sizeof(function) = 1.
    Result = E->isSizeOf() ? 1 : 4;
  } else { 
    unsigned CharSize = Ctx.Target.getCharWidth();
    if (E->isSizeOf())
      Result = Ctx.getTypeSize(E->getArgumentType()) / CharSize;
    else
      Result = Ctx.getTypeAlign(E->getArgumentType()) / CharSize;
  }
  
  Result.setIsUnsigned(E->getType()->isUnsignedIntegerType());
  return APValue(Result);
}

}
  
bool Expr::tryEvaluate(APValue& Result, ASTContext &Ctx) const
{
  llvm::APSInt sInt(1);
  
#if USE_NEW_EVALUATOR
  if (getType()->isIntegerType()) {
    if (IntExprEvaluator::Evaluate(this, sInt, Ctx)) {
      Result = APValue(sInt);
      return true;
    }
  } else
    return false;
    
#else
  if (CalcFakeICEVal(this, sInt, Ctx)) {
    Result = APValue(sInt);
    return true;
  }
#endif
  
  return false;
}
