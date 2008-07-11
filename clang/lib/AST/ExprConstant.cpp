//===--- ExprConstant.cpp - Expression Constant Evaluator -----------------===//
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
using llvm::APSInt;

#define USE_NEW_EVALUATOR 0

static bool CalcFakeICEVal(const Expr *Expr,
                           llvm::APSInt &Result,
                           ASTContext &Context) {
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

/// EvalInfo - This is a private struct used by the evaluator to capture
/// information about a subexpression as it is folded.  It retains information
/// about the AST context, but also maintains information about the folded
/// expression.
///
/// If an expression could be evaluated, it is still possible it is not a C
/// "integer constant expression" or constant expression.  If not, this struct
/// captures information about how and why not.
///
/// One bit of information passed *into* the request for constant folding
/// indicates whether the subexpression is "evaluated" or not according to C
/// rules.  For example, the RHS of (0 && foo()) is not evaluated.  We can
/// evaluate the expression regardless of what the RHS is, but C only allows
/// certain things in certain situations.
struct EvalInfo {
  ASTContext &Ctx;
  
  /// isEvaluated - True if the subexpression is required to be evaluated, false
  /// if it is short-circuited (according to C rules).
  bool isEvaluated;
  
  /// ICEDiag - If the expression is foldable, but the expression is not an
  /// integer constant expression, this contains the extension diagnostic to
  /// emit which describes why it isn't an integer constant expression.  The
  /// caller can choose to emit this or not, depending on whether they require
  /// an i-c-e or not.  DiagLoc indicates the caret position for the report.
  ///
  /// If ICEDiag is zero, then this expression is an i-c-e.
  unsigned ICEDiag;
  SourceLocation DiagLoc;

  EvalInfo(ASTContext &ctx) : Ctx(ctx), isEvaluated(true), ICEDiag(0) {}
};


static bool EvaluatePointer(const Expr *E, APValue &Result, EvalInfo &Info);
static bool EvaluateInteger(const Expr *E, APSInt  &Result, EvalInfo &Info);


//===----------------------------------------------------------------------===//
// Pointer Evaluation
//===----------------------------------------------------------------------===//

namespace {
class VISIBILITY_HIDDEN PointerExprEvaluator
  : public StmtVisitor<PointerExprEvaluator, APValue> {
  EvalInfo &Info;
public:
    
  PointerExprEvaluator(EvalInfo &info) : Info(info) {}

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
} // end anonymous namespace

static bool EvaluatePointer(const Expr* E, APValue& Result, EvalInfo &Info) {
  if (!E->getType()->isPointerType())
    return false;
  Result = PointerExprEvaluator(Info).Visit(const_cast<Expr*>(E));
  return Result.isLValue();
}

APValue PointerExprEvaluator::VisitBinaryOperator(const BinaryOperator *E) {
  if (E->getOpcode() != BinaryOperator::Add &&
      E->getOpcode() != BinaryOperator::Sub)
    return APValue();
  
  const Expr *PExp = E->getLHS();
  const Expr *IExp = E->getRHS();
  if (IExp->getType()->isPointerType())
    std::swap(PExp, IExp);
  
  APValue ResultLValue;
  if (!EvaluatePointer(PExp, ResultLValue, Info))
    return APValue();
  
  llvm::APSInt AdditionalOffset(32);
  if (!EvaluateInteger(IExp, AdditionalOffset, Info))
    return APValue();

  uint64_t Offset = ResultLValue.getLValueOffset();
  if (E->getOpcode() == BinaryOperator::Add)
    Offset += AdditionalOffset.getZExtValue();
  else
    Offset -= AdditionalOffset.getZExtValue();
    
  return APValue(ResultLValue.getLValueBase(), Offset);
}
  

APValue PointerExprEvaluator::VisitCastExpr(const CastExpr* E) {
  const Expr* SubExpr = E->getSubExpr();

   // Check for pointer->pointer cast
  if (SubExpr->getType()->isPointerType()) {
    APValue Result;
    if (EvaluatePointer(SubExpr, Result, Info))
      return Result;
    return APValue();
  }
  
  if (SubExpr->getType()->isArithmeticType()) {
    llvm::APSInt Result(32);
    if (EvaluateInteger(SubExpr, Result, Info)) {
      Result.extOrTrunc((unsigned)Info.Ctx.getTypeSize(E->getType()));
      return APValue(0, Result.getZExtValue());
    }
  }
  
  assert(0 && "Unhandled cast");
  return APValue();
}  


//===----------------------------------------------------------------------===//
// Integer Evaluation
//===----------------------------------------------------------------------===//

namespace {
class VISIBILITY_HIDDEN IntExprEvaluator
  : public StmtVisitor<IntExprEvaluator, bool> {
  EvalInfo &Info;
  APSInt &Result;
public:
  IntExprEvaluator(EvalInfo &info, APSInt &result)
    : Info(info), Result(result) {}

  unsigned getIntTypeSizeInBits(QualType T) const {
    return (unsigned)Info.Ctx.getTypeSize(T);
  }
    
  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//
    
  bool VisitStmt(Stmt *S) {
    // FIXME: Remove this when we support more expressions.
    printf("unhandled int expression");
    S->dump();  
    return false;
  }
  
  bool VisitParenExpr(ParenExpr *E) { return Visit(E->getSubExpr()); }

  bool VisitBinaryOperator(const BinaryOperator *E);

  bool VisitUnaryOperator(const UnaryOperator *E);

  bool VisitCastExpr(const CastExpr* E) {
    return HandleCast(E->getSubExpr(), E->getType());
  }
  bool VisitImplicitCastExpr(const ImplicitCastExpr* E) {
    return HandleCast(E->getSubExpr(), E->getType());
  }
  bool VisitSizeOfAlignOfTypeExpr(const SizeOfAlignOfTypeExpr *E) {
    return EvaluateSizeAlignOf(E->isSizeOf(),E->getArgumentType(),E->getType());
  }
 
  bool VisitIntegerLiteral(const IntegerLiteral *E) {
    Result = E->getValue();
    return true;
  }
private:
  bool HandleCast(const Expr* SubExpr, QualType DestType);
  bool EvaluateSizeAlignOf(bool isSizeOf, QualType SrcTy, QualType DstTy);
};
} // end anonymous namespace

static bool EvaluateInteger(const Expr* E, APSInt &Result, EvalInfo &Info) {
  return IntExprEvaluator(Info, Result).Visit(const_cast<Expr*>(E));
}


bool IntExprEvaluator::VisitBinaryOperator(const BinaryOperator *E) {
  // The LHS of a constant expr is always evaluated and needed.
  llvm::APSInt RHS(32);
  if (!Visit(E->getLHS()) || !EvaluateInteger(E->getRHS(), RHS, Info))
    return false;
  
  switch (E->getOpcode()) {
  default: return false;
  case BinaryOperator::Mul: Result *= RHS; break;
  case BinaryOperator::Add: Result += RHS; break;
  case BinaryOperator::Sub: Result -= RHS; break;
  case BinaryOperator::And: Result &= RHS; break;
  case BinaryOperator::Xor: Result ^= RHS; break;
  case BinaryOperator::Or:  Result |= RHS; break;
  case BinaryOperator::Div:
    if (RHS == 0) return false;
    Result /= RHS;
    break;
  case BinaryOperator::Rem:
    if (RHS == 0) return false;
    Result %= RHS;
    break;
  case BinaryOperator::Shl:
    Result <<= (unsigned)RHS.getLimitedValue(Result.getBitWidth()-1);
    break;
  case BinaryOperator::Shr:
    Result >>= (unsigned)RHS.getLimitedValue(Result.getBitWidth()-1);
    break;
      
  case BinaryOperator::LT:
    Result = Result < RHS;
    Result.zextOrTrunc(getIntTypeSizeInBits(E->getType()));
    break;
  case BinaryOperator::GT:
    Result = Result > RHS;
    Result.zextOrTrunc(getIntTypeSizeInBits(E->getType()));
    break;
  case BinaryOperator::LE:
    Result = Result <= RHS;
    Result.zextOrTrunc(getIntTypeSizeInBits(E->getType()));
    break;
  case BinaryOperator::GE:
    Result = Result >= RHS;
    Result.zextOrTrunc(getIntTypeSizeInBits(E->getType()));
    break;
  case BinaryOperator::EQ:
    Result = Result == RHS;
    Result.zextOrTrunc(getIntTypeSizeInBits(E->getType()));
    break;
  case BinaryOperator::NE:
    Result = Result != RHS;
    Result.zextOrTrunc(getIntTypeSizeInBits(E->getType()));
    break;
    
  case BinaryOperator::Comma:
    // C99 6.6p3: "shall not contain assignment, ..., or comma operators,
    // *except* when they are contained within a subexpression that is not
    // evaluated".  Note that Assignment can never happen due to constraints
    // on the LHS subexpr, so we don't need to check it here.
    // FIXME: Need to come up with an efficient way to deal with the C99
    // rules on evaluation while still evaluating this.  Maybe a
    // "evaluated comma" out parameter?
    return false;
  }

  Result.setIsUnsigned(E->getType()->isUnsignedIntegerType());
  return true;
}

/// EvaluateSizeAlignOf - Evaluate sizeof(SrcTy) or alignof(SrcTy) with a result
/// as a DstTy type.
bool IntExprEvaluator::EvaluateSizeAlignOf(bool isSizeOf, QualType SrcTy,
                                           QualType DstTy) {
  // Return the result in the right width.
  Result.zextOrTrunc(getIntTypeSizeInBits(DstTy));
  Result.setIsUnsigned(DstTy->isUnsignedIntegerType());

  // sizeof(void) and __alignof__(void) = 1 as a gcc extension.
  if (SrcTy->isVoidType())
    Result = 1;
  
  // sizeof(vla) is not a constantexpr: C99 6.5.3.4p2.
  if (!SrcTy->isConstantSizeType()) {
    // FIXME: Should we attempt to evaluate this?
    return false;
  }
  
  // GCC extension: sizeof(function) = 1.
  if (SrcTy->isFunctionType()) {
    // FIXME: AlignOf shouldn't be unconditionally 4!
    Result = isSizeOf ? 1 : 4;
    return true;
  }
  
  // Get information about the size or align.
  unsigned CharSize = Info.Ctx.Target.getCharWidth();
  if (isSizeOf)
    Result = getIntTypeSizeInBits(SrcTy) / CharSize;
  else
    Result = Info.Ctx.getTypeAlign(SrcTy) / CharSize;
  return true;
}

bool IntExprEvaluator::VisitUnaryOperator(const UnaryOperator *E) {
  if (E->isOffsetOfOp()) {
    Result = E->evaluateOffsetOf(Info.Ctx);
    Result.setIsUnsigned(E->getType()->isUnsignedIntegerType());
    return true;
  }
  
  if (E->isSizeOfAlignOfOp())
    return EvaluateSizeAlignOf(E->getOpcode() == UnaryOperator::SizeOf,
                               E->getSubExpr()->getType(), E->getType());
  
  // Get the operand value into 'Result'.
  if (!Visit(E->getSubExpr()))
    return false;

  switch (E->getOpcode()) {
    // Address, indirect, pre/post inc/dec, etc are not valid constant exprs.
    // See C99 6.6p3.
  default:
    return false;
  case UnaryOperator::LNot: {
    bool Val = Result == 0;
    Result.zextOrTrunc(getIntTypeSizeInBits(E->getType()));
    Result = Val;
    break;
  }
  case UnaryOperator::Extension:
  case UnaryOperator::Plus:
    // The result is always just the subexpr
    break;
  case UnaryOperator::Minus:
    Result = -Result;
    break;
  case UnaryOperator::Not:
    Result = ~Result;
    break;
  }

  Result.setIsUnsigned(E->getType()->isUnsignedIntegerType());
  return true;
}
  
bool IntExprEvaluator::HandleCast(const Expr* SubExpr, QualType DestType) {
  unsigned DestWidth = getIntTypeSizeInBits(DestType);

  // Handle simple integer->integer casts.
  if (SubExpr->getType()->isIntegerType()) {
    if (!EvaluateInteger(SubExpr, Result, Info))
      return false;
    
    // Figure out if this is a truncate, extend or noop cast.
    // If the input is signed, do a sign extend, noop, or truncate.
    if (DestType->isBooleanType()) {
      // Conversion to bool compares against zero.
      Result = Result != 0;
      Result.zextOrTrunc(DestWidth);
    } else
      Result.extOrTrunc(DestWidth);
  } else if (SubExpr->getType()->isPointerType()) {
    APValue LV;
    if (!EvaluatePointer(SubExpr, LV, Info))
      return false;
    if (LV.getLValueBase())
      return false;
    
    Result.extOrTrunc(DestWidth);
    Result = LV.getLValueOffset();
  } else {
    assert(0 && "Unhandled cast!");
  }
  
  Result.setIsUnsigned(DestType->isUnsignedIntegerType());
  return true;
}

//===----------------------------------------------------------------------===//
// Top level TryEvaluate.
//===----------------------------------------------------------------------===//

bool Expr::tryEvaluate(APValue &Result, ASTContext &Ctx) const {
  llvm::APSInt sInt(32);
#if USE_NEW_EVALUATOR
  EvalInfo Info(Ctx);
  if (getType()->isIntegerType()) {
    if (EvaluateInteger(this, sInt, Info)) {
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
