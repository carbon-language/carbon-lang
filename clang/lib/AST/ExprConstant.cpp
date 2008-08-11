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
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/Support/Compiler.h"
using namespace clang;
using llvm::APSInt;

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
  
  /// ICEDiag - If the expression is unfoldable, then ICEDiag contains the 
  /// error diagnostic indicating why it is not foldable and DiagLoc indicates a
  /// caret position for the error.  If it is foldable, but the expression is
  /// not an integer constant expression, ICEDiag contains the extension
  /// diagnostic to emit which describes why it isn't an integer constant
  /// expression.  If this expression *is* an integer-constant-expr, then
  /// ICEDiag is zero.
  ///
  /// The caller can choose to emit this diagnostic or not, depending on whether
  /// they require an i-c-e or a constant or not.  DiagLoc indicates the caret
  /// position for the report.
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
  
  if (SubExpr->getType()->isIntegralType()) {
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
    return (unsigned)Info.Ctx.getIntWidth(T);
  }
  
  bool Extension(SourceLocation L, diag::kind D) {
    Info.DiagLoc = L;
    Info.ICEDiag = D;
    return true;  // still a constant.
  }
    
  bool Error(SourceLocation L, diag::kind D) {
    // If this is in an unevaluated portion of the subexpression, ignore the
    // error.
    if (!Info.isEvaluated)
      return true;
    
    Info.DiagLoc = L;
    Info.ICEDiag = D;
    return false;
  }
    
  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//
    
  bool VisitStmt(Stmt *S) {
    return Error(S->getLocStart(), diag::err_expr_not_constant);
  }
  
  bool VisitParenExpr(ParenExpr *E) { return Visit(E->getSubExpr()); }

  bool VisitIntegerLiteral(const IntegerLiteral *E) {
    Result = E->getValue();
    Result.setIsUnsigned(E->getType()->isUnsignedIntegerType());
    return true;
  }
  bool VisitCharacterLiteral(const CharacterLiteral *E) {
    Result.zextOrTrunc(getIntTypeSizeInBits(E->getType()));
    Result = E->getValue();
    Result.setIsUnsigned(E->getType()->isUnsignedIntegerType());
    return true;
  }
  bool VisitTypesCompatibleExpr(const TypesCompatibleExpr *E) {
    Result.zextOrTrunc(getIntTypeSizeInBits(E->getType()));
    Result = Info.Ctx.typesAreCompatible(E->getArgType1(), E->getArgType2());
    return true;
  }
  bool VisitDeclRefExpr(const DeclRefExpr *E);
  bool VisitCallExpr(const CallExpr *E);
  bool VisitBinaryOperator(const BinaryOperator *E);
  bool VisitUnaryOperator(const UnaryOperator *E);

  bool VisitCastExpr(CastExpr* E) {
    return HandleCast(E->getLParenLoc(), E->getSubExpr(), E->getType());
  }
  bool VisitImplicitCastExpr(ImplicitCastExpr* E) {
    return HandleCast(E->getLocStart(), E->getSubExpr(), E->getType());
  }
  bool VisitSizeOfAlignOfTypeExpr(const SizeOfAlignOfTypeExpr *E) {
    return EvaluateSizeAlignOf(E->isSizeOf(), E->getArgumentType(),
                               E->getType());
  }
    
private:
  bool HandleCast(SourceLocation CastLoc, Expr *SubExpr, QualType DestType);
  bool EvaluateSizeAlignOf(bool isSizeOf, QualType SrcTy, QualType DstTy);
};
} // end anonymous namespace

static bool EvaluateInteger(const Expr* E, APSInt &Result, EvalInfo &Info) {
  return IntExprEvaluator(Info, Result).Visit(const_cast<Expr*>(E));
}

bool IntExprEvaluator::VisitDeclRefExpr(const DeclRefExpr *E) {
  // Enums are integer constant exprs.
  if (const EnumConstantDecl *D = dyn_cast<EnumConstantDecl>(E->getDecl())) {
    Result = D->getInitVal();
    return true;
  }
  
  // Otherwise, random variable references are not constants.
  return Error(E->getLocStart(), diag::err_expr_not_constant);
}


bool IntExprEvaluator::VisitCallExpr(const CallExpr *E) {
  Result.zextOrTrunc(getIntTypeSizeInBits(E->getType()));
  // __builtin_type_compatible_p is a constant.
  if (E->isBuiltinClassifyType(Result))
    return true;
  
  return Error(E->getLocStart(), diag::err_expr_not_constant);
}

bool IntExprEvaluator::VisitBinaryOperator(const BinaryOperator *E) {
  // The LHS of a constant expr is always evaluated and needed.
  llvm::APSInt RHS(32);
  if (!Visit(E->getLHS()))
    return false; // error in subexpression.
  
  bool OldEval = Info.isEvaluated;

  // The short-circuiting &&/|| operators don't necessarily evaluate their
  // RHS.  Make sure to pass isEvaluated down correctly.
  if ((E->getOpcode() == BinaryOperator::LAnd && Result == 0) ||
      (E->getOpcode() == BinaryOperator::LOr  && Result != 0))
    Info.isEvaluated = false;

  // FIXME: Handle pointer subtraction

  // FIXME Maybe we want to succeed even where we can't evaluate the
  // right side of LAnd/LOr?
  // For example, see http://llvm.org/bugs/show_bug.cgi?id=2525 
  if (!EvaluateInteger(E->getRHS(), RHS, Info))
    return false;
  Info.isEvaluated = OldEval;
  
  switch (E->getOpcode()) {
  default: return Error(E->getOperatorLoc(), diag::err_expr_not_constant);
  case BinaryOperator::Mul: Result *= RHS; return true;
  case BinaryOperator::Add: Result += RHS; return true;
  case BinaryOperator::Sub: Result -= RHS; return true;
  case BinaryOperator::And: Result &= RHS; return true;
  case BinaryOperator::Xor: Result ^= RHS; return true;
  case BinaryOperator::Or:  Result |= RHS; return true;
  case BinaryOperator::Div:
    if (RHS == 0)
      return Error(E->getOperatorLoc(), diag::err_expr_divide_by_zero);
    Result /= RHS;
    return true;
  case BinaryOperator::Rem:
    if (RHS == 0)
      return Error(E->getOperatorLoc(), diag::err_expr_divide_by_zero);
    Result %= RHS;
    return true;
  case BinaryOperator::Shl:
    // FIXME: Warn about out of range shift amounts!
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
  case BinaryOperator::LAnd:
    Result = Result != 0 && RHS != 0;
    Result.zextOrTrunc(getIntTypeSizeInBits(E->getType()));
    break;
  case BinaryOperator::LOr:
    Result = Result != 0 || RHS != 0;
    Result.zextOrTrunc(getIntTypeSizeInBits(E->getType()));
    break;
      
    
  case BinaryOperator::Comma:
    // Result of the comma is just the result of the RHS.
    Result = RHS;

    // C99 6.6p3: "shall not contain assignment, ..., or comma operators,
    // *except* when they are contained within a subexpression that is not
    // evaluated".  Note that Assignment can never happen due to constraints
    // on the LHS subexpr, so we don't need to check it here.
    if (!Info.isEvaluated)
      return true;
      
    // If the value is evaluated, we can accept it as an extension.
    return Extension(E->getOperatorLoc(), diag::ext_comma_in_constant_expr);
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
  // Special case unary operators that do not need their subexpression
  // evaluated.  offsetof/sizeof/alignof are all special.
  if (E->isOffsetOfOp()) {
    Result.zextOrTrunc(getIntTypeSizeInBits(E->getType()));
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
  default:
    // Address, indirect, pre/post inc/dec, etc are not valid constant exprs.
    // See C99 6.6p3.
    return Error(E->getOperatorLoc(), diag::err_expr_not_constant);
  case UnaryOperator::LNot: {
    bool Val = Result == 0;
    Result.zextOrTrunc(getIntTypeSizeInBits(E->getType()));
    Result = Val;
    break;
  }
  case UnaryOperator::Extension:
    // FIXME: Should extension allow i-c-e extension expressions in its scope?
    // If so, we could clear the diagnostic ID.
  case UnaryOperator::Plus:
    // The result is always just the subexpr. 
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
  
/// HandleCast - This is used to evaluate implicit or explicit casts where the
/// result type is integer.
bool IntExprEvaluator::HandleCast(SourceLocation CastLoc,
                                  Expr *SubExpr, QualType DestType) {
  unsigned DestWidth = getIntTypeSizeInBits(DestType);

  // Handle simple integer->integer casts.
  if (SubExpr->getType()->isIntegerType()) {
    if (!Visit(SubExpr))
      return false;
    
    // Figure out if this is a truncate, extend or noop cast.
    // If the input is signed, do a sign extend, noop, or truncate.
    if (DestType->isBooleanType()) {
      // Conversion to bool compares against zero.
      Result = Result != 0;
      Result.zextOrTrunc(DestWidth);
    } else
      Result.extOrTrunc(DestWidth);
    Result.setIsUnsigned(DestType->isUnsignedIntegerType());
    return true;
  }
  
  // FIXME: Clean this up!
  if (SubExpr->getType()->isPointerType()) {
    APValue LV;
    if (!EvaluatePointer(SubExpr, LV, Info))
      return false;
    if (LV.getLValueBase())
      return false;
    
    Result.extOrTrunc(DestWidth);
    Result = LV.getLValueOffset();
    Result.setIsUnsigned(DestType->isUnsignedIntegerType());
    return true;
  }
  
  if (!SubExpr->getType()->isRealFloatingType())
    return Error(CastLoc, diag::err_expr_not_constant);

  // FIXME: Generalize floating point constant folding!  For now we just permit
  // which is allowed by integer constant expressions.
  
  // Allow floating constants that are the immediate operands of casts or that
  // are parenthesized.
  const Expr *Operand = SubExpr;
  while (const ParenExpr *PE = dyn_cast<ParenExpr>(Operand))
    Operand = PE->getSubExpr();
  
  // If this isn't a floating literal, we can't handle it.
  const FloatingLiteral *FL = dyn_cast<FloatingLiteral>(Operand);
  if (!FL)
    return Error(CastLoc, diag::err_expr_not_constant);
  
  // If the destination is boolean, compare against zero.
  if (DestType->isBooleanType()) {
    Result = !FL->getValue().isZero();
    Result.zextOrTrunc(DestWidth);
    Result.setIsUnsigned(DestType->isUnsignedIntegerType());
    return true;
  }     
  
  // Determine whether we are converting to unsigned or signed.
  bool DestSigned = DestType->isSignedIntegerType();
  
  // FIXME: Warning for overflow.
  uint64_t Space[4]; 
  (void)FL->getValue().convertToInteger(Space, DestWidth, DestSigned,
                                        llvm::APFloat::rmTowardZero);
  Result = llvm::APInt(DestWidth, 4, Space);
  Result.setIsUnsigned(!DestSigned);
  return true;
}

//===----------------------------------------------------------------------===//
// Top level TryEvaluate.
//===----------------------------------------------------------------------===//

bool Expr::tryEvaluate(APValue &Result, ASTContext &Ctx) const {
  llvm::APSInt sInt(32);
  
  EvalInfo Info(Ctx);
  if (getType()->isIntegerType()) {
    if (EvaluateInteger(this, sInt, Info)) {
      Result = APValue(sInt);
      return true;
    }
  } else
    return false;
      
  return false;
}
