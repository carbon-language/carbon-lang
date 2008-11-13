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
#include "clang/AST/RecordLayout.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/Support/Compiler.h"
using namespace clang;
using llvm::APSInt;
using llvm::APFloat;

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


static bool EvaluateLValue(const Expr *E, APValue &Result, EvalInfo &Info);
static bool EvaluatePointer(const Expr *E, APValue &Result, EvalInfo &Info);
static bool EvaluateInteger(const Expr *E, APSInt  &Result, EvalInfo &Info);
static bool EvaluateFloat(const Expr *E, APFloat &Result, EvalInfo &Info);

//===----------------------------------------------------------------------===//
// Misc utilities
//===----------------------------------------------------------------------===//

static bool HandleConversionToBool(Expr* E, bool& Result, EvalInfo &Info) {
  if (E->getType()->isIntegralType()) {
    APSInt IntResult;
    if (!EvaluateInteger(E, IntResult, Info))
      return false;
    Result = IntResult != 0;
    return true;
  } else if (E->getType()->isRealFloatingType()) {
    APFloat FloatResult(0.0);
    if (!EvaluateFloat(E, FloatResult, Info))
      return false;
    Result = !FloatResult.isZero();
    return true;
  } else if (E->getType()->isPointerType()) {
    APValue PointerResult;
    if (!EvaluatePointer(E, PointerResult, Info))
      return false;
    // FIXME: Is this accurate for all kinds of bases?  If not, what would
    // the check look like?
    Result = PointerResult.getLValueBase() || PointerResult.getLValueOffset();
    return true;
  }

  return false;
}

//===----------------------------------------------------------------------===//
// LValue Evaluation
//===----------------------------------------------------------------------===//
namespace {
class VISIBILITY_HIDDEN LValueExprEvaluator
  : public StmtVisitor<LValueExprEvaluator, APValue> {
  EvalInfo &Info;
public:
    
  LValueExprEvaluator(EvalInfo &info) : Info(info) {}

  APValue VisitStmt(Stmt *S) {
#if 0
    // FIXME: Remove this when we support more expressions.
    printf("Unhandled pointer statement\n");
    S->dump();  
#endif
    return APValue();
  }

  APValue VisitParenExpr(ParenExpr *E) { return Visit(E->getSubExpr()); }
  APValue VisitDeclRefExpr(DeclRefExpr *E) { return APValue(E, 0); }
  APValue VisitPredefinedExpr(PredefinedExpr *E) { return APValue(E, 0); }
  APValue VisitCompoundLiteralExpr(CompoundLiteralExpr *E);
  APValue VisitMemberExpr(MemberExpr *E);
  APValue VisitStringLiteral(StringLiteral *E) { return APValue(E, 0); }
};
} // end anonymous namespace

static bool EvaluateLValue(const Expr* E, APValue& Result, EvalInfo &Info) {
  Result = LValueExprEvaluator(Info).Visit(const_cast<Expr*>(E));
  return Result.isLValue();
}

APValue LValueExprEvaluator::VisitCompoundLiteralExpr(CompoundLiteralExpr *E) {
  if (E->isFileScope())
    return APValue(E, 0);
  return APValue();
}

APValue LValueExprEvaluator::VisitMemberExpr(MemberExpr *E) {
  APValue result;
  QualType Ty;
  if (E->isArrow()) {
    if (!EvaluatePointer(E->getBase(), result, Info))
      return APValue();
    Ty = E->getBase()->getType()->getAsPointerType()->getPointeeType();
  } else {
    result = Visit(E->getBase());
    if (result.isUninit())
      return APValue();
    Ty = E->getBase()->getType();
  }

  RecordDecl *RD = Ty->getAsRecordType()->getDecl();
  const ASTRecordLayout &RL = Info.Ctx.getASTRecordLayout(RD);
  FieldDecl *FD = E->getMemberDecl();
    
  // FIXME: This is linear time.
  unsigned i = 0, e = 0;
  for (i = 0, e = RD->getNumMembers(); i != e; i++) {
    if (RD->getMember(i) == FD)
      break;
  }

  result.setLValue(result.getLValueBase(),
                   result.getLValueOffset() + RL.getFieldOffset(i) / 8);

  return result;
}


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
    return APValue();
  }

  APValue VisitParenExpr(ParenExpr *E) { return Visit(E->getSubExpr()); }

  APValue VisitBinaryOperator(const BinaryOperator *E);
  APValue VisitCastExpr(const CastExpr* E);
  APValue VisitUnaryOperator(const UnaryOperator *E);
  APValue VisitObjCStringLiteral(ObjCStringLiteral *E)
      { return APValue(E, 0); }
  APValue VisitConditionalOperator(ConditionalOperator *E);
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

  QualType PointeeType = PExp->getType()->getAsPointerType()->getPointeeType();
  uint64_t SizeOfPointee = Info.Ctx.getTypeSize(PointeeType) / 8;

  uint64_t Offset = ResultLValue.getLValueOffset();

  if (E->getOpcode() == BinaryOperator::Add)
    Offset += AdditionalOffset.getLimitedValue() * SizeOfPointee;
  else
    Offset -= AdditionalOffset.getLimitedValue() * SizeOfPointee;

  return APValue(ResultLValue.getLValueBase(), Offset);
}

APValue PointerExprEvaluator::VisitUnaryOperator(const UnaryOperator *E) {
  if (E->getOpcode() == UnaryOperator::Extension) {
    // FIXME: Deal with warnings?
    return Visit(E->getSubExpr());
  }

  if (E->getOpcode() == UnaryOperator::AddrOf) {
    APValue result;
    if (EvaluateLValue(E->getSubExpr(), result, Info))
      return result;
  }

  return APValue();
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

  if (SubExpr->getType()->isFunctionType() ||
      SubExpr->getType()->isArrayType()) {
    APValue Result;
    if (EvaluateLValue(SubExpr, Result, Info))
      return Result;
    return APValue();
  }

  //assert(0 && "Unhandled cast");
  return APValue();
}  

APValue PointerExprEvaluator::VisitConditionalOperator(ConditionalOperator *E) {
  bool BoolResult;
  if (!HandleConversionToBool(E->getCond(), BoolResult, Info))
    return APValue();

  Expr* EvalExpr = BoolResult ? E->getTrueExpr() : E->getFalseExpr();

  APValue Result;
  if (EvaluatePointer(EvalExpr, Result, Info))
    return Result;
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
    
  bool Error(SourceLocation L, diag::kind D, QualType ExprTy) {
    // If this is in an unevaluated portion of the subexpression, ignore the
    // error.
    if (!Info.isEvaluated) {
      // If error is ignored because the value isn't evaluated, get the real
      // type at least to prevent errors downstream.
      Result.zextOrTrunc(getIntTypeSizeInBits(ExprTy));
      Result.setIsUnsigned(ExprTy->isUnsignedIntegerType());
      return true;
    }
    
    // Take the first error.
    if (Info.ICEDiag == 0) {
      Info.DiagLoc = L;
      Info.ICEDiag = D;
    }
    return false;
  }
    
  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//
  
  bool VisitStmt(Stmt *) {
    assert(0 && "This should be called on integers, stmts are not integers");
    return false;
  }
    
  bool VisitExpr(Expr *E) {
    return Error(E->getLocStart(), diag::err_expr_not_constant, E->getType());
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
    // Per gcc docs "this built-in function ignores top level
    // qualifiers".  We need to use the canonical version to properly
    // be able to strip CRV qualifiers from the type.
    QualType T0 = Info.Ctx.getCanonicalType(E->getArgType1());
    QualType T1 = Info.Ctx.getCanonicalType(E->getArgType2());
    Result = Info.Ctx.typesAreCompatible(T0.getUnqualifiedType(), 
                                         T1.getUnqualifiedType());
    return true;
  }
  bool VisitDeclRefExpr(const DeclRefExpr *E);
  bool VisitCallExpr(const CallExpr *E);
  bool VisitBinOpComma(const BinaryOperator *E);
  bool VisitBinaryOperator(const BinaryOperator *E);
  bool VisitUnaryOperator(const UnaryOperator *E);

  bool VisitCastExpr(CastExpr* E) {
    return HandleCast(E->getLocStart(), E->getSubExpr(), E->getType());
  }
  bool VisitSizeOfAlignOfExpr(const SizeOfAlignOfExpr *E);

private:
  bool HandleCast(SourceLocation CastLoc, Expr *SubExpr, QualType DestType);
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
  return Error(E->getLocStart(), diag::err_expr_not_constant, E->getType());
}

/// EvaluateBuiltinClassifyType - Evaluate __builtin_classify_type the same way
/// as GCC.
static int EvaluateBuiltinClassifyType(const CallExpr *E) {
  // The following enum mimics the values returned by GCC.
  enum gcc_type_class {
    no_type_class = -1,
    void_type_class, integer_type_class, char_type_class,
    enumeral_type_class, boolean_type_class,
    pointer_type_class, reference_type_class, offset_type_class,
    real_type_class, complex_type_class,
    function_type_class, method_type_class,
    record_type_class, union_type_class,
    array_type_class, string_type_class,
    lang_type_class
  };
  
  // If no argument was supplied, default to "no_type_class". This isn't 
  // ideal, however it is what gcc does.
  if (E->getNumArgs() == 0)
    return no_type_class;
  
  QualType ArgTy = E->getArg(0)->getType();
  if (ArgTy->isVoidType())
    return void_type_class;
  else if (ArgTy->isEnumeralType())
    return enumeral_type_class;
  else if (ArgTy->isBooleanType())
    return boolean_type_class;
  else if (ArgTy->isCharType())
    return string_type_class; // gcc doesn't appear to use char_type_class
  else if (ArgTy->isIntegerType())
    return integer_type_class;
  else if (ArgTy->isPointerType())
    return pointer_type_class;
  else if (ArgTy->isReferenceType())
    return reference_type_class;
  else if (ArgTy->isRealType())
    return real_type_class;
  else if (ArgTy->isComplexType())
    return complex_type_class;
  else if (ArgTy->isFunctionType())
    return function_type_class;
  else if (ArgTy->isStructureType())
    return record_type_class;
  else if (ArgTy->isUnionType())
    return union_type_class;
  else if (ArgTy->isArrayType())
    return array_type_class;
  else if (ArgTy->isUnionType())
    return union_type_class;
  else  // FIXME: offset_type_class, method_type_class, & lang_type_class?
    assert(0 && "CallExpr::isBuiltinClassifyType(): unimplemented type");
  return -1;
}

bool IntExprEvaluator::VisitCallExpr(const CallExpr *E) {
  Result.zextOrTrunc(getIntTypeSizeInBits(E->getType()));
  
  switch (E->isBuiltinCall()) {
  default:
    return Error(E->getLocStart(), diag::err_expr_not_constant, E->getType());
  case Builtin::BI__builtin_classify_type:
    Result.setIsSigned(true);
    Result = EvaluateBuiltinClassifyType(E);
    return true;
    
  case Builtin::BI__builtin_constant_p: {
    // __builtin_constant_p always has one operand: it returns true if that
    // operand can be folded, false otherwise.
    APValue Res;
    Result = E->getArg(0)->tryEvaluate(Res, Info.Ctx);
    return true;
  }
  }
}

bool IntExprEvaluator::VisitBinOpComma(const BinaryOperator *E) {
  llvm::APSInt RHS(32);

  // Require that we be able to evaluate the LHS.
  if (!E->getLHS()->isEvaluatable(Info.Ctx))
    return false;

  bool OldEval = Info.isEvaluated;
  if (!EvaluateInteger(E->getRHS(), RHS, Info))
    return false;
  Info.isEvaluated = OldEval;

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

bool IntExprEvaluator::VisitBinaryOperator(const BinaryOperator *E) {
  // Comma operator requires special handling.
  if (E->getOpcode() == BinaryOperator::Comma)
    return VisitBinOpComma(E);
  
  // The LHS of a constant expr is always evaluated and needed.
  llvm::APSInt RHS(32);
  if (!Visit(E->getLHS())) {
    // If the LHS is unfoldable, we generally can't fold this.  However, if this
    // is a logical operator like &&/||, and if we know that the RHS determines
    // the outcome of the result (e.g. X && 0), return the outcome.
    if (!E->isLogicalOp())
      return false;

    // If this is a logical op, see if the RHS determines the outcome.
    EvalInfo Info2(Info.Ctx);
    if (!EvaluateInteger(E->getRHS(), RHS, Info2))
      return false;
    
    // X && 0 -> 0, X || 1 -> 1.
    if ((E->getOpcode() == BinaryOperator::LAnd && RHS == 0) ||
        (E->getOpcode() == BinaryOperator::LOr  && RHS != 0)) {
      Result = RHS != 0;
      Result.zextOrTrunc(getIntTypeSizeInBits(E->getType()));
      Result.setIsUnsigned(E->getType()->isUnsignedIntegerType());
      return true;
    }
    
    return false; // error in subexpression.
  }
  
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
  default:
    return Error(E->getOperatorLoc(), diag::err_expr_not_constant,E->getType());
  case BinaryOperator::Mul: Result *= RHS; return true;
  case BinaryOperator::Add: Result += RHS; return true;
  case BinaryOperator::Sub: Result -= RHS; return true;
  case BinaryOperator::And: Result &= RHS; return true;
  case BinaryOperator::Xor: Result ^= RHS; return true;
  case BinaryOperator::Or:  Result |= RHS; return true;
  case BinaryOperator::Div:
    if (RHS == 0)
      return Error(E->getOperatorLoc(), diag::err_expr_divide_by_zero,
                   E->getType());
    Result /= RHS;
    break;
  case BinaryOperator::Rem:
    if (RHS == 0)
      return Error(E->getOperatorLoc(), diag::err_expr_divide_by_zero,
                   E->getType());
    Result %= RHS;
    break;
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
}

  Result.setIsUnsigned(E->getType()->isUnsignedIntegerType());
  return true;
}

/// VisitSizeAlignOfExpr - Evaluate a sizeof or alignof with a result as the
/// expression's type.
bool IntExprEvaluator::VisitSizeOfAlignOfExpr(const SizeOfAlignOfExpr *E) {
  QualType DstTy = E->getType();
  // Return the result in the right width.
  Result.zextOrTrunc(getIntTypeSizeInBits(DstTy));
  Result.setIsUnsigned(DstTy->isUnsignedIntegerType());

  QualType SrcTy = E->getTypeOfArgument();

  // sizeof(void) and __alignof__(void) = 1 as a gcc extension.
  if (SrcTy->isVoidType()) {
    Result = 1;
    return true;
  }
  
  // sizeof(vla) is not a constantexpr: C99 6.5.3.4p2.
  // FIXME: But alignof(vla) is!
  if (!SrcTy->isConstantSizeType()) {
    // FIXME: Should we attempt to evaluate this?
    return false;
  }

  bool isSizeOf = E->isSizeOf();
  
  // GCC extension: sizeof(function) = 1.
  if (SrcTy->isFunctionType()) {
    // FIXME: AlignOf shouldn't be unconditionally 4!
    Result = isSizeOf ? 1 : 4;
    return true;
  }
  
  // Get information about the size or align.
  unsigned CharSize = Info.Ctx.Target.getCharWidth();
  if (isSizeOf)
    Result = Info.Ctx.getTypeSize(SrcTy) / CharSize;
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
  
  // Get the operand value into 'Result'.
  if (!Visit(E->getSubExpr()))
    return false;

  switch (E->getOpcode()) {
  default:
    // Address, indirect, pre/post inc/dec, etc are not valid constant exprs.
    // See C99 6.6p3.
    return Error(E->getOperatorLoc(), diag::err_expr_not_constant,
                 E->getType());
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

  if (DestType->isBooleanType()) {
    bool BoolResult;
    if (!HandleConversionToBool(SubExpr, BoolResult, Info))
      return false;
    Result.zextOrTrunc(DestWidth);
    Result.setIsUnsigned(DestType->isUnsignedIntegerType());
    Result = BoolResult;
    return true;
  }

  // Handle simple integer->integer casts.
  if (SubExpr->getType()->isIntegerType()) {
    if (!Visit(SubExpr))
      return false;
    
    // Figure out if this is a truncate, extend or noop cast.
    // If the input is signed, do a sign extend, noop, or truncate.
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
    return Error(CastLoc, diag::err_expr_not_constant, DestType);

  APFloat F(0.0);
  if (!EvaluateFloat(SubExpr, F, Info))
    return Error(CastLoc, diag::err_expr_not_constant, DestType);
  
  // Determine whether we are converting to unsigned or signed.
  bool DestSigned = DestType->isSignedIntegerType();
  
  // FIXME: Warning for overflow.
  uint64_t Space[4];
  bool ignored;
  (void)F.convertToInteger(Space, DestWidth, DestSigned,
                           llvm::APFloat::rmTowardZero, &ignored);
  Result = llvm::APInt(DestWidth, 4, Space);
  Result.setIsUnsigned(!DestSigned);
  return true;
}

//===----------------------------------------------------------------------===//
// Float Evaluation
//===----------------------------------------------------------------------===//

namespace {
class VISIBILITY_HIDDEN FloatExprEvaluator
  : public StmtVisitor<FloatExprEvaluator, bool> {
  EvalInfo &Info;
  APFloat &Result;
public:
  FloatExprEvaluator(EvalInfo &info, APFloat &result)
    : Info(info), Result(result) {}

  bool VisitStmt(Stmt *S) {
    return false;
  }

  bool VisitParenExpr(ParenExpr *E) { return Visit(E->getSubExpr()); }
  bool VisitCallExpr(const CallExpr *E);

  bool VisitUnaryOperator(const UnaryOperator *E);
  bool VisitBinaryOperator(const BinaryOperator *E);
  bool VisitFloatingLiteral(const FloatingLiteral *E);
  bool VisitCastExpr(CastExpr *E);
  bool VisitCXXZeroInitValueExpr(CXXZeroInitValueExpr *E);
};
} // end anonymous namespace

static bool EvaluateFloat(const Expr* E, APFloat& Result, EvalInfo &Info) {
  return FloatExprEvaluator(Info, Result).Visit(const_cast<Expr*>(E));
}

bool FloatExprEvaluator::VisitCallExpr(const CallExpr *E) {
  switch (E->isBuiltinCall()) {
  default: return false;
  case Builtin::BI__builtin_huge_val:
  case Builtin::BI__builtin_huge_valf:
  case Builtin::BI__builtin_huge_vall:
  case Builtin::BI__builtin_inf:
  case Builtin::BI__builtin_inff:
  case Builtin::BI__builtin_infl: {
    const llvm::fltSemantics &Sem =
      Info.Ctx.getFloatTypeSemantics(E->getType());
    Result = llvm::APFloat::getInf(Sem);
    return true;
  }
      
  case Builtin::BI__builtin_nan:
  case Builtin::BI__builtin_nanf:
  case Builtin::BI__builtin_nanl:
    // If this is __builtin_nan("") turn this into a simple nan, otherwise we
    // can't constant fold it.
    if (const StringLiteral *S = 
        dyn_cast<StringLiteral>(E->getArg(0)->IgnoreParenCasts())) {
      if (!S->isWide() && S->getByteLength() == 0) { // empty string.
        const llvm::fltSemantics &Sem =
          Info.Ctx.getFloatTypeSemantics(E->getType());
        Result = llvm::APFloat::getNaN(Sem);
        return true;
      }
    }
    return false;

  case Builtin::BI__builtin_fabs:
  case Builtin::BI__builtin_fabsf:
  case Builtin::BI__builtin_fabsl:
    if (!EvaluateFloat(E->getArg(0), Result, Info))
      return false;
    
    if (Result.isNegative())
      Result.changeSign();
    return true;

  case Builtin::BI__builtin_copysign: 
  case Builtin::BI__builtin_copysignf: 
  case Builtin::BI__builtin_copysignl: {
    APFloat RHS(0.);
    if (!EvaluateFloat(E->getArg(0), Result, Info) ||
        !EvaluateFloat(E->getArg(1), RHS, Info))
      return false;
    Result.copySign(RHS);
    return true;
  }
  }
}

bool FloatExprEvaluator::VisitUnaryOperator(const UnaryOperator *E) {
  if (!EvaluateFloat(E->getSubExpr(), Result, Info))
    return false;

  switch (E->getOpcode()) {
  default: return false;
  case UnaryOperator::Plus: 
    return true;
  case UnaryOperator::Minus:
    Result.changeSign();
    return true;
  }
}

bool FloatExprEvaluator::VisitBinaryOperator(const BinaryOperator *E) {
  // FIXME: Diagnostics?  I really don't understand how the warnings
  // and errors are supposed to work.
  APFloat RHS(0.0);
  if (!EvaluateFloat(E->getLHS(), Result, Info))
    return false;
  if (!EvaluateFloat(E->getRHS(), RHS, Info))
    return false;

  switch (E->getOpcode()) {
  default: return false;
  case BinaryOperator::Mul:
    Result.multiply(RHS, APFloat::rmNearestTiesToEven);
    return true;
  case BinaryOperator::Add:
    Result.add(RHS, APFloat::rmNearestTiesToEven);
    return true;
  case BinaryOperator::Sub:
    Result.subtract(RHS, APFloat::rmNearestTiesToEven);
    return true;
  case BinaryOperator::Div:
    Result.divide(RHS, APFloat::rmNearestTiesToEven);
    return true;
  case BinaryOperator::Rem:
    Result.mod(RHS, APFloat::rmNearestTiesToEven);
    return true;
  }
}

bool FloatExprEvaluator::VisitFloatingLiteral(const FloatingLiteral *E) {
  Result = E->getValue();
  return true;
}

bool FloatExprEvaluator::VisitCastExpr(CastExpr *E) {
  Expr* SubExpr = E->getSubExpr();
  const llvm::fltSemantics& destSemantics =
      Info.Ctx.getFloatTypeSemantics(E->getType());
  if (SubExpr->getType()->isIntegralType()) {
    APSInt IntResult;
    if (!EvaluateInteger(E, IntResult, Info))
      return false;
    Result = APFloat(destSemantics, 1);
    Result.convertFromAPInt(IntResult, IntResult.isSigned(),
                            APFloat::rmNearestTiesToEven);
    return true;
  }
  if (SubExpr->getType()->isRealFloatingType()) {
    if (!Visit(SubExpr))
      return false;
    bool ignored;
    Result.convert(destSemantics, APFloat::rmNearestTiesToEven, &ignored);
    return true;
  }

  return false;
}

bool FloatExprEvaluator::VisitCXXZeroInitValueExpr(CXXZeroInitValueExpr *E) {
  Result = APFloat::getZero(Info.Ctx.getFloatTypeSemantics(E->getType()));
  return true;
}

//===----------------------------------------------------------------------===//
// Top level TryEvaluate.
//===----------------------------------------------------------------------===//

/// tryEvaluate - Return true if this is a constant which we can fold using
/// any crazy technique (that has nothing to do with language standards) that
/// we want to.  If this function returns true, it returns the folded constant
/// in Result.
bool Expr::tryEvaluate(APValue &Result, ASTContext &Ctx) const {
  EvalInfo Info(Ctx);
  if (getType()->isIntegerType()) {
    llvm::APSInt sInt(32);
    if (EvaluateInteger(this, sInt, Info)) {
      Result = APValue(sInt);
      return true;
    }
  } else if (getType()->isPointerType()) {
    if (EvaluatePointer(this, Result, Info)) {
      return true;
    }
  } else if (getType()->isRealFloatingType()) {
    llvm::APFloat f(0.0);
    if (EvaluateFloat(this, f, Info)) {
      Result = APValue(f);
      return true;
    }
  }
      
  return false;
}

/// isEvaluatable - Call tryEvaluate to see if this expression can be constant
/// folded, but discard the result.
bool Expr::isEvaluatable(ASTContext &Ctx) const {
  APValue V;
  return tryEvaluate(V, Ctx);
}
