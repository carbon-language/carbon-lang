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
#include "clang/AST/CharUnits.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/ASTDiagnostic.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/SmallString.h"
#include <cstring>

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

  /// EvalResult - Contains information about the evaluation.
  Expr::EvalResult &EvalResult;

  /// AnyLValue - Stack based LValue results are not discarded.
  bool AnyLValue;

  EvalInfo(ASTContext &ctx, Expr::EvalResult& evalresult,
           bool anylvalue = false)
    : Ctx(ctx), EvalResult(evalresult), AnyLValue(anylvalue) {}
};


static bool EvaluateLValue(const Expr *E, APValue &Result, EvalInfo &Info);
static bool EvaluatePointer(const Expr *E, APValue &Result, EvalInfo &Info);
static bool EvaluateInteger(const Expr *E, APSInt  &Result, EvalInfo &Info);
static bool EvaluateIntegerOrLValue(const Expr *E, APValue  &Result,
                                    EvalInfo &Info);
static bool EvaluateFloat(const Expr *E, APFloat &Result, EvalInfo &Info);
static bool EvaluateComplex(const Expr *E, APValue &Result, EvalInfo &Info);

//===----------------------------------------------------------------------===//
// Misc utilities
//===----------------------------------------------------------------------===//

static bool EvalPointerValueAsBool(APValue& Value, bool& Result) {
  // FIXME: Is this accurate for all kinds of bases?  If not, what would
  // the check look like?
  Result = Value.getLValueBase() || !Value.getLValueOffset().isZero();
  return true;
}

static bool HandleConversionToBool(const Expr* E, bool& Result,
                                   EvalInfo &Info) {
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
  } else if (E->getType()->hasPointerRepresentation()) {
    APValue PointerResult;
    if (!EvaluatePointer(E, PointerResult, Info))
      return false;
    return EvalPointerValueAsBool(PointerResult, Result);
  } else if (E->getType()->isAnyComplexType()) {
    APValue ComplexResult;
    if (!EvaluateComplex(E, ComplexResult, Info))
      return false;
    if (ComplexResult.isComplexFloat()) {
      Result = !ComplexResult.getComplexFloatReal().isZero() ||
               !ComplexResult.getComplexFloatImag().isZero();
    } else {
      Result = ComplexResult.getComplexIntReal().getBoolValue() ||
               ComplexResult.getComplexIntImag().getBoolValue();
    }
    return true;
  }

  return false;
}

static APSInt HandleFloatToIntCast(QualType DestType, QualType SrcType,
                                   APFloat &Value, ASTContext &Ctx) {
  unsigned DestWidth = Ctx.getIntWidth(DestType);
  // Determine whether we are converting to unsigned or signed.
  bool DestSigned = DestType->isSignedIntegerType();

  // FIXME: Warning for overflow.
  uint64_t Space[4];
  bool ignored;
  (void)Value.convertToInteger(Space, DestWidth, DestSigned,
                               llvm::APFloat::rmTowardZero, &ignored);
  return APSInt(llvm::APInt(DestWidth, 4, Space), !DestSigned);
}

static APFloat HandleFloatToFloatCast(QualType DestType, QualType SrcType,
                                      APFloat &Value, ASTContext &Ctx) {
  bool ignored;
  APFloat Result = Value;
  Result.convert(Ctx.getFloatTypeSemantics(DestType),
                 APFloat::rmNearestTiesToEven, &ignored);
  return Result;
}

static APSInt HandleIntToIntCast(QualType DestType, QualType SrcType,
                                 APSInt &Value, ASTContext &Ctx) {
  unsigned DestWidth = Ctx.getIntWidth(DestType);
  APSInt Result = Value;
  // Figure out if this is a truncate, extend or noop cast.
  // If the input is signed, do a sign extend, noop, or truncate.
  Result.extOrTrunc(DestWidth);
  Result.setIsUnsigned(DestType->isUnsignedIntegerType());
  return Result;
}

static APFloat HandleIntToFloatCast(QualType DestType, QualType SrcType,
                                    APSInt &Value, ASTContext &Ctx) {

  APFloat Result(Ctx.getFloatTypeSemantics(DestType), 1);
  Result.convertFromAPInt(Value, Value.isSigned(),
                          APFloat::rmNearestTiesToEven);
  return Result;
}

namespace {
class HasSideEffect
  : public StmtVisitor<HasSideEffect, bool> {
  EvalInfo &Info;
public:

  HasSideEffect(EvalInfo &info) : Info(info) {}

  // Unhandled nodes conservatively default to having side effects.
  bool VisitStmt(Stmt *S) {
    return true;
  }

  bool VisitParenExpr(ParenExpr *E) { return Visit(E->getSubExpr()); }
  bool VisitDeclRefExpr(DeclRefExpr *E) {
    if (Info.Ctx.getCanonicalType(E->getType()).isVolatileQualified())
      return true;
    return false;
  }
  // We don't want to evaluate BlockExprs multiple times, as they generate
  // a ton of code.
  bool VisitBlockExpr(BlockExpr *E) { return true; }
  bool VisitPredefinedExpr(PredefinedExpr *E) { return false; }
  bool VisitCompoundLiteralExpr(CompoundLiteralExpr *E)
    { return Visit(E->getInitializer()); }
  bool VisitMemberExpr(MemberExpr *E) { return Visit(E->getBase()); }
  bool VisitIntegerLiteral(IntegerLiteral *E) { return false; }
  bool VisitFloatingLiteral(FloatingLiteral *E) { return false; }
  bool VisitStringLiteral(StringLiteral *E) { return false; }
  bool VisitCharacterLiteral(CharacterLiteral *E) { return false; }
  bool VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr *E) { return false; }
  bool VisitArraySubscriptExpr(ArraySubscriptExpr *E)
    { return Visit(E->getLHS()) || Visit(E->getRHS()); }
  bool VisitChooseExpr(ChooseExpr *E)
    { return Visit(E->getChosenSubExpr(Info.Ctx)); }
  bool VisitCastExpr(CastExpr *E) { return Visit(E->getSubExpr()); }
  bool VisitBinAssign(BinaryOperator *E) { return true; }
  bool VisitCompoundAssignOperator(BinaryOperator *E) { return true; }
  bool VisitBinaryOperator(BinaryOperator *E)
  { return Visit(E->getLHS()) || Visit(E->getRHS()); }
  bool VisitUnaryPreInc(UnaryOperator *E) { return true; }
  bool VisitUnaryPostInc(UnaryOperator *E) { return true; }
  bool VisitUnaryPreDec(UnaryOperator *E) { return true; }
  bool VisitUnaryPostDec(UnaryOperator *E) { return true; }
  bool VisitUnaryDeref(UnaryOperator *E) {
    if (Info.Ctx.getCanonicalType(E->getType()).isVolatileQualified())
      return true;
    return Visit(E->getSubExpr());
  }
  bool VisitUnaryOperator(UnaryOperator *E) { return Visit(E->getSubExpr()); }
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// LValue Evaluation
//===----------------------------------------------------------------------===//
namespace {
class LValueExprEvaluator
  : public StmtVisitor<LValueExprEvaluator, APValue> {
  EvalInfo &Info;
public:

  LValueExprEvaluator(EvalInfo &info) : Info(info) {}

  APValue VisitStmt(Stmt *S) {
    return APValue();
  }

  APValue VisitParenExpr(ParenExpr *E) { return Visit(E->getSubExpr()); }
  APValue VisitDeclRefExpr(DeclRefExpr *E);
  APValue VisitPredefinedExpr(PredefinedExpr *E) { return APValue(E); }
  APValue VisitCompoundLiteralExpr(CompoundLiteralExpr *E);
  APValue VisitMemberExpr(MemberExpr *E);
  APValue VisitStringLiteral(StringLiteral *E) { return APValue(E); }
  APValue VisitObjCEncodeExpr(ObjCEncodeExpr *E) { return APValue(E); }
  APValue VisitArraySubscriptExpr(ArraySubscriptExpr *E);
  APValue VisitUnaryDeref(UnaryOperator *E);
  APValue VisitUnaryExtension(const UnaryOperator *E)
    { return Visit(E->getSubExpr()); }
  APValue VisitChooseExpr(const ChooseExpr *E)
    { return Visit(E->getChosenSubExpr(Info.Ctx)); }

  APValue VisitCastExpr(CastExpr *E) {
    switch (E->getCastKind()) {
    default:
      return APValue();

    case CastExpr::CK_NoOp:
      return Visit(E->getSubExpr());
    }
  }
  // FIXME: Missing: __real__, __imag__
};
} // end anonymous namespace

static bool EvaluateLValue(const Expr* E, APValue& Result, EvalInfo &Info) {
  Result = LValueExprEvaluator(Info).Visit(const_cast<Expr*>(E));
  return Result.isLValue();
}

APValue LValueExprEvaluator::VisitDeclRefExpr(DeclRefExpr *E) {
  if (isa<FunctionDecl>(E->getDecl())) {
    return APValue(E);
  } else if (VarDecl* VD = dyn_cast<VarDecl>(E->getDecl())) {
    if (!Info.AnyLValue && !VD->hasGlobalStorage())
      return APValue();
    if (!VD->getType()->isReferenceType())
      return APValue(E);
    // FIXME: Check whether VD might be overridden!
    const VarDecl *Def = 0;
    if (const Expr *Init = VD->getDefinition(Def))
      return Visit(const_cast<Expr *>(Init));
  }

  return APValue();
}

APValue LValueExprEvaluator::VisitCompoundLiteralExpr(CompoundLiteralExpr *E) {
  if (!Info.AnyLValue && !E->isFileScope())
    return APValue();
  return APValue(E);
}

APValue LValueExprEvaluator::VisitMemberExpr(MemberExpr *E) {
  APValue result;
  QualType Ty;
  if (E->isArrow()) {
    if (!EvaluatePointer(E->getBase(), result, Info))
      return APValue();
    Ty = E->getBase()->getType()->getAs<PointerType>()->getPointeeType();
  } else {
    result = Visit(E->getBase());
    if (result.isUninit())
      return APValue();
    Ty = E->getBase()->getType();
  }

  RecordDecl *RD = Ty->getAs<RecordType>()->getDecl();
  const ASTRecordLayout &RL = Info.Ctx.getASTRecordLayout(RD);

  FieldDecl *FD = dyn_cast<FieldDecl>(E->getMemberDecl());
  if (!FD) // FIXME: deal with other kinds of member expressions
    return APValue();

  if (FD->getType()->isReferenceType())
    return APValue();

  // FIXME: This is linear time.
  unsigned i = 0;
  for (RecordDecl::field_iterator Field = RD->field_begin(),
                               FieldEnd = RD->field_end();
       Field != FieldEnd; (void)++Field, ++i) {
    if (*Field == FD)
      break;
  }

  result.setLValue(result.getLValueBase(),
                   result.getLValueOffset() + 
                       CharUnits::fromQuantity(RL.getFieldOffset(i) / 8));

  return result;
}

APValue LValueExprEvaluator::VisitArraySubscriptExpr(ArraySubscriptExpr *E) {
  APValue Result;

  if (!EvaluatePointer(E->getBase(), Result, Info))
    return APValue();

  APSInt Index;
  if (!EvaluateInteger(E->getIdx(), Index, Info))
    return APValue();

  CharUnits ElementSize = Info.Ctx.getTypeSizeInChars(E->getType());

  CharUnits Offset = Index.getSExtValue() * ElementSize;
  Result.setLValue(Result.getLValueBase(),
                   Result.getLValueOffset() + Offset);
  return Result;
}

APValue LValueExprEvaluator::VisitUnaryDeref(UnaryOperator *E) {
  APValue Result;
  if (!EvaluatePointer(E->getSubExpr(), Result, Info))
    return APValue();
  return Result;
}

//===----------------------------------------------------------------------===//
// Pointer Evaluation
//===----------------------------------------------------------------------===//

namespace {
class PointerExprEvaluator
  : public StmtVisitor<PointerExprEvaluator, APValue> {
  EvalInfo &Info;
public:

  PointerExprEvaluator(EvalInfo &info) : Info(info) {}

  APValue VisitStmt(Stmt *S) {
    return APValue();
  }

  APValue VisitParenExpr(ParenExpr *E) { return Visit(E->getSubExpr()); }

  APValue VisitBinaryOperator(const BinaryOperator *E);
  APValue VisitCastExpr(CastExpr* E);
  APValue VisitUnaryExtension(const UnaryOperator *E)
      { return Visit(E->getSubExpr()); }
  APValue VisitUnaryAddrOf(const UnaryOperator *E);
  APValue VisitObjCStringLiteral(ObjCStringLiteral *E)
      { return APValue(E); }
  APValue VisitAddrLabelExpr(AddrLabelExpr *E)
      { return APValue(E); }
  APValue VisitCallExpr(CallExpr *E);
  APValue VisitBlockExpr(BlockExpr *E) {
    if (!E->hasBlockDeclRefExprs())
      return APValue(E);
    return APValue();
  }
  APValue VisitImplicitValueInitExpr(ImplicitValueInitExpr *E)
      { return APValue((Expr*)0); }
  APValue VisitConditionalOperator(ConditionalOperator *E);
  APValue VisitChooseExpr(ChooseExpr *E)
      { return Visit(E->getChosenSubExpr(Info.Ctx)); }
  APValue VisitCXXNullPtrLiteralExpr(CXXNullPtrLiteralExpr *E)
      { return APValue((Expr*)0); }
  // FIXME: Missing: @protocol, @selector
};
} // end anonymous namespace

static bool EvaluatePointer(const Expr* E, APValue& Result, EvalInfo &Info) {
  if (!E->getType()->hasPointerRepresentation())
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

  QualType PointeeType = PExp->getType()->getAs<PointerType>()->getPointeeType();
  CharUnits SizeOfPointee;

  // Explicitly handle GNU void* and function pointer arithmetic extensions.
  if (PointeeType->isVoidType() || PointeeType->isFunctionType())
    SizeOfPointee = CharUnits::One();
  else
    SizeOfPointee = Info.Ctx.getTypeSizeInChars(PointeeType);

  CharUnits Offset = ResultLValue.getLValueOffset();

  if (E->getOpcode() == BinaryOperator::Add)
    Offset += AdditionalOffset.getLimitedValue() * SizeOfPointee;
  else
    Offset -= AdditionalOffset.getLimitedValue() * SizeOfPointee;

  return APValue(ResultLValue.getLValueBase(), Offset);
}

APValue PointerExprEvaluator::VisitUnaryAddrOf(const UnaryOperator *E) {
  APValue result;
  if (EvaluateLValue(E->getSubExpr(), result, Info))
    return result;
  return APValue();
}


APValue PointerExprEvaluator::VisitCastExpr(CastExpr* E) {
  Expr* SubExpr = E->getSubExpr();

  switch (E->getCastKind()) {
  default:
    break;

  case CastExpr::CK_Unknown: {
    // FIXME: The handling for CK_Unknown is ugly/shouldn't be necessary!

    // Check for pointer->pointer cast
    if (SubExpr->getType()->isPointerType() ||
        SubExpr->getType()->isObjCObjectPointerType() ||
        SubExpr->getType()->isNullPtrType() ||
        SubExpr->getType()->isBlockPointerType())
      return Visit(SubExpr);

    if (SubExpr->getType()->isIntegralType()) {
      APValue Result;
      if (!EvaluateIntegerOrLValue(SubExpr, Result, Info))
        break;

      if (Result.isInt()) {
        Result.getInt().extOrTrunc((unsigned)Info.Ctx.getTypeSize(E->getType()));
        return APValue(0, 
                       CharUnits::fromQuantity(Result.getInt().getZExtValue()));
      }

      // Cast is of an lvalue, no need to change value.
      return Result;
    }
    break;
  }

  case CastExpr::CK_NoOp:
  case CastExpr::CK_BitCast:
  case CastExpr::CK_AnyPointerToObjCPointerCast:
  case CastExpr::CK_AnyPointerToBlockPointerCast:
    return Visit(SubExpr);

  case CastExpr::CK_IntegralToPointer: {
    APValue Result;
    if (!EvaluateIntegerOrLValue(SubExpr, Result, Info))
      break;

    if (Result.isInt()) {
      Result.getInt().extOrTrunc((unsigned)Info.Ctx.getTypeSize(E->getType()));
      return APValue(0, 
                     CharUnits::fromQuantity(Result.getInt().getZExtValue()));
    }

    // Cast is of an lvalue, no need to change value.
    return Result;
  }
  case CastExpr::CK_ArrayToPointerDecay:
  case CastExpr::CK_FunctionToPointerDecay: {
    APValue Result;
    if (EvaluateLValue(SubExpr, Result, Info))
      return Result;
    break;
  }
  }

  return APValue();
}

APValue PointerExprEvaluator::VisitCallExpr(CallExpr *E) {
  if (E->isBuiltinCall(Info.Ctx) ==
        Builtin::BI__builtin___CFStringMakeConstantString)
    return APValue(E);
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
// Vector Evaluation
//===----------------------------------------------------------------------===//

namespace {
  class VectorExprEvaluator
  : public StmtVisitor<VectorExprEvaluator, APValue> {
    EvalInfo &Info;
    APValue GetZeroVector(QualType VecType);
  public:

    VectorExprEvaluator(EvalInfo &info) : Info(info) {}

    APValue VisitStmt(Stmt *S) {
      return APValue();
    }

    APValue VisitParenExpr(ParenExpr *E)
        { return Visit(E->getSubExpr()); }
    APValue VisitUnaryExtension(const UnaryOperator *E)
      { return Visit(E->getSubExpr()); }
    APValue VisitUnaryPlus(const UnaryOperator *E)
      { return Visit(E->getSubExpr()); }
    APValue VisitUnaryReal(const UnaryOperator *E)
      { return Visit(E->getSubExpr()); }
    APValue VisitImplicitValueInitExpr(const ImplicitValueInitExpr *E)
      { return GetZeroVector(E->getType()); }
    APValue VisitCastExpr(const CastExpr* E);
    APValue VisitCompoundLiteralExpr(const CompoundLiteralExpr *E);
    APValue VisitInitListExpr(const InitListExpr *E);
    APValue VisitConditionalOperator(const ConditionalOperator *E);
    APValue VisitChooseExpr(const ChooseExpr *E)
      { return Visit(E->getChosenSubExpr(Info.Ctx)); }
    APValue VisitUnaryImag(const UnaryOperator *E);
    // FIXME: Missing: unary -, unary ~, binary add/sub/mul/div,
    //                 binary comparisons, binary and/or/xor,
    //                 shufflevector, ExtVectorElementExpr
    //        (Note that these require implementing conversions
    //         between vector types.)
  };
} // end anonymous namespace

static bool EvaluateVector(const Expr* E, APValue& Result, EvalInfo &Info) {
  if (!E->getType()->isVectorType())
    return false;
  Result = VectorExprEvaluator(Info).Visit(const_cast<Expr*>(E));
  return !Result.isUninit();
}

APValue VectorExprEvaluator::VisitCastExpr(const CastExpr* E) {
  const VectorType *VTy = E->getType()->getAs<VectorType>();
  QualType EltTy = VTy->getElementType();
  unsigned NElts = VTy->getNumElements();
  unsigned EltWidth = Info.Ctx.getTypeSize(EltTy);

  const Expr* SE = E->getSubExpr();
  QualType SETy = SE->getType();
  APValue Result = APValue();

  // Check for vector->vector bitcast and scalar->vector splat.
  if (SETy->isVectorType()) {
    return this->Visit(const_cast<Expr*>(SE));
  } else if (SETy->isIntegerType()) {
    APSInt IntResult;
    if (!EvaluateInteger(SE, IntResult, Info))
      return APValue();
    Result = APValue(IntResult);
  } else if (SETy->isRealFloatingType()) {
    APFloat F(0.0);
    if (!EvaluateFloat(SE, F, Info))
      return APValue();
    Result = APValue(F);
  } else
    return APValue();

  // For casts of a scalar to ExtVector, convert the scalar to the element type
  // and splat it to all elements.
  if (E->getType()->isExtVectorType()) {
    if (EltTy->isIntegerType() && Result.isInt())
      Result = APValue(HandleIntToIntCast(EltTy, SETy, Result.getInt(),
                                          Info.Ctx));
    else if (EltTy->isIntegerType())
      Result = APValue(HandleFloatToIntCast(EltTy, SETy, Result.getFloat(),
                                            Info.Ctx));
    else if (EltTy->isRealFloatingType() && Result.isInt())
      Result = APValue(HandleIntToFloatCast(EltTy, SETy, Result.getInt(),
                                            Info.Ctx));
    else if (EltTy->isRealFloatingType())
      Result = APValue(HandleFloatToFloatCast(EltTy, SETy, Result.getFloat(),
                                              Info.Ctx));
    else
      return APValue();

    // Splat and create vector APValue.
    llvm::SmallVector<APValue, 4> Elts(NElts, Result);
    return APValue(&Elts[0], Elts.size());
  }

  // For casts of a scalar to regular gcc-style vector type, bitcast the scalar
  // to the vector. To construct the APValue vector initializer, bitcast the
  // initializing value to an APInt, and shift out the bits pertaining to each
  // element.
  APSInt Init;
  Init = Result.isInt() ? Result.getInt() : Result.getFloat().bitcastToAPInt();

  llvm::SmallVector<APValue, 4> Elts;
  for (unsigned i = 0; i != NElts; ++i) {
    APSInt Tmp = Init;
    Tmp.extOrTrunc(EltWidth);

    if (EltTy->isIntegerType())
      Elts.push_back(APValue(Tmp));
    else if (EltTy->isRealFloatingType())
      Elts.push_back(APValue(APFloat(Tmp)));
    else
      return APValue();

    Init >>= EltWidth;
  }
  return APValue(&Elts[0], Elts.size());
}

APValue
VectorExprEvaluator::VisitCompoundLiteralExpr(const CompoundLiteralExpr *E) {
  return this->Visit(const_cast<Expr*>(E->getInitializer()));
}

APValue
VectorExprEvaluator::VisitInitListExpr(const InitListExpr *E) {
  const VectorType *VT = E->getType()->getAs<VectorType>();
  unsigned NumInits = E->getNumInits();
  unsigned NumElements = VT->getNumElements();

  QualType EltTy = VT->getElementType();
  llvm::SmallVector<APValue, 4> Elements;

  for (unsigned i = 0; i < NumElements; i++) {
    if (EltTy->isIntegerType()) {
      llvm::APSInt sInt(32);
      if (i < NumInits) {
        if (!EvaluateInteger(E->getInit(i), sInt, Info))
          return APValue();
      } else {
        sInt = Info.Ctx.MakeIntValue(0, EltTy);
      }
      Elements.push_back(APValue(sInt));
    } else {
      llvm::APFloat f(0.0);
      if (i < NumInits) {
        if (!EvaluateFloat(E->getInit(i), f, Info))
          return APValue();
      } else {
        f = APFloat::getZero(Info.Ctx.getFloatTypeSemantics(EltTy));
      }
      Elements.push_back(APValue(f));
    }
  }
  return APValue(&Elements[0], Elements.size());
}

APValue
VectorExprEvaluator::GetZeroVector(QualType T) {
  const VectorType *VT = T->getAs<VectorType>();
  QualType EltTy = VT->getElementType();
  APValue ZeroElement;
  if (EltTy->isIntegerType())
    ZeroElement = APValue(Info.Ctx.MakeIntValue(0, EltTy));
  else
    ZeroElement =
        APValue(APFloat::getZero(Info.Ctx.getFloatTypeSemantics(EltTy)));

  llvm::SmallVector<APValue, 4> Elements(VT->getNumElements(), ZeroElement);
  return APValue(&Elements[0], Elements.size());
}

APValue VectorExprEvaluator::VisitConditionalOperator(const ConditionalOperator *E) {
  bool BoolResult;
  if (!HandleConversionToBool(E->getCond(), BoolResult, Info))
    return APValue();

  Expr* EvalExpr = BoolResult ? E->getTrueExpr() : E->getFalseExpr();

  APValue Result;
  if (EvaluateVector(EvalExpr, Result, Info))
    return Result;
  return APValue();
}

APValue VectorExprEvaluator::VisitUnaryImag(const UnaryOperator *E) {
  if (!E->getSubExpr()->isEvaluatable(Info.Ctx))
    Info.EvalResult.HasSideEffects = true;
  return GetZeroVector(E->getType());
}

//===----------------------------------------------------------------------===//
// Integer Evaluation
//===----------------------------------------------------------------------===//

namespace {
class IntExprEvaluator
  : public StmtVisitor<IntExprEvaluator, bool> {
  EvalInfo &Info;
  APValue &Result;
public:
  IntExprEvaluator(EvalInfo &info, APValue &result)
    : Info(info), Result(result) {}

  bool Success(const llvm::APSInt &SI, const Expr *E) {
    assert(E->getType()->isIntegralType() && "Invalid evaluation result.");
    assert(SI.isSigned() == E->getType()->isSignedIntegerType() &&
           "Invalid evaluation result.");
    assert(SI.getBitWidth() == Info.Ctx.getIntWidth(E->getType()) &&
           "Invalid evaluation result.");
    Result = APValue(SI);
    return true;
  }

  bool Success(const llvm::APInt &I, const Expr *E) {
    assert(E->getType()->isIntegralType() && "Invalid evaluation result.");
    assert(I.getBitWidth() == Info.Ctx.getIntWidth(E->getType()) &&
           "Invalid evaluation result.");
    Result = APValue(APSInt(I));
    Result.getInt().setIsUnsigned(E->getType()->isUnsignedIntegerType());
    return true;
  }

  bool Success(uint64_t Value, const Expr *E) {
    assert(E->getType()->isIntegralType() && "Invalid evaluation result.");
    Result = APValue(Info.Ctx.MakeIntValue(Value, E->getType()));
    return true;
  }

  bool Error(SourceLocation L, diag::kind D, const Expr *E) {
    // Take the first error.
    if (Info.EvalResult.Diag == 0) {
      Info.EvalResult.DiagLoc = L;
      Info.EvalResult.Diag = D;
      Info.EvalResult.DiagExpr = E;
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
    return Error(E->getLocStart(), diag::note_invalid_subexpr_in_ice, E);
  }

  bool VisitParenExpr(ParenExpr *E) { return Visit(E->getSubExpr()); }

  bool VisitIntegerLiteral(const IntegerLiteral *E) {
    return Success(E->getValue(), E);
  }
  bool VisitCharacterLiteral(const CharacterLiteral *E) {
    return Success(E->getValue(), E);
  }
  bool VisitTypesCompatibleExpr(const TypesCompatibleExpr *E) {
    // Per gcc docs "this built-in function ignores top level
    // qualifiers".  We need to use the canonical version to properly
    // be able to strip CRV qualifiers from the type.
    QualType T0 = Info.Ctx.getCanonicalType(E->getArgType1());
    QualType T1 = Info.Ctx.getCanonicalType(E->getArgType2());
    return Success(Info.Ctx.typesAreCompatible(T0.getUnqualifiedType(),
                                               T1.getUnqualifiedType()),
                   E);
  }

  bool CheckReferencedDecl(const Expr *E, const Decl *D);
  bool VisitDeclRefExpr(const DeclRefExpr *E) {
    return CheckReferencedDecl(E, E->getDecl());
  }
  bool VisitMemberExpr(const MemberExpr *E) {
    if (CheckReferencedDecl(E, E->getMemberDecl())) {
      // Conservatively assume a MemberExpr will have side-effects
      Info.EvalResult.HasSideEffects = true;
      return true;
    }
    return false;
  }

  bool VisitCallExpr(const CallExpr *E);
  bool VisitBinaryOperator(const BinaryOperator *E);
  bool VisitUnaryOperator(const UnaryOperator *E);
  bool VisitConditionalOperator(const ConditionalOperator *E);

  bool VisitCastExpr(CastExpr* E);
  bool VisitSizeOfAlignOfExpr(const SizeOfAlignOfExpr *E);

  bool VisitCXXBoolLiteralExpr(const CXXBoolLiteralExpr *E) {
    return Success(E->getValue(), E);
  }

  bool VisitGNUNullExpr(const GNUNullExpr *E) {
    return Success(0, E);
  }

  bool VisitCXXZeroInitValueExpr(const CXXZeroInitValueExpr *E) {
    return Success(0, E);
  }

  bool VisitImplicitValueInitExpr(const ImplicitValueInitExpr *E) {
    return Success(0, E);
  }

  bool VisitUnaryTypeTraitExpr(const UnaryTypeTraitExpr *E) {
    return Success(E->EvaluateTrait(Info.Ctx), E);
  }

  bool VisitChooseExpr(const ChooseExpr *E) {
    return Visit(E->getChosenSubExpr(Info.Ctx));
  }

  bool VisitUnaryReal(const UnaryOperator *E);
  bool VisitUnaryImag(const UnaryOperator *E);

private:
  unsigned GetAlignOfExpr(const Expr *E);
  unsigned GetAlignOfType(QualType T);
  // FIXME: Missing: array subscript of vector, member of vector
};
} // end anonymous namespace

static bool EvaluateIntegerOrLValue(const Expr* E, APValue &Result, EvalInfo &Info) {
  if (!E->getType()->isIntegralType())
    return false;

  return IntExprEvaluator(Info, Result).Visit(const_cast<Expr*>(E));
}

static bool EvaluateInteger(const Expr* E, APSInt &Result, EvalInfo &Info) {
  APValue Val;
  if (!EvaluateIntegerOrLValue(E, Val, Info) || !Val.isInt())
    return false;
  Result = Val.getInt();
  return true;
}

bool IntExprEvaluator::CheckReferencedDecl(const Expr* E, const Decl* D) {
  // Enums are integer constant exprs.
  if (const EnumConstantDecl *ECD = dyn_cast<EnumConstantDecl>(D))
    return Success(ECD->getInitVal(), E);

  // In C++, const, non-volatile integers initialized with ICEs are ICEs.
  // In C, they can also be folded, although they are not ICEs.
  if (Info.Ctx.getCanonicalType(E->getType()).getCVRQualifiers() 
                                                        == Qualifiers::Const) {
    if (const VarDecl *VD = dyn_cast<VarDecl>(D)) {
      const VarDecl *Def = 0;
      if (const Expr *Init = VD->getDefinition(Def)) {
        if (APValue *V = VD->getEvaluatedValue()) {
          if (V->isInt())
            return Success(V->getInt(), E);
          return Error(E->getLocStart(), diag::note_invalid_subexpr_in_ice, E);
        }

        if (VD->isEvaluatingValue())
          return Error(E->getLocStart(), diag::note_invalid_subexpr_in_ice, E);

        VD->setEvaluatingValue();

        if (Visit(const_cast<Expr*>(Init))) {
          // Cache the evaluated value in the variable declaration.
          VD->setEvaluatedValue(Result);
          return true;
        }

        VD->setEvaluatedValue(APValue());
        return false;
      }
    }
  }

  // Otherwise, random variable references are not constants.
  return Error(E->getLocStart(), diag::note_invalid_subexpr_in_ice, E);
}

/// EvaluateBuiltinClassifyType - Evaluate __builtin_classify_type the same way
/// as GCC.
static int EvaluateBuiltinClassifyType(const CallExpr *E) {
  // The following enum mimics the values returned by GCC.
  // FIXME: Does GCC differ between lvalue and rvalue references here?
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
  switch (E->isBuiltinCall(Info.Ctx)) {
  default:
    return Error(E->getLocStart(), diag::note_invalid_subexpr_in_ice, E);

  case Builtin::BI__builtin_object_size: {
    const Expr *Arg = E->getArg(0)->IgnoreParens();
    Expr::EvalResult Base;
    
    // TODO: Perhaps we should let LLVM lower this?
    if (Arg->EvaluateAsAny(Base, Info.Ctx)
        && Base.Val.getKind() == APValue::LValue
        && !Base.HasSideEffects)
      if (const Expr *LVBase = Base.Val.getLValueBase())
        if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(LVBase)) {
          if (const VarDecl *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
            if (!VD->getType()->isIncompleteType()
                && VD->getType()->isObjectType()
                && !VD->getType()->isVariablyModifiedType()
                && !VD->getType()->isDependentType()) {
              CharUnits Size = Info.Ctx.getTypeSizeInChars(VD->getType());
              CharUnits Offset = Base.Val.getLValueOffset();
              if (!Offset.isNegative() && Offset <= Size)
                Size -= Offset;
              else
                Size = CharUnits::Zero();
              return Success(Size.getQuantity(), E);
            }
          }
        }

    // If evaluating the argument has side-effects we can't determine
    // the size of the object and lower it to unknown now.
    if (E->getArg(0)->HasSideEffects(Info.Ctx)) {
      if (E->getArg(1)->EvaluateAsInt(Info.Ctx).getZExtValue() <= 1)
        return Success(-1ULL, E);
      return Success(0, E);
    }

    return Error(E->getLocStart(), diag::note_invalid_subexpr_in_ice, E);
  }

  case Builtin::BI__builtin_classify_type:
    return Success(EvaluateBuiltinClassifyType(E), E);

  case Builtin::BI__builtin_constant_p:
    // __builtin_constant_p always has one operand: it returns true if that
    // operand can be folded, false otherwise.
    return Success(E->getArg(0)->isEvaluatable(Info.Ctx), E);
      
  case Builtin::BI__builtin_eh_return_data_regno: {
    int Operand = E->getArg(0)->EvaluateAsInt(Info.Ctx).getZExtValue();
    Operand = Info.Ctx.Target.getEHDataRegisterNumber(Operand);
    return Success(Operand, E);
  }
  }
}

bool IntExprEvaluator::VisitBinaryOperator(const BinaryOperator *E) {
  if (E->getOpcode() == BinaryOperator::Comma) {
    if (!Visit(E->getRHS()))
      return false;

    // If we can't evaluate the LHS, it might have side effects;
    // conservatively mark it.
    if (!E->getLHS()->isEvaluatable(Info.Ctx))
      Info.EvalResult.HasSideEffects = true;

    return true;
  }

  if (E->isLogicalOp()) {
    // These need to be handled specially because the operands aren't
    // necessarily integral
    bool lhsResult, rhsResult;

    if (HandleConversionToBool(E->getLHS(), lhsResult, Info)) {
      // We were able to evaluate the LHS, see if we can get away with not
      // evaluating the RHS: 0 && X -> 0, 1 || X -> 1
      if (lhsResult == (E->getOpcode() == BinaryOperator::LOr))
        return Success(lhsResult, E);

      if (HandleConversionToBool(E->getRHS(), rhsResult, Info)) {
        if (E->getOpcode() == BinaryOperator::LOr)
          return Success(lhsResult || rhsResult, E);
        else
          return Success(lhsResult && rhsResult, E);
      }
    } else {
      if (HandleConversionToBool(E->getRHS(), rhsResult, Info)) {
        // We can't evaluate the LHS; however, sometimes the result
        // is determined by the RHS: X && 0 -> 0, X || 1 -> 1.
        if (rhsResult == (E->getOpcode() == BinaryOperator::LOr) ||
            !rhsResult == (E->getOpcode() == BinaryOperator::LAnd)) {
          // Since we weren't able to evaluate the left hand side, it
          // must have had side effects.
          Info.EvalResult.HasSideEffects = true;

          return Success(rhsResult, E);
        }
      }
    }

    return false;
  }

  QualType LHSTy = E->getLHS()->getType();
  QualType RHSTy = E->getRHS()->getType();

  if (LHSTy->isAnyComplexType()) {
    assert(RHSTy->isAnyComplexType() && "Invalid comparison");
    APValue LHS, RHS;

    if (!EvaluateComplex(E->getLHS(), LHS, Info))
      return false;

    if (!EvaluateComplex(E->getRHS(), RHS, Info))
      return false;

    if (LHS.isComplexFloat()) {
      APFloat::cmpResult CR_r =
        LHS.getComplexFloatReal().compare(RHS.getComplexFloatReal());
      APFloat::cmpResult CR_i =
        LHS.getComplexFloatImag().compare(RHS.getComplexFloatImag());

      if (E->getOpcode() == BinaryOperator::EQ)
        return Success((CR_r == APFloat::cmpEqual &&
                        CR_i == APFloat::cmpEqual), E);
      else {
        assert(E->getOpcode() == BinaryOperator::NE &&
               "Invalid complex comparison.");
        return Success(((CR_r == APFloat::cmpGreaterThan ||
                         CR_r == APFloat::cmpLessThan) &&
                        (CR_i == APFloat::cmpGreaterThan ||
                         CR_i == APFloat::cmpLessThan)), E);
      }
    } else {
      if (E->getOpcode() == BinaryOperator::EQ)
        return Success((LHS.getComplexIntReal() == RHS.getComplexIntReal() &&
                        LHS.getComplexIntImag() == RHS.getComplexIntImag()), E);
      else {
        assert(E->getOpcode() == BinaryOperator::NE &&
               "Invalid compex comparison.");
        return Success((LHS.getComplexIntReal() != RHS.getComplexIntReal() ||
                        LHS.getComplexIntImag() != RHS.getComplexIntImag()), E);
      }
    }
  }

  if (LHSTy->isRealFloatingType() &&
      RHSTy->isRealFloatingType()) {
    APFloat RHS(0.0), LHS(0.0);

    if (!EvaluateFloat(E->getRHS(), RHS, Info))
      return false;

    if (!EvaluateFloat(E->getLHS(), LHS, Info))
      return false;

    APFloat::cmpResult CR = LHS.compare(RHS);

    switch (E->getOpcode()) {
    default:
      assert(0 && "Invalid binary operator!");
    case BinaryOperator::LT:
      return Success(CR == APFloat::cmpLessThan, E);
    case BinaryOperator::GT:
      return Success(CR == APFloat::cmpGreaterThan, E);
    case BinaryOperator::LE:
      return Success(CR == APFloat::cmpLessThan || CR == APFloat::cmpEqual, E);
    case BinaryOperator::GE:
      return Success(CR == APFloat::cmpGreaterThan || CR == APFloat::cmpEqual,
                     E);
    case BinaryOperator::EQ:
      return Success(CR == APFloat::cmpEqual, E);
    case BinaryOperator::NE:
      return Success(CR == APFloat::cmpGreaterThan
                     || CR == APFloat::cmpLessThan, E);
    }
  }

  if (LHSTy->isPointerType() && RHSTy->isPointerType()) {
    if (E->getOpcode() == BinaryOperator::Sub || E->isEqualityOp()) {
      APValue LHSValue;
      if (!EvaluatePointer(E->getLHS(), LHSValue, Info))
        return false;

      APValue RHSValue;
      if (!EvaluatePointer(E->getRHS(), RHSValue, Info))
        return false;

      // Reject any bases from the normal codepath; we special-case comparisons
      // to null.
      if (LHSValue.getLValueBase()) {
        if (!E->isEqualityOp())
          return false;
        if (RHSValue.getLValueBase() || !RHSValue.getLValueOffset().isZero())
          return false;
        bool bres;
        if (!EvalPointerValueAsBool(LHSValue, bres))
          return false;
        return Success(bres ^ (E->getOpcode() == BinaryOperator::EQ), E);
      } else if (RHSValue.getLValueBase()) {
        if (!E->isEqualityOp())
          return false;
        if (LHSValue.getLValueBase() || !LHSValue.getLValueOffset().isZero())
          return false;
        bool bres;
        if (!EvalPointerValueAsBool(RHSValue, bres))
          return false;
        return Success(bres ^ (E->getOpcode() == BinaryOperator::EQ), E);
      }

      if (E->getOpcode() == BinaryOperator::Sub) {
        const QualType Type = E->getLHS()->getType();
        const QualType ElementType = Type->getAs<PointerType>()->getPointeeType();

        CharUnits ElementSize = CharUnits::One();
        if (!ElementType->isVoidType() && !ElementType->isFunctionType())
          ElementSize = Info.Ctx.getTypeSizeInChars(ElementType);

        CharUnits Diff = LHSValue.getLValueOffset() - 
                             RHSValue.getLValueOffset();
        return Success(Diff / ElementSize, E);
      }
      bool Result;
      if (E->getOpcode() == BinaryOperator::EQ) {
        Result = LHSValue.getLValueOffset() == RHSValue.getLValueOffset();
      } else {
        Result = LHSValue.getLValueOffset() != RHSValue.getLValueOffset();
      }
      return Success(Result, E);
    }
  }
  if (!LHSTy->isIntegralType() ||
      !RHSTy->isIntegralType()) {
    // We can't continue from here for non-integral types, and they
    // could potentially confuse the following operations.
    return false;
  }

  // The LHS of a constant expr is always evaluated and needed.
  if (!Visit(E->getLHS()))
    return false; // error in subexpression.

  APValue RHSVal;
  if (!EvaluateIntegerOrLValue(E->getRHS(), RHSVal, Info))
    return false;

  // Handle cases like (unsigned long)&a + 4.
  if (E->isAdditiveOp() && Result.isLValue() && RHSVal.isInt()) {
    CharUnits Offset = Result.getLValueOffset();
    CharUnits AdditionalOffset = CharUnits::fromQuantity(
                                     RHSVal.getInt().getZExtValue());
    if (E->getOpcode() == BinaryOperator::Add)
      Offset += AdditionalOffset;
    else
      Offset -= AdditionalOffset;
    Result = APValue(Result.getLValueBase(), Offset);
    return true;
  }

  // Handle cases like 4 + (unsigned long)&a
  if (E->getOpcode() == BinaryOperator::Add &&
        RHSVal.isLValue() && Result.isInt()) {
    CharUnits Offset = RHSVal.getLValueOffset();
    Offset += CharUnits::fromQuantity(Result.getInt().getZExtValue());
    Result = APValue(RHSVal.getLValueBase(), Offset);
    return true;
  }

  // All the following cases expect both operands to be an integer
  if (!Result.isInt() || !RHSVal.isInt())
    return false;

  APSInt& RHS = RHSVal.getInt();

  switch (E->getOpcode()) {
  default:
    return Error(E->getOperatorLoc(), diag::note_invalid_subexpr_in_ice, E);
  case BinaryOperator::Mul: return Success(Result.getInt() * RHS, E);
  case BinaryOperator::Add: return Success(Result.getInt() + RHS, E);
  case BinaryOperator::Sub: return Success(Result.getInt() - RHS, E);
  case BinaryOperator::And: return Success(Result.getInt() & RHS, E);
  case BinaryOperator::Xor: return Success(Result.getInt() ^ RHS, E);
  case BinaryOperator::Or:  return Success(Result.getInt() | RHS, E);
  case BinaryOperator::Div:
    if (RHS == 0)
      return Error(E->getOperatorLoc(), diag::note_expr_divide_by_zero, E);
    return Success(Result.getInt() / RHS, E);
  case BinaryOperator::Rem:
    if (RHS == 0)
      return Error(E->getOperatorLoc(), diag::note_expr_divide_by_zero, E);
    return Success(Result.getInt() % RHS, E);
  case BinaryOperator::Shl: {
    // FIXME: Warn about out of range shift amounts!
    unsigned SA =
      (unsigned) RHS.getLimitedValue(Result.getInt().getBitWidth()-1);
    return Success(Result.getInt() << SA, E);
  }
  case BinaryOperator::Shr: {
    unsigned SA =
      (unsigned) RHS.getLimitedValue(Result.getInt().getBitWidth()-1);
    return Success(Result.getInt() >> SA, E);
  }

  case BinaryOperator::LT: return Success(Result.getInt() < RHS, E);
  case BinaryOperator::GT: return Success(Result.getInt() > RHS, E);
  case BinaryOperator::LE: return Success(Result.getInt() <= RHS, E);
  case BinaryOperator::GE: return Success(Result.getInt() >= RHS, E);
  case BinaryOperator::EQ: return Success(Result.getInt() == RHS, E);
  case BinaryOperator::NE: return Success(Result.getInt() != RHS, E);
  }
}

bool IntExprEvaluator::VisitConditionalOperator(const ConditionalOperator *E) {
  bool Cond;
  if (!HandleConversionToBool(E->getCond(), Cond, Info))
    return false;

  return Visit(Cond ? E->getTrueExpr() : E->getFalseExpr());
}

unsigned IntExprEvaluator::GetAlignOfType(QualType T) {
  // C++ [expr.sizeof]p2: "When applied to a reference or a reference type,
  //   the result is the size of the referenced type."
  // C++ [expr.alignof]p3: "When alignof is applied to a reference type, the
  //   result shall be the alignment of the referenced type."
  if (const ReferenceType *Ref = T->getAs<ReferenceType>())
    T = Ref->getPointeeType();

  // Get information about the alignment.
  unsigned CharSize = Info.Ctx.Target.getCharWidth();

  // __alignof is defined to return the preferred alignment.
  return Info.Ctx.getPreferredTypeAlign(T.getTypePtr()) / CharSize;
}

unsigned IntExprEvaluator::GetAlignOfExpr(const Expr *E) {
  E = E->IgnoreParens();

  // alignof decl is always accepted, even if it doesn't make sense: we default
  // to 1 in those cases.
  if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E))
    return Info.Ctx.getDeclAlignInBytes(DRE->getDecl(), /*RefAsPointee*/true);

  if (const MemberExpr *ME = dyn_cast<MemberExpr>(E))
    return Info.Ctx.getDeclAlignInBytes(ME->getMemberDecl(),
                                        /*RefAsPointee*/true);

  return GetAlignOfType(E->getType());
}


/// VisitSizeAlignOfExpr - Evaluate a sizeof or alignof with a result as the
/// expression's type.
bool IntExprEvaluator::VisitSizeOfAlignOfExpr(const SizeOfAlignOfExpr *E) {
  // Handle alignof separately.
  if (!E->isSizeOf()) {
    if (E->isArgumentType())
      return Success(GetAlignOfType(E->getArgumentType()), E);
    else
      return Success(GetAlignOfExpr(E->getArgumentExpr()), E);
  }

  QualType SrcTy = E->getTypeOfArgument();
  // C++ [expr.sizeof]p2: "When applied to a reference or a reference type,
  //   the result is the size of the referenced type."
  // C++ [expr.alignof]p3: "When alignof is applied to a reference type, the
  //   result shall be the alignment of the referenced type."
  if (const ReferenceType *Ref = SrcTy->getAs<ReferenceType>())
    SrcTy = Ref->getPointeeType();

  // sizeof(void), __alignof__(void), sizeof(function) = 1 as a gcc
  // extension.
  if (SrcTy->isVoidType() || SrcTy->isFunctionType())
    return Success(1, E);

  // sizeof(vla) is not a constantexpr: C99 6.5.3.4p2.
  if (!SrcTy->isConstantSizeType())
    return false;

  // Get information about the size.
  return Success(Info.Ctx.getTypeSizeInChars(SrcTy).getQuantity(), E);
}

bool IntExprEvaluator::VisitUnaryOperator(const UnaryOperator *E) {
  // Special case unary operators that do not need their subexpression
  // evaluated.  offsetof/sizeof/alignof are all special.
  if (E->isOffsetOfOp()) {
    // The AST for offsetof is defined in such a way that we can just
    // directly Evaluate it as an l-value.
    APValue LV;
    if (!EvaluateLValue(E->getSubExpr(), LV, Info))
      return false;
    if (LV.getLValueBase())
      return false;
    return Success(LV.getLValueOffset().getQuantity(), E);
  }

  if (E->getOpcode() == UnaryOperator::LNot) {
    // LNot's operand isn't necessarily an integer, so we handle it specially.
    bool bres;
    if (!HandleConversionToBool(E->getSubExpr(), bres, Info))
      return false;
    return Success(!bres, E);
  }

  // Only handle integral operations...
  if (!E->getSubExpr()->getType()->isIntegralType())
    return false;

  // Get the operand value into 'Result'.
  if (!Visit(E->getSubExpr()))
    return false;

  switch (E->getOpcode()) {
  default:
    // Address, indirect, pre/post inc/dec, etc are not valid constant exprs.
    // See C99 6.6p3.
    return Error(E->getOperatorLoc(), diag::note_invalid_subexpr_in_ice, E);
  case UnaryOperator::Extension:
    // FIXME: Should extension allow i-c-e extension expressions in its scope?
    // If so, we could clear the diagnostic ID.
    return true;
  case UnaryOperator::Plus:
    // The result is always just the subexpr.
    return true;
  case UnaryOperator::Minus:
    if (!Result.isInt()) return false;
    return Success(-Result.getInt(), E);
  case UnaryOperator::Not:
    if (!Result.isInt()) return false;
    return Success(~Result.getInt(), E);
  }
}

/// HandleCast - This is used to evaluate implicit or explicit casts where the
/// result type is integer.
bool IntExprEvaluator::VisitCastExpr(CastExpr *E) {
  Expr *SubExpr = E->getSubExpr();
  QualType DestType = E->getType();
  QualType SrcType = SubExpr->getType();

  if (DestType->isBooleanType()) {
    bool BoolResult;
    if (!HandleConversionToBool(SubExpr, BoolResult, Info))
      return false;
    return Success(BoolResult, E);
  }

  // Handle simple integer->integer casts.
  if (SrcType->isIntegralType()) {
    if (!Visit(SubExpr))
      return false;

    if (!Result.isInt()) {
      // Only allow casts of lvalues if they are lossless.
      return Info.Ctx.getTypeSize(DestType) == Info.Ctx.getTypeSize(SrcType);
    }

    return Success(HandleIntToIntCast(DestType, SrcType,
                                      Result.getInt(), Info.Ctx), E);
  }

  // FIXME: Clean this up!
  if (SrcType->isPointerType()) {
    APValue LV;
    if (!EvaluatePointer(SubExpr, LV, Info))
      return false;

    if (LV.getLValueBase()) {
      // Only allow based lvalue casts if they are lossless.
      if (Info.Ctx.getTypeSize(DestType) != Info.Ctx.getTypeSize(SrcType))
        return false;

      Result = LV;
      return true;
    }

    APSInt AsInt = Info.Ctx.MakeIntValue(LV.getLValueOffset().getQuantity(), 
                                         SrcType);
    return Success(HandleIntToIntCast(DestType, SrcType, AsInt, Info.Ctx), E);
  }

  if (SrcType->isArrayType() || SrcType->isFunctionType()) {
    // This handles double-conversion cases, where there's both
    // an l-value promotion and an implicit conversion to int.
    APValue LV;
    if (!EvaluateLValue(SubExpr, LV, Info))
      return false;

    if (Info.Ctx.getTypeSize(DestType) != Info.Ctx.getTypeSize(Info.Ctx.VoidPtrTy))
      return false;

    Result = LV;
    return true;
  }

  if (SrcType->isAnyComplexType()) {
    APValue C;
    if (!EvaluateComplex(SubExpr, C, Info))
      return false;
    if (C.isComplexFloat())
      return Success(HandleFloatToIntCast(DestType, SrcType,
                                          C.getComplexFloatReal(), Info.Ctx),
                     E);
    else
      return Success(HandleIntToIntCast(DestType, SrcType,
                                        C.getComplexIntReal(), Info.Ctx), E);
  }
  // FIXME: Handle vectors

  if (!SrcType->isRealFloatingType())
    return Error(E->getExprLoc(), diag::note_invalid_subexpr_in_ice, E);

  APFloat F(0.0);
  if (!EvaluateFloat(SubExpr, F, Info))
    return Error(E->getExprLoc(), diag::note_invalid_subexpr_in_ice, E);

  return Success(HandleFloatToIntCast(DestType, SrcType, F, Info.Ctx), E);
}

bool IntExprEvaluator::VisitUnaryReal(const UnaryOperator *E) {
  if (E->getSubExpr()->getType()->isAnyComplexType()) {
    APValue LV;
    if (!EvaluateComplex(E->getSubExpr(), LV, Info) || !LV.isComplexInt())
      return Error(E->getExprLoc(), diag::note_invalid_subexpr_in_ice, E);
    return Success(LV.getComplexIntReal(), E);
  }

  return Visit(E->getSubExpr());
}

bool IntExprEvaluator::VisitUnaryImag(const UnaryOperator *E) {
  if (E->getSubExpr()->getType()->isComplexIntegerType()) {
    APValue LV;
    if (!EvaluateComplex(E->getSubExpr(), LV, Info) || !LV.isComplexInt())
      return Error(E->getExprLoc(), diag::note_invalid_subexpr_in_ice, E);
    return Success(LV.getComplexIntImag(), E);
  }

  if (!E->getSubExpr()->isEvaluatable(Info.Ctx))
    Info.EvalResult.HasSideEffects = true;
  return Success(0, E);
}

//===----------------------------------------------------------------------===//
// Float Evaluation
//===----------------------------------------------------------------------===//

namespace {
class FloatExprEvaluator
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
  bool VisitConditionalOperator(ConditionalOperator *E);

  bool VisitChooseExpr(const ChooseExpr *E)
    { return Visit(E->getChosenSubExpr(Info.Ctx)); }
  bool VisitUnaryExtension(const UnaryOperator *E)
    { return Visit(E->getSubExpr()); }

  // FIXME: Missing: __real__/__imag__, array subscript of vector,
  //                 member of vector, ImplicitValueInitExpr
};
} // end anonymous namespace

static bool EvaluateFloat(const Expr* E, APFloat& Result, EvalInfo &Info) {
  return FloatExprEvaluator(Info, Result).Visit(const_cast<Expr*>(E));
}

bool FloatExprEvaluator::VisitCallExpr(const CallExpr *E) {
  switch (E->isBuiltinCall(Info.Ctx)) {
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
    // If this is __builtin_nan() turn this into a nan, otherwise we
    // can't constant fold it.
    if (const StringLiteral *S =
        dyn_cast<StringLiteral>(E->getArg(0)->IgnoreParenCasts())) {
      if (!S->isWide()) {
        const llvm::fltSemantics &Sem =
          Info.Ctx.getFloatTypeSemantics(E->getType());
        unsigned Type = 0;
        if (!S->getString().empty() && S->getString().getAsInteger(0, Type))
          return false;
        Result = llvm::APFloat::getNaN(Sem, false, Type);
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
  if (E->getOpcode() == UnaryOperator::Deref)
    return false;

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
  if (E->getOpcode() == BinaryOperator::Comma) {
    if (!EvaluateFloat(E->getRHS(), Result, Info))
      return false;

    // If we can't evaluate the LHS, it might have side effects;
    // conservatively mark it.
    if (!E->getLHS()->isEvaluatable(Info.Ctx))
      Info.EvalResult.HasSideEffects = true;

    return true;
  }

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
  }
}

bool FloatExprEvaluator::VisitFloatingLiteral(const FloatingLiteral *E) {
  Result = E->getValue();
  return true;
}

bool FloatExprEvaluator::VisitCastExpr(CastExpr *E) {
  Expr* SubExpr = E->getSubExpr();

  if (SubExpr->getType()->isIntegralType()) {
    APSInt IntResult;
    if (!EvaluateInteger(SubExpr, IntResult, Info))
      return false;
    Result = HandleIntToFloatCast(E->getType(), SubExpr->getType(),
                                  IntResult, Info.Ctx);
    return true;
  }
  if (SubExpr->getType()->isRealFloatingType()) {
    if (!Visit(SubExpr))
      return false;
    Result = HandleFloatToFloatCast(E->getType(), SubExpr->getType(),
                                    Result, Info.Ctx);
    return true;
  }
  // FIXME: Handle complex types

  return false;
}

bool FloatExprEvaluator::VisitCXXZeroInitValueExpr(CXXZeroInitValueExpr *E) {
  Result = APFloat::getZero(Info.Ctx.getFloatTypeSemantics(E->getType()));
  return true;
}

bool FloatExprEvaluator::VisitConditionalOperator(ConditionalOperator *E) {
  bool Cond;
  if (!HandleConversionToBool(E->getCond(), Cond, Info))
    return false;

  return Visit(Cond ? E->getTrueExpr() : E->getFalseExpr());
}

//===----------------------------------------------------------------------===//
// Complex Evaluation (for float and integer)
//===----------------------------------------------------------------------===//

namespace {
class ComplexExprEvaluator
  : public StmtVisitor<ComplexExprEvaluator, APValue> {
  EvalInfo &Info;

public:
  ComplexExprEvaluator(EvalInfo &info) : Info(info) {}

  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//

  APValue VisitStmt(Stmt *S) {
    return APValue();
  }

  APValue VisitParenExpr(ParenExpr *E) { return Visit(E->getSubExpr()); }

  APValue VisitImaginaryLiteral(ImaginaryLiteral *E) {
    Expr* SubExpr = E->getSubExpr();

    if (SubExpr->getType()->isRealFloatingType()) {
      APFloat Result(0.0);

      if (!EvaluateFloat(SubExpr, Result, Info))
        return APValue();

      return APValue(APFloat(Result.getSemantics(), APFloat::fcZero, false),
                     Result);
    } else {
      assert(SubExpr->getType()->isIntegerType() &&
             "Unexpected imaginary literal.");

      llvm::APSInt Result;
      if (!EvaluateInteger(SubExpr, Result, Info))
        return APValue();

      llvm::APSInt Zero(Result.getBitWidth(), !Result.isSigned());
      Zero = 0;
      return APValue(Zero, Result);
    }
  }

  APValue VisitCastExpr(CastExpr *E) {
    Expr* SubExpr = E->getSubExpr();
    QualType EltType = E->getType()->getAs<ComplexType>()->getElementType();
    QualType SubType = SubExpr->getType();

    if (SubType->isRealFloatingType()) {
      APFloat Result(0.0);

      if (!EvaluateFloat(SubExpr, Result, Info))
        return APValue();

      if (EltType->isRealFloatingType()) {
        Result = HandleFloatToFloatCast(EltType, SubType, Result, Info.Ctx);
        return APValue(Result,
                       APFloat(Result.getSemantics(), APFloat::fcZero, false));
      } else {
        llvm::APSInt IResult;
        IResult = HandleFloatToIntCast(EltType, SubType, Result, Info.Ctx);
        llvm::APSInt Zero(IResult.getBitWidth(), !IResult.isSigned());
        Zero = 0;
        return APValue(IResult, Zero);
      }
    } else if (SubType->isIntegerType()) {
      APSInt Result;

      if (!EvaluateInteger(SubExpr, Result, Info))
        return APValue();

      if (EltType->isRealFloatingType()) {
        APFloat FResult =
            HandleIntToFloatCast(EltType, SubType, Result, Info.Ctx);
        return APValue(FResult,
                       APFloat(FResult.getSemantics(), APFloat::fcZero, false));
      } else {
        Result = HandleIntToIntCast(EltType, SubType, Result, Info.Ctx);
        llvm::APSInt Zero(Result.getBitWidth(), !Result.isSigned());
        Zero = 0;
        return APValue(Result, Zero);
      }
    } else if (const ComplexType *CT = SubType->getAs<ComplexType>()) {
      APValue Src;

      if (!EvaluateComplex(SubExpr, Src, Info))
        return APValue();

      QualType SrcType = CT->getElementType();

      if (Src.isComplexFloat()) {
        if (EltType->isRealFloatingType()) {
          return APValue(HandleFloatToFloatCast(EltType, SrcType,
                                                Src.getComplexFloatReal(),
                                                Info.Ctx),
                         HandleFloatToFloatCast(EltType, SrcType,
                                                Src.getComplexFloatImag(),
                                                Info.Ctx));
        } else {
          return APValue(HandleFloatToIntCast(EltType, SrcType,
                                              Src.getComplexFloatReal(),
                                              Info.Ctx),
                         HandleFloatToIntCast(EltType, SrcType,
                                              Src.getComplexFloatImag(),
                                              Info.Ctx));
        }
      } else {
        assert(Src.isComplexInt() && "Invalid evaluate result.");
        if (EltType->isRealFloatingType()) {
          return APValue(HandleIntToFloatCast(EltType, SrcType,
                                              Src.getComplexIntReal(),
                                              Info.Ctx),
                         HandleIntToFloatCast(EltType, SrcType,
                                              Src.getComplexIntImag(),
                                              Info.Ctx));
        } else {
          return APValue(HandleIntToIntCast(EltType, SrcType,
                                            Src.getComplexIntReal(),
                                            Info.Ctx),
                         HandleIntToIntCast(EltType, SrcType,
                                            Src.getComplexIntImag(),
                                            Info.Ctx));
        }
      }
    }

    // FIXME: Handle more casts.
    return APValue();
  }

  APValue VisitBinaryOperator(const BinaryOperator *E);
  APValue VisitChooseExpr(const ChooseExpr *E)
    { return Visit(E->getChosenSubExpr(Info.Ctx)); }
  APValue VisitUnaryExtension(const UnaryOperator *E)
    { return Visit(E->getSubExpr()); }
  // FIXME Missing: unary +/-/~, binary div, ImplicitValueInitExpr,
  //                conditional ?:, comma
};
} // end anonymous namespace

static bool EvaluateComplex(const Expr *E, APValue &Result, EvalInfo &Info) {
  Result = ComplexExprEvaluator(Info).Visit(const_cast<Expr*>(E));
  assert((!Result.isComplexFloat() ||
          (&Result.getComplexFloatReal().getSemantics() ==
           &Result.getComplexFloatImag().getSemantics())) &&
         "Invalid complex evaluation.");
  return Result.isComplexFloat() || Result.isComplexInt();
}

APValue ComplexExprEvaluator::VisitBinaryOperator(const BinaryOperator *E) {
  APValue Result, RHS;

  if (!EvaluateComplex(E->getLHS(), Result, Info))
    return APValue();

  if (!EvaluateComplex(E->getRHS(), RHS, Info))
    return APValue();

  assert(Result.isComplexFloat() == RHS.isComplexFloat() &&
         "Invalid operands to binary operator.");
  switch (E->getOpcode()) {
  default: return APValue();
  case BinaryOperator::Add:
    if (Result.isComplexFloat()) {
      Result.getComplexFloatReal().add(RHS.getComplexFloatReal(),
                                       APFloat::rmNearestTiesToEven);
      Result.getComplexFloatImag().add(RHS.getComplexFloatImag(),
                                       APFloat::rmNearestTiesToEven);
    } else {
      Result.getComplexIntReal() += RHS.getComplexIntReal();
      Result.getComplexIntImag() += RHS.getComplexIntImag();
    }
    break;
  case BinaryOperator::Sub:
    if (Result.isComplexFloat()) {
      Result.getComplexFloatReal().subtract(RHS.getComplexFloatReal(),
                                            APFloat::rmNearestTiesToEven);
      Result.getComplexFloatImag().subtract(RHS.getComplexFloatImag(),
                                            APFloat::rmNearestTiesToEven);
    } else {
      Result.getComplexIntReal() -= RHS.getComplexIntReal();
      Result.getComplexIntImag() -= RHS.getComplexIntImag();
    }
    break;
  case BinaryOperator::Mul:
    if (Result.isComplexFloat()) {
      APValue LHS = Result;
      APFloat &LHS_r = LHS.getComplexFloatReal();
      APFloat &LHS_i = LHS.getComplexFloatImag();
      APFloat &RHS_r = RHS.getComplexFloatReal();
      APFloat &RHS_i = RHS.getComplexFloatImag();

      APFloat Tmp = LHS_r;
      Tmp.multiply(RHS_r, APFloat::rmNearestTiesToEven);
      Result.getComplexFloatReal() = Tmp;
      Tmp = LHS_i;
      Tmp.multiply(RHS_i, APFloat::rmNearestTiesToEven);
      Result.getComplexFloatReal().subtract(Tmp, APFloat::rmNearestTiesToEven);

      Tmp = LHS_r;
      Tmp.multiply(RHS_i, APFloat::rmNearestTiesToEven);
      Result.getComplexFloatImag() = Tmp;
      Tmp = LHS_i;
      Tmp.multiply(RHS_r, APFloat::rmNearestTiesToEven);
      Result.getComplexFloatImag().add(Tmp, APFloat::rmNearestTiesToEven);
    } else {
      APValue LHS = Result;
      Result.getComplexIntReal() =
        (LHS.getComplexIntReal() * RHS.getComplexIntReal() -
         LHS.getComplexIntImag() * RHS.getComplexIntImag());
      Result.getComplexIntImag() =
        (LHS.getComplexIntReal() * RHS.getComplexIntImag() +
         LHS.getComplexIntImag() * RHS.getComplexIntReal());
    }
    break;
  }

  return Result;
}

//===----------------------------------------------------------------------===//
// Top level Expr::Evaluate method.
//===----------------------------------------------------------------------===//

/// Evaluate - Return true if this is a constant which we can fold using
/// any crazy technique (that has nothing to do with language standards) that
/// we want to.  If this function returns true, it returns the folded constant
/// in Result.
bool Expr::Evaluate(EvalResult &Result, ASTContext &Ctx) const {
  EvalInfo Info(Ctx, Result);

  if (getType()->isVectorType()) {
    if (!EvaluateVector(this, Result.Val, Info))
      return false;
  } else if (getType()->isIntegerType()) {
    if (!IntExprEvaluator(Info, Result.Val).Visit(const_cast<Expr*>(this)))
      return false;
  } else if (getType()->hasPointerRepresentation()) {
    if (!EvaluatePointer(this, Result.Val, Info))
      return false;
  } else if (getType()->isRealFloatingType()) {
    llvm::APFloat f(0.0);
    if (!EvaluateFloat(this, f, Info))
      return false;

    Result.Val = APValue(f);
  } else if (getType()->isAnyComplexType()) {
    if (!EvaluateComplex(this, Result.Val, Info))
      return false;
  } else
    return false;

  return true;
}

bool Expr::EvaluateAsAny(EvalResult &Result, ASTContext &Ctx) const {
  EvalInfo Info(Ctx, Result, true);

  if (getType()->isVectorType()) {
    if (!EvaluateVector(this, Result.Val, Info))
      return false;
  } else if (getType()->isIntegerType()) {
    if (!IntExprEvaluator(Info, Result.Val).Visit(const_cast<Expr*>(this)))
      return false;
  } else if (getType()->hasPointerRepresentation()) {
    if (!EvaluatePointer(this, Result.Val, Info))
      return false;
  } else if (getType()->isRealFloatingType()) {
    llvm::APFloat f(0.0);
    if (!EvaluateFloat(this, f, Info))
      return false;

    Result.Val = APValue(f);
  } else if (getType()->isAnyComplexType()) {
    if (!EvaluateComplex(this, Result.Val, Info))
      return false;
  } else
    return false;

  return true;
}

bool Expr::EvaluateAsBooleanCondition(bool &Result, ASTContext &Ctx) const {
  EvalResult Scratch;
  EvalInfo Info(Ctx, Scratch);

  return HandleConversionToBool(this, Result, Info);
}

bool Expr::EvaluateAsLValue(EvalResult &Result, ASTContext &Ctx) const {
  EvalInfo Info(Ctx, Result);

  return EvaluateLValue(this, Result.Val, Info) && !Result.HasSideEffects;
}

bool Expr::EvaluateAsAnyLValue(EvalResult &Result, ASTContext &Ctx) const {
  EvalInfo Info(Ctx, Result, true);

  return EvaluateLValue(this, Result.Val, Info) && !Result.HasSideEffects;
}

/// isEvaluatable - Call Evaluate to see if this expression can be constant
/// folded, but discard the result.
bool Expr::isEvaluatable(ASTContext &Ctx) const {
  EvalResult Result;
  return Evaluate(Result, Ctx) && !Result.HasSideEffects;
}

bool Expr::HasSideEffects(ASTContext &Ctx) const {
  Expr::EvalResult Result;
  EvalInfo Info(Ctx, Result);
  return HasSideEffect(Info).Visit(const_cast<Expr*>(this));
}

APSInt Expr::EvaluateAsInt(ASTContext &Ctx) const {
  EvalResult EvalResult;
  bool Result = Evaluate(EvalResult, Ctx);
  Result = Result;
  assert(Result && "Could not evaluate expression");
  assert(EvalResult.Val.isInt() && "Expression did not evaluate to integer");

  return EvalResult.Val.getInt();
}
