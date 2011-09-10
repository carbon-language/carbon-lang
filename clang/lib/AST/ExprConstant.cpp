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
#include "clang/AST/TypeLoc.h"
#include "clang/AST/ASTDiagnostic.h"
#include "clang/AST/Expr.h"
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
namespace {
  struct EvalInfo {
    const ASTContext &Ctx;

    /// EvalResult - Contains information about the evaluation.
    Expr::EvalResult &EvalResult;

    typedef llvm::DenseMap<const OpaqueValueExpr*, APValue> MapTy;
    MapTy OpaqueValues;
    const APValue *getOpaqueValue(const OpaqueValueExpr *e) const {
      MapTy::const_iterator i = OpaqueValues.find(e);
      if (i == OpaqueValues.end()) return 0;
      return &i->second;
    }

    EvalInfo(const ASTContext &ctx, Expr::EvalResult &evalresult)
      : Ctx(ctx), EvalResult(evalresult) {}
  };

  struct ComplexValue {
  private:
    bool IsInt;

  public:
    APSInt IntReal, IntImag;
    APFloat FloatReal, FloatImag;

    ComplexValue() : FloatReal(APFloat::Bogus), FloatImag(APFloat::Bogus) {}

    void makeComplexFloat() { IsInt = false; }
    bool isComplexFloat() const { return !IsInt; }
    APFloat &getComplexFloatReal() { return FloatReal; }
    APFloat &getComplexFloatImag() { return FloatImag; }

    void makeComplexInt() { IsInt = true; }
    bool isComplexInt() const { return IsInt; }
    APSInt &getComplexIntReal() { return IntReal; }
    APSInt &getComplexIntImag() { return IntImag; }

    void moveInto(APValue &v) const {
      if (isComplexFloat())
        v = APValue(FloatReal, FloatImag);
      else
        v = APValue(IntReal, IntImag);
    }
    void setFrom(const APValue &v) {
      assert(v.isComplexFloat() || v.isComplexInt());
      if (v.isComplexFloat()) {
        makeComplexFloat();
        FloatReal = v.getComplexFloatReal();
        FloatImag = v.getComplexFloatImag();
      } else {
        makeComplexInt();
        IntReal = v.getComplexIntReal();
        IntImag = v.getComplexIntImag();
      }
    }
  };

  struct LValue {
    const Expr *Base;
    CharUnits Offset;

    const Expr *getLValueBase() { return Base; }
    CharUnits getLValueOffset() { return Offset; }

    void moveInto(APValue &v) const {
      v = APValue(Base, Offset);
    }
    void setFrom(const APValue &v) {
      assert(v.isLValue());
      Base = v.getLValueBase();
      Offset = v.getLValueOffset();
    }
  };
}

static bool Evaluate(EvalInfo &info, const Expr *E);
static bool EvaluateLValue(const Expr *E, LValue &Result, EvalInfo &Info);
static bool EvaluatePointer(const Expr *E, LValue &Result, EvalInfo &Info);
static bool EvaluateInteger(const Expr *E, APSInt  &Result, EvalInfo &Info);
static bool EvaluateIntegerOrLValue(const Expr *E, APValue  &Result,
                                    EvalInfo &Info);
static bool EvaluateFloat(const Expr *E, APFloat &Result, EvalInfo &Info);
static bool EvaluateComplex(const Expr *E, ComplexValue &Res, EvalInfo &Info);

//===----------------------------------------------------------------------===//
// Misc utilities
//===----------------------------------------------------------------------===//

static bool IsGlobalLValue(const Expr* E) {
  if (!E) return true;

  if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E)) {
    if (isa<FunctionDecl>(DRE->getDecl()))
      return true;
    if (const VarDecl *VD = dyn_cast<VarDecl>(DRE->getDecl()))
      return VD->hasGlobalStorage();
    return false;
  }

  if (const CompoundLiteralExpr *CLE = dyn_cast<CompoundLiteralExpr>(E))
    return CLE->isFileScope();

  return true;
}

static bool EvalPointerValueAsBool(LValue& Value, bool& Result) {
  const Expr* Base = Value.Base;

  // A null base expression indicates a null pointer.  These are always
  // evaluatable, and they are false unless the offset is zero.
  if (!Base) {
    Result = !Value.Offset.isZero();
    return true;
  }

  // Require the base expression to be a global l-value.
  if (!IsGlobalLValue(Base)) return false;

  // We have a non-null base expression.  These are generally known to
  // be true, but if it'a decl-ref to a weak symbol it can be null at
  // runtime.
  Result = true;

  const DeclRefExpr* DeclRef = dyn_cast<DeclRefExpr>(Base);
  if (!DeclRef)
    return true;

  // If it's a weak symbol, it isn't constant-evaluable.
  const ValueDecl* Decl = DeclRef->getDecl();
  if (Decl->hasAttr<WeakAttr>() ||
      Decl->hasAttr<WeakRefAttr>() ||
      Decl->isWeakImported())
    return false;

  return true;
}

static bool HandleConversionToBool(const Expr* E, bool& Result,
                                   EvalInfo &Info) {
  if (E->getType()->isIntegralOrEnumerationType()) {
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
    LValue PointerResult;
    if (!EvaluatePointer(E, PointerResult, Info))
      return false;
    return EvalPointerValueAsBool(PointerResult, Result);
  } else if (E->getType()->isAnyComplexType()) {
    ComplexValue ComplexResult;
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
                                   APFloat &Value, const ASTContext &Ctx) {
  unsigned DestWidth = Ctx.getIntWidth(DestType);
  // Determine whether we are converting to unsigned or signed.
  bool DestSigned = DestType->isSignedIntegerOrEnumerationType();

  // FIXME: Warning for overflow.
  APSInt Result(DestWidth, !DestSigned);
  bool ignored;
  (void)Value.convertToInteger(Result, llvm::APFloat::rmTowardZero, &ignored);
  return Result;
}

static APFloat HandleFloatToFloatCast(QualType DestType, QualType SrcType,
                                      APFloat &Value, const ASTContext &Ctx) {
  bool ignored;
  APFloat Result = Value;
  Result.convert(Ctx.getFloatTypeSemantics(DestType),
                 APFloat::rmNearestTiesToEven, &ignored);
  return Result;
}

static APSInt HandleIntToIntCast(QualType DestType, QualType SrcType,
                                 APSInt &Value, const ASTContext &Ctx) {
  unsigned DestWidth = Ctx.getIntWidth(DestType);
  APSInt Result = Value;
  // Figure out if this is a truncate, extend or noop cast.
  // If the input is signed, do a sign extend, noop, or truncate.
  Result = Result.extOrTrunc(DestWidth);
  Result.setIsUnsigned(DestType->isUnsignedIntegerOrEnumerationType());
  return Result;
}

static APFloat HandleIntToFloatCast(QualType DestType, QualType SrcType,
                                    APSInt &Value, const ASTContext &Ctx) {

  APFloat Result(Ctx.getFloatTypeSemantics(DestType), 1);
  Result.convertFromAPInt(Value, Value.isSigned(),
                          APFloat::rmNearestTiesToEven);
  return Result;
}

namespace {
class HasSideEffect
  : public ConstStmtVisitor<HasSideEffect, bool> {
  EvalInfo &Info;
public:

  HasSideEffect(EvalInfo &info) : Info(info) {}

  // Unhandled nodes conservatively default to having side effects.
  bool VisitStmt(const Stmt *S) {
    return true;
  }

  bool VisitParenExpr(const ParenExpr *E) { return Visit(E->getSubExpr()); }
  bool VisitGenericSelectionExpr(const GenericSelectionExpr *E) {
    return Visit(E->getResultExpr());
  }
  bool VisitDeclRefExpr(const DeclRefExpr *E) {
    if (Info.Ctx.getCanonicalType(E->getType()).isVolatileQualified())
      return true;
    return false;
  }
  bool VisitObjCIvarRefExpr(const ObjCIvarRefExpr *E) {
    if (Info.Ctx.getCanonicalType(E->getType()).isVolatileQualified())
      return true;
    return false;
  }
  bool VisitBlockDeclRefExpr (const BlockDeclRefExpr *E) {
    if (Info.Ctx.getCanonicalType(E->getType()).isVolatileQualified())
      return true;
    return false;
  }

  // We don't want to evaluate BlockExprs multiple times, as they generate
  // a ton of code.
  bool VisitBlockExpr(const BlockExpr *E) { return true; }
  bool VisitPredefinedExpr(const PredefinedExpr *E) { return false; }
  bool VisitCompoundLiteralExpr(const CompoundLiteralExpr *E)
    { return Visit(E->getInitializer()); }
  bool VisitMemberExpr(const MemberExpr *E) { return Visit(E->getBase()); }
  bool VisitIntegerLiteral(const IntegerLiteral *E) { return false; }
  bool VisitFloatingLiteral(const FloatingLiteral *E) { return false; }
  bool VisitStringLiteral(const StringLiteral *E) { return false; }
  bool VisitCharacterLiteral(const CharacterLiteral *E) { return false; }
  bool VisitUnaryExprOrTypeTraitExpr(const UnaryExprOrTypeTraitExpr *E)
    { return false; }
  bool VisitArraySubscriptExpr(const ArraySubscriptExpr *E)
    { return Visit(E->getLHS()) || Visit(E->getRHS()); }
  bool VisitChooseExpr(const ChooseExpr *E)
    { return Visit(E->getChosenSubExpr(Info.Ctx)); }
  bool VisitCastExpr(const CastExpr *E) { return Visit(E->getSubExpr()); }
  bool VisitBinAssign(const BinaryOperator *E) { return true; }
  bool VisitCompoundAssignOperator(const BinaryOperator *E) { return true; }
  bool VisitBinaryOperator(const BinaryOperator *E)
  { return Visit(E->getLHS()) || Visit(E->getRHS()); }
  bool VisitUnaryPreInc(const UnaryOperator *E) { return true; }
  bool VisitUnaryPostInc(const UnaryOperator *E) { return true; }
  bool VisitUnaryPreDec(const UnaryOperator *E) { return true; }
  bool VisitUnaryPostDec(const UnaryOperator *E) { return true; }
  bool VisitUnaryDeref(const UnaryOperator *E) {
    if (Info.Ctx.getCanonicalType(E->getType()).isVolatileQualified())
      return true;
    return Visit(E->getSubExpr());
  }
  bool VisitUnaryOperator(const UnaryOperator *E) { return Visit(E->getSubExpr()); }
    
  // Has side effects if any element does.
  bool VisitInitListExpr(const InitListExpr *E) {
    for (unsigned i = 0, e = E->getNumInits(); i != e; ++i)
      if (Visit(E->getInit(i))) return true;
    if (const Expr *filler = E->getArrayFiller())
      return Visit(filler);
    return false;
  }
    
  bool VisitSizeOfPackExpr(const SizeOfPackExpr *) { return false; }
};

class OpaqueValueEvaluation {
  EvalInfo &info;
  OpaqueValueExpr *opaqueValue;

public:
  OpaqueValueEvaluation(EvalInfo &info, OpaqueValueExpr *opaqueValue,
                        Expr *value)
    : info(info), opaqueValue(opaqueValue) {

    // If evaluation fails, fail immediately.
    if (!Evaluate(info, value)) {
      this->opaqueValue = 0;
      return;
    }
    info.OpaqueValues[opaqueValue] = info.EvalResult.Val;
  }

  bool hasError() const { return opaqueValue == 0; }

  ~OpaqueValueEvaluation() {
    if (opaqueValue) info.OpaqueValues.erase(opaqueValue);
  }
};
  
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Generic Evaluation
//===----------------------------------------------------------------------===//
namespace {

template <class Derived, typename RetTy=void>
class ExprEvaluatorBase
  : public ConstStmtVisitor<Derived, RetTy> {
private:
  RetTy DerivedSuccess(const APValue &V, const Expr *E) {
    return static_cast<Derived*>(this)->Success(V, E);
  }
  RetTy DerivedError(const Expr *E) {
    return static_cast<Derived*>(this)->Error(E);
  }

protected:
  EvalInfo &Info;
  typedef ConstStmtVisitor<Derived, RetTy> StmtVisitorTy;
  typedef ExprEvaluatorBase ExprEvaluatorBaseTy;

public:
  ExprEvaluatorBase(EvalInfo &Info) : Info(Info) {}

  RetTy VisitStmt(const Stmt *) {
    assert(0 && "Expression evaluator should not be called on stmts");
    return DerivedError(0);
  }
  RetTy VisitExpr(const Expr *E) {
    return DerivedError(E);
  }

  RetTy VisitParenExpr(const ParenExpr *E)
    { return StmtVisitorTy::Visit(E->getSubExpr()); }
  RetTy VisitUnaryExtension(const UnaryOperator *E)
    { return StmtVisitorTy::Visit(E->getSubExpr()); }
  RetTy VisitUnaryPlus(const UnaryOperator *E)
    { return StmtVisitorTy::Visit(E->getSubExpr()); }
  RetTy VisitChooseExpr(const ChooseExpr *E)
    { return StmtVisitorTy::Visit(E->getChosenSubExpr(Info.Ctx)); }
  RetTy VisitGenericSelectionExpr(const GenericSelectionExpr *E)
    { return StmtVisitorTy::Visit(E->getResultExpr()); }
  RetTy VisitSubstNonTypeTemplateParmExpr(const SubstNonTypeTemplateParmExpr *E)
    { return StmtVisitorTy::Visit(E->getReplacement()); }

  RetTy VisitBinaryConditionalOperator(const BinaryConditionalOperator *E) {
    OpaqueValueEvaluation opaque(Info, E->getOpaqueValue(), E->getCommon());
    if (opaque.hasError())
      return DerivedError(E);

    bool cond;
    if (!HandleConversionToBool(E->getCond(), cond, Info))
      return DerivedError(E);

    return StmtVisitorTy::Visit(cond ? E->getTrueExpr() : E->getFalseExpr());
  }

  RetTy VisitConditionalOperator(const ConditionalOperator *E) {
    bool BoolResult;
    if (!HandleConversionToBool(E->getCond(), BoolResult, Info))
      return DerivedError(E);

    Expr* EvalExpr = BoolResult ? E->getTrueExpr() : E->getFalseExpr();
    return StmtVisitorTy::Visit(EvalExpr);
  }

  RetTy VisitOpaqueValueExpr(const OpaqueValueExpr *E) {
    const APValue *value = Info.getOpaqueValue(E);
    if (!value)
      return (E->getSourceExpr() ? StmtVisitorTy::Visit(E->getSourceExpr())
                                 : DerivedError(E));
    return DerivedSuccess(*value, E);
  }
};

}

//===----------------------------------------------------------------------===//
// LValue Evaluation
//===----------------------------------------------------------------------===//
namespace {
class LValueExprEvaluator
  : public ExprEvaluatorBase<LValueExprEvaluator, bool> {
  LValue &Result;
  const Decl *PrevDecl;

  bool Success(const Expr *E) {
    Result.Base = E;
    Result.Offset = CharUnits::Zero();
    return true;
  }
public:

  LValueExprEvaluator(EvalInfo &info, LValue &Result) :
    ExprEvaluatorBaseTy(info), Result(Result), PrevDecl(0) {}

  bool Success(const APValue &V, const Expr *E) {
    Result.setFrom(V);
    return true;
  }
  bool Error(const Expr *E) {
    return false;
  }
  
  bool VisitDeclRefExpr(const DeclRefExpr *E);
  bool VisitPredefinedExpr(const PredefinedExpr *E) { return Success(E); }
  bool VisitCompoundLiteralExpr(const CompoundLiteralExpr *E);
  bool VisitMemberExpr(const MemberExpr *E);
  bool VisitStringLiteral(const StringLiteral *E) { return Success(E); }
  bool VisitObjCEncodeExpr(const ObjCEncodeExpr *E) { return Success(E); }
  bool VisitArraySubscriptExpr(const ArraySubscriptExpr *E);
  bool VisitUnaryDeref(const UnaryOperator *E);

  bool VisitCastExpr(const CastExpr *E) {
    switch (E->getCastKind()) {
    default:
      return false;

    case CK_NoOp:
      return Visit(E->getSubExpr());
    }
  }
  // FIXME: Missing: __real__, __imag__

};
} // end anonymous namespace

static bool EvaluateLValue(const Expr* E, LValue& Result, EvalInfo &Info) {
  return LValueExprEvaluator(Info, Result).Visit(E);
}

bool LValueExprEvaluator::VisitDeclRefExpr(const DeclRefExpr *E) {
  if (isa<FunctionDecl>(E->getDecl())) {
    return Success(E);
  } else if (const VarDecl* VD = dyn_cast<VarDecl>(E->getDecl())) {
    if (!VD->getType()->isReferenceType())
      return Success(E);
    // Reference parameters can refer to anything even if they have an
    // "initializer" in the form of a default argument.
    if (!isa<ParmVarDecl>(VD)) {
      // FIXME: Check whether VD might be overridden!

      // Check for recursive initializers of references.
      if (PrevDecl == VD)
        return Error(E);
      PrevDecl = VD;
      if (const Expr *Init = VD->getAnyInitializer())
        return Visit(Init);
    }
  }

  return ExprEvaluatorBaseTy::VisitDeclRefExpr(E);
}

bool
LValueExprEvaluator::VisitCompoundLiteralExpr(const CompoundLiteralExpr *E) {
  return Success(E);
}

bool LValueExprEvaluator::VisitMemberExpr(const MemberExpr *E) {
  QualType Ty;
  if (E->isArrow()) {
    if (!EvaluatePointer(E->getBase(), Result, Info))
      return false;
    Ty = E->getBase()->getType()->getAs<PointerType>()->getPointeeType();
  } else {
    if (!Visit(E->getBase()))
      return false;
    Ty = E->getBase()->getType();
  }

  const RecordDecl *RD = Ty->getAs<RecordType>()->getDecl();
  const ASTRecordLayout &RL = Info.Ctx.getASTRecordLayout(RD);

  const FieldDecl *FD = dyn_cast<FieldDecl>(E->getMemberDecl());
  if (!FD) // FIXME: deal with other kinds of member expressions
    return false;

  if (FD->getType()->isReferenceType())
    return false;

  unsigned i = FD->getFieldIndex();
  Result.Offset += Info.Ctx.toCharUnitsFromBits(RL.getFieldOffset(i));
  return true;
}

bool LValueExprEvaluator::VisitArraySubscriptExpr(const ArraySubscriptExpr *E) {
  if (!EvaluatePointer(E->getBase(), Result, Info))
    return false;

  APSInt Index;
  if (!EvaluateInteger(E->getIdx(), Index, Info))
    return false;

  CharUnits ElementSize = Info.Ctx.getTypeSizeInChars(E->getType());
  Result.Offset += Index.getSExtValue() * ElementSize;
  return true;
}

bool LValueExprEvaluator::VisitUnaryDeref(const UnaryOperator *E) {
  return EvaluatePointer(E->getSubExpr(), Result, Info);
}

//===----------------------------------------------------------------------===//
// Pointer Evaluation
//===----------------------------------------------------------------------===//

namespace {
class PointerExprEvaluator
  : public ExprEvaluatorBase<PointerExprEvaluator, bool> {
  LValue &Result;

  bool Success(const Expr *E) {
    Result.Base = E;
    Result.Offset = CharUnits::Zero();
    return true;
  }
public:

  PointerExprEvaluator(EvalInfo &info, LValue &Result)
    : ExprEvaluatorBaseTy(info), Result(Result) {}

  bool Success(const APValue &V, const Expr *E) {
    Result.setFrom(V);
    return true;
  }
  bool Error(const Stmt *S) {
    return false;
  }

  bool VisitBinaryOperator(const BinaryOperator *E);
  bool VisitCastExpr(const CastExpr* E);
  bool VisitUnaryAddrOf(const UnaryOperator *E);
  bool VisitObjCStringLiteral(const ObjCStringLiteral *E)
      { return Success(E); }
  bool VisitAddrLabelExpr(const AddrLabelExpr *E)
      { return Success(E); }
  bool VisitCallExpr(const CallExpr *E);
  bool VisitBlockExpr(const BlockExpr *E) {
    if (!E->getBlockDecl()->hasCaptures())
      return Success(E);
    return false;
  }
  bool VisitImplicitValueInitExpr(const ImplicitValueInitExpr *E)
      { return Success((Expr*)0); }
  bool VisitCXXNullPtrLiteralExpr(const CXXNullPtrLiteralExpr *E)
      { return Success((Expr*)0); }

  // FIXME: Missing: @protocol, @selector
};
} // end anonymous namespace

static bool EvaluatePointer(const Expr* E, LValue& Result, EvalInfo &Info) {
  assert(E->getType()->hasPointerRepresentation());
  return PointerExprEvaluator(Info, Result).Visit(E);
}

bool PointerExprEvaluator::VisitBinaryOperator(const BinaryOperator *E) {
  if (E->getOpcode() != BO_Add &&
      E->getOpcode() != BO_Sub)
    return false;

  const Expr *PExp = E->getLHS();
  const Expr *IExp = E->getRHS();
  if (IExp->getType()->isPointerType())
    std::swap(PExp, IExp);

  if (!EvaluatePointer(PExp, Result, Info))
    return false;

  llvm::APSInt Offset;
  if (!EvaluateInteger(IExp, Offset, Info))
    return false;
  int64_t AdditionalOffset
    = Offset.isSigned() ? Offset.getSExtValue()
                        : static_cast<int64_t>(Offset.getZExtValue());

  // Compute the new offset in the appropriate width.

  QualType PointeeType =
    PExp->getType()->getAs<PointerType>()->getPointeeType();
  CharUnits SizeOfPointee;

  // Explicitly handle GNU void* and function pointer arithmetic extensions.
  if (PointeeType->isVoidType() || PointeeType->isFunctionType())
    SizeOfPointee = CharUnits::One();
  else
    SizeOfPointee = Info.Ctx.getTypeSizeInChars(PointeeType);

  if (E->getOpcode() == BO_Add)
    Result.Offset += AdditionalOffset * SizeOfPointee;
  else
    Result.Offset -= AdditionalOffset * SizeOfPointee;

  return true;
}

bool PointerExprEvaluator::VisitUnaryAddrOf(const UnaryOperator *E) {
  return EvaluateLValue(E->getSubExpr(), Result, Info);
}


bool PointerExprEvaluator::VisitCastExpr(const CastExpr* E) {
  const Expr* SubExpr = E->getSubExpr();

  switch (E->getCastKind()) {
  default:
    break;

  case CK_NoOp:
  case CK_BitCast:
  case CK_CPointerToObjCPointerCast:
  case CK_BlockPointerToObjCPointerCast:
  case CK_AnyPointerToBlockPointerCast:
    return Visit(SubExpr);

  case CK_DerivedToBase:
  case CK_UncheckedDerivedToBase: {
    LValue BaseLV;
    if (!EvaluatePointer(E->getSubExpr(), BaseLV, Info))
      return false;

    // Now figure out the necessary offset to add to the baseLV to get from
    // the derived class to the base class.
    CharUnits Offset = CharUnits::Zero();

    QualType Ty = E->getSubExpr()->getType();
    const CXXRecordDecl *DerivedDecl = 
      Ty->getAs<PointerType>()->getPointeeType()->getAsCXXRecordDecl();

    for (CastExpr::path_const_iterator PathI = E->path_begin(), 
         PathE = E->path_end(); PathI != PathE; ++PathI) {
      const CXXBaseSpecifier *Base = *PathI;

      // FIXME: If the base is virtual, we'd need to determine the type of the
      // most derived class and we don't support that right now.
      if (Base->isVirtual())
        return false;

      const CXXRecordDecl *BaseDecl = Base->getType()->getAsCXXRecordDecl();
      const ASTRecordLayout &Layout = Info.Ctx.getASTRecordLayout(DerivedDecl);

      Offset += Layout.getBaseClassOffset(BaseDecl);
      DerivedDecl = BaseDecl;
    }

    Result.Base = BaseLV.getLValueBase();
    Result.Offset = BaseLV.getLValueOffset() + Offset;
    return true;
  }

  case CK_NullToPointer: {
    Result.Base = 0;
    Result.Offset = CharUnits::Zero();
    return true;
  }

  case CK_IntegralToPointer: {
    APValue Value;
    if (!EvaluateIntegerOrLValue(SubExpr, Value, Info))
      break;

    if (Value.isInt()) {
      Value.getInt() = Value.getInt().extOrTrunc((unsigned)Info.Ctx.getTypeSize(E->getType()));
      Result.Base = 0;
      Result.Offset = CharUnits::fromQuantity(Value.getInt().getZExtValue());
      return true;
    } else {
      // Cast is of an lvalue, no need to change value.
      Result.Base = Value.getLValueBase();
      Result.Offset = Value.getLValueOffset();
      return true;
    }
  }
  case CK_ArrayToPointerDecay:
  case CK_FunctionToPointerDecay:
    return EvaluateLValue(SubExpr, Result, Info);
  }

  return false;
}

bool PointerExprEvaluator::VisitCallExpr(const CallExpr *E) {
  if (E->isBuiltinCall(Info.Ctx) ==
        Builtin::BI__builtin___CFStringMakeConstantString ||
      E->isBuiltinCall(Info.Ctx) ==
        Builtin::BI__builtin___NSStringMakeConstantString)
    return Success(E);

  return ExprEvaluatorBaseTy::VisitCallExpr(E);
}

//===----------------------------------------------------------------------===//
// Vector Evaluation
//===----------------------------------------------------------------------===//

namespace {
  class VectorExprEvaluator
  : public ExprEvaluatorBase<VectorExprEvaluator, APValue> {
    APValue GetZeroVector(QualType VecType);
  public:

    VectorExprEvaluator(EvalInfo &info) : ExprEvaluatorBaseTy(info) {}

    APValue Success(const APValue &V, const Expr *E) { return V; }
    APValue Error(const Expr *E) { return APValue(); }

    APValue VisitUnaryReal(const UnaryOperator *E)
      { return Visit(E->getSubExpr()); }
    APValue VisitImplicitValueInitExpr(const ImplicitValueInitExpr *E)
      { return GetZeroVector(E->getType()); }
    APValue VisitCastExpr(const CastExpr* E);
    APValue VisitCompoundLiteralExpr(const CompoundLiteralExpr *E);
    APValue VisitInitListExpr(const InitListExpr *E);
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
  Result = VectorExprEvaluator(Info).Visit(E);
  return !Result.isUninit();
}

APValue VectorExprEvaluator::VisitCastExpr(const CastExpr* E) {
  const VectorType *VTy = E->getType()->getAs<VectorType>();
  QualType EltTy = VTy->getElementType();
  unsigned NElts = VTy->getNumElements();
  unsigned EltWidth = Info.Ctx.getTypeSize(EltTy);

  const Expr* SE = E->getSubExpr();
  QualType SETy = SE->getType();

  switch (E->getCastKind()) {
  case CK_VectorSplat: {
    APValue Result = APValue();
    if (SETy->isIntegerType()) {
      APSInt IntResult;
      if (!EvaluateInteger(SE, IntResult, Info))
         return APValue();
      Result = APValue(IntResult);
    } else if (SETy->isRealFloatingType()) {
       APFloat F(0.0);
       if (!EvaluateFloat(SE, F, Info))
         return APValue();
       Result = APValue(F);
    } else {
      return APValue();
    }

    // Splat and create vector APValue.
    SmallVector<APValue, 4> Elts(NElts, Result);
    return APValue(&Elts[0], Elts.size());
  }
  case CK_BitCast: {
    if (SETy->isVectorType())
      return Visit(SE);

    if (!SETy->isIntegerType())
      return APValue();

    APSInt Init;
    if (!EvaluateInteger(SE, Init, Info))
      return APValue();

    assert((EltTy->isIntegerType() || EltTy->isRealFloatingType()) &&
           "Vectors must be composed of ints or floats");

    SmallVector<APValue, 4> Elts;
    for (unsigned i = 0; i != NElts; ++i) {
      APSInt Tmp = Init.extOrTrunc(EltWidth);

      if (EltTy->isIntegerType())
        Elts.push_back(APValue(Tmp));
      else
        Elts.push_back(APValue(APFloat(Tmp)));

      Init >>= EltWidth;
    }
    return APValue(&Elts[0], Elts.size());
  }
  case CK_LValueToRValue:
  case CK_NoOp:
    return Visit(SE);
  default:
    return APValue();
  }
}

APValue
VectorExprEvaluator::VisitCompoundLiteralExpr(const CompoundLiteralExpr *E) {
  return this->Visit(E->getInitializer());
}

APValue
VectorExprEvaluator::VisitInitListExpr(const InitListExpr *E) {
  const VectorType *VT = E->getType()->getAs<VectorType>();
  unsigned NumInits = E->getNumInits();
  unsigned NumElements = VT->getNumElements();

  QualType EltTy = VT->getElementType();
  SmallVector<APValue, 4> Elements;

  // If a vector is initialized with a single element, that value
  // becomes every element of the vector, not just the first.
  // This is the behavior described in the IBM AltiVec documentation.
  if (NumInits == 1) {
    
    // Handle the case where the vector is initialized by a another 
    // vector (OpenCL 6.1.6).
    if (E->getInit(0)->getType()->isVectorType())
      return this->Visit(const_cast<Expr*>(E->getInit(0)));
    
    APValue InitValue;
    if (EltTy->isIntegerType()) {
      llvm::APSInt sInt(32);
      if (!EvaluateInteger(E->getInit(0), sInt, Info))
        return APValue();
      InitValue = APValue(sInt);
    } else {
      llvm::APFloat f(0.0);
      if (!EvaluateFloat(E->getInit(0), f, Info))
        return APValue();
      InitValue = APValue(f);
    }
    for (unsigned i = 0; i < NumElements; i++) {
      Elements.push_back(InitValue);
    }
  } else {
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

  SmallVector<APValue, 4> Elements(VT->getNumElements(), ZeroElement);
  return APValue(&Elements[0], Elements.size());
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
  : public ExprEvaluatorBase<IntExprEvaluator, bool> {
  APValue &Result;
public:
  IntExprEvaluator(EvalInfo &info, APValue &result)
    : ExprEvaluatorBaseTy(info), Result(result) {}

  bool Success(const llvm::APSInt &SI, const Expr *E) {
    assert(E->getType()->isIntegralOrEnumerationType() &&
           "Invalid evaluation result.");
    assert(SI.isSigned() == E->getType()->isSignedIntegerOrEnumerationType() &&
           "Invalid evaluation result.");
    assert(SI.getBitWidth() == Info.Ctx.getIntWidth(E->getType()) &&
           "Invalid evaluation result.");
    Result = APValue(SI);
    return true;
  }

  bool Success(const llvm::APInt &I, const Expr *E) {
    assert(E->getType()->isIntegralOrEnumerationType() && 
           "Invalid evaluation result.");
    assert(I.getBitWidth() == Info.Ctx.getIntWidth(E->getType()) &&
           "Invalid evaluation result.");
    Result = APValue(APSInt(I));
    Result.getInt().setIsUnsigned(
                            E->getType()->isUnsignedIntegerOrEnumerationType());
    return true;
  }

  bool Success(uint64_t Value, const Expr *E) {
    assert(E->getType()->isIntegralOrEnumerationType() && 
           "Invalid evaluation result.");
    Result = APValue(Info.Ctx.MakeIntValue(Value, E->getType()));
    return true;
  }

  bool Success(CharUnits Size, const Expr *E) {
    return Success(Size.getQuantity(), E);
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

  bool Success(const APValue &V, const Expr *E) {
    return Success(V.getInt(), E);
  }
  bool Error(const Expr *E) {
    return Error(E->getLocStart(), diag::note_invalid_subexpr_in_ice, E);
  }

  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//

  bool VisitIntegerLiteral(const IntegerLiteral *E) {
    return Success(E->getValue(), E);
  }
  bool VisitCharacterLiteral(const CharacterLiteral *E) {
    return Success(E->getValue(), E);
  }

  bool CheckReferencedDecl(const Expr *E, const Decl *D);
  bool VisitDeclRefExpr(const DeclRefExpr *E) {
    if (CheckReferencedDecl(E, E->getDecl()))
      return true;

    return ExprEvaluatorBaseTy::VisitDeclRefExpr(E);
  }
  bool VisitMemberExpr(const MemberExpr *E) {
    if (CheckReferencedDecl(E, E->getMemberDecl())) {
      // Conservatively assume a MemberExpr will have side-effects
      Info.EvalResult.HasSideEffects = true;
      return true;
    }

    return ExprEvaluatorBaseTy::VisitMemberExpr(E);
  }

  bool VisitCallExpr(const CallExpr *E);
  bool VisitBinaryOperator(const BinaryOperator *E);
  bool VisitOffsetOfExpr(const OffsetOfExpr *E);
  bool VisitUnaryOperator(const UnaryOperator *E);

  bool VisitCastExpr(const CastExpr* E);
  bool VisitUnaryExprOrTypeTraitExpr(const UnaryExprOrTypeTraitExpr *E);

  bool VisitCXXBoolLiteralExpr(const CXXBoolLiteralExpr *E) {
    return Success(E->getValue(), E);
  }

  bool VisitGNUNullExpr(const GNUNullExpr *E) {
    return Success(0, E);
  }

  bool VisitCXXScalarValueInitExpr(const CXXScalarValueInitExpr *E) {
    return Success(0, E);
  }

  bool VisitImplicitValueInitExpr(const ImplicitValueInitExpr *E) {
    return Success(0, E);
  }

  bool VisitUnaryTypeTraitExpr(const UnaryTypeTraitExpr *E) {
    return Success(E->getValue(), E);
  }

  bool VisitBinaryTypeTraitExpr(const BinaryTypeTraitExpr *E) {
    return Success(E->getValue(), E);
  }

  bool VisitArrayTypeTraitExpr(const ArrayTypeTraitExpr *E) {
    return Success(E->getValue(), E);
  }

  bool VisitExpressionTraitExpr(const ExpressionTraitExpr *E) {
    return Success(E->getValue(), E);
  }

  bool VisitUnaryReal(const UnaryOperator *E);
  bool VisitUnaryImag(const UnaryOperator *E);

  bool VisitCXXNoexceptExpr(const CXXNoexceptExpr *E);
  bool VisitSizeOfPackExpr(const SizeOfPackExpr *E);
    
private:
  CharUnits GetAlignOfExpr(const Expr *E);
  CharUnits GetAlignOfType(QualType T);
  static QualType GetObjectType(const Expr *E);
  bool TryEvaluateBuiltinObjectSize(const CallExpr *E);
  // FIXME: Missing: array subscript of vector, member of vector
};
} // end anonymous namespace

static bool EvaluateIntegerOrLValue(const Expr* E, APValue &Result, EvalInfo &Info) {
  assert(E->getType()->isIntegralOrEnumerationType());
  return IntExprEvaluator(Info, Result).Visit(E);
}

static bool EvaluateInteger(const Expr* E, APSInt &Result, EvalInfo &Info) {
  assert(E->getType()->isIntegralOrEnumerationType());

  APValue Val;
  if (!EvaluateIntegerOrLValue(E, Val, Info) || !Val.isInt())
    return false;
  Result = Val.getInt();
  return true;
}

bool IntExprEvaluator::CheckReferencedDecl(const Expr* E, const Decl* D) {
  // Enums are integer constant exprs.
  if (const EnumConstantDecl *ECD = dyn_cast<EnumConstantDecl>(D)) {
    // Check for signedness/width mismatches between E type and ECD value.
    bool SameSign = (ECD->getInitVal().isSigned()
                     == E->getType()->isSignedIntegerOrEnumerationType());
    bool SameWidth = (ECD->getInitVal().getBitWidth()
                      == Info.Ctx.getIntWidth(E->getType()));
    if (SameSign && SameWidth)
      return Success(ECD->getInitVal(), E);
    else {
      // Get rid of mismatch (otherwise Success assertions will fail)
      // by computing a new value matching the type of E.
      llvm::APSInt Val = ECD->getInitVal();
      if (!SameSign)
        Val.setIsSigned(!ECD->getInitVal().isSigned());
      if (!SameWidth)
        Val = Val.extOrTrunc(Info.Ctx.getIntWidth(E->getType()));
      return Success(Val, E);
    }
  }

  // In C++, const, non-volatile integers initialized with ICEs are ICEs.
  // In C, they can also be folded, although they are not ICEs.
  if (Info.Ctx.getCanonicalType(E->getType()).getCVRQualifiers() 
                                                        == Qualifiers::Const) {

    if (isa<ParmVarDecl>(D))
      return false;

    if (const VarDecl *VD = dyn_cast<VarDecl>(D)) {
      if (const Expr *Init = VD->getAnyInitializer()) {
        if (APValue *V = VD->getEvaluatedValue()) {
          if (V->isInt())
            return Success(V->getInt(), E);
          return false;
        }

        if (VD->isEvaluatingValue())
          return false;

        VD->setEvaluatingValue();

        Expr::EvalResult EResult;
        if (Init->Evaluate(EResult, Info.Ctx) && !EResult.HasSideEffects &&
            EResult.Val.isInt()) {
          // Cache the evaluated value in the variable declaration.
          Result = EResult.Val;
          VD->setEvaluatedValue(Result);
          return true;
        }

        VD->setEvaluatedValue(APValue());
      }
    }
  }

  // Otherwise, random variable references are not constants.
  return false;
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
  else if (ArgTy->isStructureOrClassType())
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

/// Retrieves the "underlying object type" of the given expression,
/// as used by __builtin_object_size.
QualType IntExprEvaluator::GetObjectType(const Expr *E) {
  if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E)) {
    if (const VarDecl *VD = dyn_cast<VarDecl>(DRE->getDecl()))
      return VD->getType();
  } else if (isa<CompoundLiteralExpr>(E)) {
    return E->getType();
  }

  return QualType();
}

bool IntExprEvaluator::TryEvaluateBuiltinObjectSize(const CallExpr *E) {
  // TODO: Perhaps we should let LLVM lower this?
  LValue Base;
  if (!EvaluatePointer(E->getArg(0), Base, Info))
    return false;

  // If we can prove the base is null, lower to zero now.
  const Expr *LVBase = Base.getLValueBase();
  if (!LVBase) return Success(0, E);

  QualType T = GetObjectType(LVBase);
  if (T.isNull() ||
      T->isIncompleteType() ||
      T->isFunctionType() ||
      T->isVariablyModifiedType() ||
      T->isDependentType())
    return false;

  CharUnits Size = Info.Ctx.getTypeSizeInChars(T);
  CharUnits Offset = Base.getLValueOffset();

  if (!Offset.isNegative() && Offset <= Size)
    Size -= Offset;
  else
    Size = CharUnits::Zero();
  return Success(Size, E);
}

bool IntExprEvaluator::VisitCallExpr(const CallExpr *E) {
  switch (E->isBuiltinCall(Info.Ctx)) {
  default:
    return ExprEvaluatorBaseTy::VisitCallExpr(E);

  case Builtin::BI__builtin_object_size: {
    if (TryEvaluateBuiltinObjectSize(E))
      return true;

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
    Operand = Info.Ctx.getTargetInfo().getEHDataRegisterNumber(Operand);
    return Success(Operand, E);
  }

  case Builtin::BI__builtin_expect:
    return Visit(E->getArg(0));
      
  case Builtin::BIstrlen:
  case Builtin::BI__builtin_strlen:
    // As an extension, we support strlen() and __builtin_strlen() as constant
    // expressions when the argument is a string literal.
    if (const StringLiteral *S
               = dyn_cast<StringLiteral>(E->getArg(0)->IgnoreParenImpCasts())) {
      // The string literal may have embedded null characters. Find the first
      // one and truncate there.
      StringRef Str = S->getString();
      StringRef::size_type Pos = Str.find(0);
      if (Pos != StringRef::npos)
        Str = Str.substr(0, Pos);
      
      return Success(Str.size(), E);
    }
      
    return Error(E->getLocStart(), diag::note_invalid_subexpr_in_ice, E);
  }
}

bool IntExprEvaluator::VisitBinaryOperator(const BinaryOperator *E) {
  if (E->getOpcode() == BO_Comma) {
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
      if (lhsResult == (E->getOpcode() == BO_LOr))
        return Success(lhsResult, E);

      if (HandleConversionToBool(E->getRHS(), rhsResult, Info)) {
        if (E->getOpcode() == BO_LOr)
          return Success(lhsResult || rhsResult, E);
        else
          return Success(lhsResult && rhsResult, E);
      }
    } else {
      if (HandleConversionToBool(E->getRHS(), rhsResult, Info)) {
        // We can't evaluate the LHS; however, sometimes the result
        // is determined by the RHS: X && 0 -> 0, X || 1 -> 1.
        if (rhsResult == (E->getOpcode() == BO_LOr) ||
            !rhsResult == (E->getOpcode() == BO_LAnd)) {
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
    ComplexValue LHS, RHS;

    if (!EvaluateComplex(E->getLHS(), LHS, Info))
      return false;

    if (!EvaluateComplex(E->getRHS(), RHS, Info))
      return false;

    if (LHS.isComplexFloat()) {
      APFloat::cmpResult CR_r =
        LHS.getComplexFloatReal().compare(RHS.getComplexFloatReal());
      APFloat::cmpResult CR_i =
        LHS.getComplexFloatImag().compare(RHS.getComplexFloatImag());

      if (E->getOpcode() == BO_EQ)
        return Success((CR_r == APFloat::cmpEqual &&
                        CR_i == APFloat::cmpEqual), E);
      else {
        assert(E->getOpcode() == BO_NE &&
               "Invalid complex comparison.");
        return Success(((CR_r == APFloat::cmpGreaterThan ||
                         CR_r == APFloat::cmpLessThan ||
                         CR_r == APFloat::cmpUnordered) ||
                        (CR_i == APFloat::cmpGreaterThan ||
                         CR_i == APFloat::cmpLessThan ||
                         CR_i == APFloat::cmpUnordered)), E);
      }
    } else {
      if (E->getOpcode() == BO_EQ)
        return Success((LHS.getComplexIntReal() == RHS.getComplexIntReal() &&
                        LHS.getComplexIntImag() == RHS.getComplexIntImag()), E);
      else {
        assert(E->getOpcode() == BO_NE &&
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
    case BO_LT:
      return Success(CR == APFloat::cmpLessThan, E);
    case BO_GT:
      return Success(CR == APFloat::cmpGreaterThan, E);
    case BO_LE:
      return Success(CR == APFloat::cmpLessThan || CR == APFloat::cmpEqual, E);
    case BO_GE:
      return Success(CR == APFloat::cmpGreaterThan || CR == APFloat::cmpEqual,
                     E);
    case BO_EQ:
      return Success(CR == APFloat::cmpEqual, E);
    case BO_NE:
      return Success(CR == APFloat::cmpGreaterThan
                     || CR == APFloat::cmpLessThan
                     || CR == APFloat::cmpUnordered, E);
    }
  }

  if (LHSTy->isPointerType() && RHSTy->isPointerType()) {
    if (E->getOpcode() == BO_Sub || E->isEqualityOp()) {
      LValue LHSValue;
      if (!EvaluatePointer(E->getLHS(), LHSValue, Info))
        return false;

      LValue RHSValue;
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
        return Success(bres ^ (E->getOpcode() == BO_EQ), E);
      } else if (RHSValue.getLValueBase()) {
        if (!E->isEqualityOp())
          return false;
        if (LHSValue.getLValueBase() || !LHSValue.getLValueOffset().isZero())
          return false;
        bool bres;
        if (!EvalPointerValueAsBool(RHSValue, bres))
          return false;
        return Success(bres ^ (E->getOpcode() == BO_EQ), E);
      }

      if (E->getOpcode() == BO_Sub) {
        QualType Type = E->getLHS()->getType();
        QualType ElementType = Type->getAs<PointerType>()->getPointeeType();

        CharUnits ElementSize = CharUnits::One();
        if (!ElementType->isVoidType() && !ElementType->isFunctionType())
          ElementSize = Info.Ctx.getTypeSizeInChars(ElementType);

        CharUnits Diff = LHSValue.getLValueOffset() - 
                             RHSValue.getLValueOffset();
        return Success(Diff / ElementSize, E);
      }
      bool Result;
      if (E->getOpcode() == BO_EQ) {
        Result = LHSValue.getLValueOffset() == RHSValue.getLValueOffset();
      } else {
        Result = LHSValue.getLValueOffset() != RHSValue.getLValueOffset();
      }
      return Success(Result, E);
    }
  }
  if (!LHSTy->isIntegralOrEnumerationType() ||
      !RHSTy->isIntegralOrEnumerationType()) {
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
    if (E->getOpcode() == BO_Add)
      Offset += AdditionalOffset;
    else
      Offset -= AdditionalOffset;
    Result = APValue(Result.getLValueBase(), Offset);
    return true;
  }

  // Handle cases like 4 + (unsigned long)&a
  if (E->getOpcode() == BO_Add &&
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
  case BO_Mul: return Success(Result.getInt() * RHS, E);
  case BO_Add: return Success(Result.getInt() + RHS, E);
  case BO_Sub: return Success(Result.getInt() - RHS, E);
  case BO_And: return Success(Result.getInt() & RHS, E);
  case BO_Xor: return Success(Result.getInt() ^ RHS, E);
  case BO_Or:  return Success(Result.getInt() | RHS, E);
  case BO_Div:
    if (RHS == 0)
      return Error(E->getOperatorLoc(), diag::note_expr_divide_by_zero, E);
    return Success(Result.getInt() / RHS, E);
  case BO_Rem:
    if (RHS == 0)
      return Error(E->getOperatorLoc(), diag::note_expr_divide_by_zero, E);
    return Success(Result.getInt() % RHS, E);
  case BO_Shl: {
    // During constant-folding, a negative shift is an opposite shift.
    if (RHS.isSigned() && RHS.isNegative()) {
      RHS = -RHS;
      goto shift_right;
    }

  shift_left:
    unsigned SA
      = (unsigned) RHS.getLimitedValue(Result.getInt().getBitWidth()-1);
    return Success(Result.getInt() << SA, E);
  }
  case BO_Shr: {
    // During constant-folding, a negative shift is an opposite shift.
    if (RHS.isSigned() && RHS.isNegative()) {
      RHS = -RHS;
      goto shift_left;
    }

  shift_right:
    unsigned SA =
      (unsigned) RHS.getLimitedValue(Result.getInt().getBitWidth()-1);
    return Success(Result.getInt() >> SA, E);
  }

  case BO_LT: return Success(Result.getInt() < RHS, E);
  case BO_GT: return Success(Result.getInt() > RHS, E);
  case BO_LE: return Success(Result.getInt() <= RHS, E);
  case BO_GE: return Success(Result.getInt() >= RHS, E);
  case BO_EQ: return Success(Result.getInt() == RHS, E);
  case BO_NE: return Success(Result.getInt() != RHS, E);
  }
}

CharUnits IntExprEvaluator::GetAlignOfType(QualType T) {
  // C++ [expr.sizeof]p2: "When applied to a reference or a reference type,
  //   the result is the size of the referenced type."
  // C++ [expr.alignof]p3: "When alignof is applied to a reference type, the
  //   result shall be the alignment of the referenced type."
  if (const ReferenceType *Ref = T->getAs<ReferenceType>())
    T = Ref->getPointeeType();

  // __alignof is defined to return the preferred alignment.
  return Info.Ctx.toCharUnitsFromBits(
    Info.Ctx.getPreferredTypeAlign(T.getTypePtr()));
}

CharUnits IntExprEvaluator::GetAlignOfExpr(const Expr *E) {
  E = E->IgnoreParens();

  // alignof decl is always accepted, even if it doesn't make sense: we default
  // to 1 in those cases.
  if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E))
    return Info.Ctx.getDeclAlign(DRE->getDecl(), 
                                 /*RefAsPointee*/true);

  if (const MemberExpr *ME = dyn_cast<MemberExpr>(E))
    return Info.Ctx.getDeclAlign(ME->getMemberDecl(),
                                 /*RefAsPointee*/true);

  return GetAlignOfType(E->getType());
}


/// VisitUnaryExprOrTypeTraitExpr - Evaluate a sizeof, alignof or vec_step with
/// a result as the expression's type.
bool IntExprEvaluator::VisitUnaryExprOrTypeTraitExpr(
                                    const UnaryExprOrTypeTraitExpr *E) {
  switch(E->getKind()) {
  case UETT_AlignOf: {
    if (E->isArgumentType())
      return Success(GetAlignOfType(E->getArgumentType()), E);
    else
      return Success(GetAlignOfExpr(E->getArgumentExpr()), E);
  }

  case UETT_VecStep: {
    QualType Ty = E->getTypeOfArgument();

    if (Ty->isVectorType()) {
      unsigned n = Ty->getAs<VectorType>()->getNumElements();

      // The vec_step built-in functions that take a 3-component
      // vector return 4. (OpenCL 1.1 spec 6.11.12)
      if (n == 3)
        n = 4;

      return Success(n, E);
    } else
      return Success(1, E);
  }

  case UETT_SizeOf: {
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
    return Success(Info.Ctx.getTypeSizeInChars(SrcTy), E);
  }
  }

  llvm_unreachable("unknown expr/type trait");
  return false;
}

bool IntExprEvaluator::VisitOffsetOfExpr(const OffsetOfExpr *OOE) {
  CharUnits Result;
  unsigned n = OOE->getNumComponents();
  if (n == 0)
    return false;
  QualType CurrentType = OOE->getTypeSourceInfo()->getType();
  for (unsigned i = 0; i != n; ++i) {
    OffsetOfExpr::OffsetOfNode ON = OOE->getComponent(i);
    switch (ON.getKind()) {
    case OffsetOfExpr::OffsetOfNode::Array: {
      const Expr *Idx = OOE->getIndexExpr(ON.getArrayExprIndex());
      APSInt IdxResult;
      if (!EvaluateInteger(Idx, IdxResult, Info))
        return false;
      const ArrayType *AT = Info.Ctx.getAsArrayType(CurrentType);
      if (!AT)
        return false;
      CurrentType = AT->getElementType();
      CharUnits ElementSize = Info.Ctx.getTypeSizeInChars(CurrentType);
      Result += IdxResult.getSExtValue() * ElementSize;
        break;
    }
        
    case OffsetOfExpr::OffsetOfNode::Field: {
      FieldDecl *MemberDecl = ON.getField();
      const RecordType *RT = CurrentType->getAs<RecordType>();
      if (!RT) 
        return false;
      RecordDecl *RD = RT->getDecl();
      const ASTRecordLayout &RL = Info.Ctx.getASTRecordLayout(RD);
      unsigned i = MemberDecl->getFieldIndex();
      assert(i < RL.getFieldCount() && "offsetof field in wrong type");
      Result += Info.Ctx.toCharUnitsFromBits(RL.getFieldOffset(i));
      CurrentType = MemberDecl->getType().getNonReferenceType();
      break;
    }
        
    case OffsetOfExpr::OffsetOfNode::Identifier:
      llvm_unreachable("dependent __builtin_offsetof");
      return false;
        
    case OffsetOfExpr::OffsetOfNode::Base: {
      CXXBaseSpecifier *BaseSpec = ON.getBase();
      if (BaseSpec->isVirtual())
        return false;

      // Find the layout of the class whose base we are looking into.
      const RecordType *RT = CurrentType->getAs<RecordType>();
      if (!RT) 
        return false;
      RecordDecl *RD = RT->getDecl();
      const ASTRecordLayout &RL = Info.Ctx.getASTRecordLayout(RD);

      // Find the base class itself.
      CurrentType = BaseSpec->getType();
      const RecordType *BaseRT = CurrentType->getAs<RecordType>();
      if (!BaseRT)
        return false;
      
      // Add the offset to the base.
      Result += RL.getBaseClassOffset(cast<CXXRecordDecl>(BaseRT->getDecl()));
      break;
    }
    }
  }
  return Success(Result, OOE);
}

bool IntExprEvaluator::VisitUnaryOperator(const UnaryOperator *E) {
  if (E->getOpcode() == UO_LNot) {
    // LNot's operand isn't necessarily an integer, so we handle it specially.
    bool bres;
    if (!HandleConversionToBool(E->getSubExpr(), bres, Info))
      return false;
    return Success(!bres, E);
  }

  // Only handle integral operations...
  if (!E->getSubExpr()->getType()->isIntegralOrEnumerationType())
    return false;

  // Get the operand value into 'Result'.
  if (!Visit(E->getSubExpr()))
    return false;

  switch (E->getOpcode()) {
  default:
    // Address, indirect, pre/post inc/dec, etc are not valid constant exprs.
    // See C99 6.6p3.
    return Error(E->getOperatorLoc(), diag::note_invalid_subexpr_in_ice, E);
  case UO_Extension:
    // FIXME: Should extension allow i-c-e extension expressions in its scope?
    // If so, we could clear the diagnostic ID.
    return true;
  case UO_Plus:
    // The result is always just the subexpr.
    return true;
  case UO_Minus:
    if (!Result.isInt()) return false;
    return Success(-Result.getInt(), E);
  case UO_Not:
    if (!Result.isInt()) return false;
    return Success(~Result.getInt(), E);
  }
}

/// HandleCast - This is used to evaluate implicit or explicit casts where the
/// result type is integer.
bool IntExprEvaluator::VisitCastExpr(const CastExpr *E) {
  const Expr *SubExpr = E->getSubExpr();
  QualType DestType = E->getType();
  QualType SrcType = SubExpr->getType();

  switch (E->getCastKind()) {
  case CK_BaseToDerived:
  case CK_DerivedToBase:
  case CK_UncheckedDerivedToBase:
  case CK_Dynamic:
  case CK_ToUnion:
  case CK_ArrayToPointerDecay:
  case CK_FunctionToPointerDecay:
  case CK_NullToPointer:
  case CK_NullToMemberPointer:
  case CK_BaseToDerivedMemberPointer:
  case CK_DerivedToBaseMemberPointer:
  case CK_ConstructorConversion:
  case CK_IntegralToPointer:
  case CK_ToVoid:
  case CK_VectorSplat:
  case CK_IntegralToFloating:
  case CK_FloatingCast:
  case CK_CPointerToObjCPointerCast:
  case CK_BlockPointerToObjCPointerCast:
  case CK_AnyPointerToBlockPointerCast:
  case CK_ObjCObjectLValueCast:
  case CK_FloatingRealToComplex:
  case CK_FloatingComplexToReal:
  case CK_FloatingComplexCast:
  case CK_FloatingComplexToIntegralComplex:
  case CK_IntegralRealToComplex:
  case CK_IntegralComplexCast:
  case CK_IntegralComplexToFloatingComplex:
    llvm_unreachable("invalid cast kind for integral value");

  case CK_BitCast:
  case CK_Dependent:
  case CK_GetObjCProperty:
  case CK_LValueBitCast:
  case CK_UserDefinedConversion:
  case CK_ARCProduceObject:
  case CK_ARCConsumeObject:
  case CK_ARCReclaimReturnedObject:
  case CK_ARCExtendBlockObject:
    return false;

  case CK_LValueToRValue:
  case CK_NoOp:
    return Visit(E->getSubExpr());

  case CK_MemberPointerToBoolean:
  case CK_PointerToBoolean:
  case CK_IntegralToBoolean:
  case CK_FloatingToBoolean:
  case CK_FloatingComplexToBoolean:
  case CK_IntegralComplexToBoolean: {
    bool BoolResult;
    if (!HandleConversionToBool(SubExpr, BoolResult, Info))
      return false;
    return Success(BoolResult, E);
  }

  case CK_IntegralCast: {
    if (!Visit(SubExpr))
      return false;

    if (!Result.isInt()) {
      // Only allow casts of lvalues if they are lossless.
      return Info.Ctx.getTypeSize(DestType) == Info.Ctx.getTypeSize(SrcType);
    }

    return Success(HandleIntToIntCast(DestType, SrcType,
                                      Result.getInt(), Info.Ctx), E);
  }

  case CK_PointerToIntegral: {
    LValue LV;
    if (!EvaluatePointer(SubExpr, LV, Info))
      return false;

    if (LV.getLValueBase()) {
      // Only allow based lvalue casts if they are lossless.
      if (Info.Ctx.getTypeSize(DestType) != Info.Ctx.getTypeSize(SrcType))
        return false;

      LV.moveInto(Result);
      return true;
    }

    APSInt AsInt = Info.Ctx.MakeIntValue(LV.getLValueOffset().getQuantity(), 
                                         SrcType);
    return Success(HandleIntToIntCast(DestType, SrcType, AsInt, Info.Ctx), E);
  }

  case CK_IntegralComplexToReal: {
    ComplexValue C;
    if (!EvaluateComplex(SubExpr, C, Info))
      return false;
    return Success(C.getComplexIntReal(), E);
  }

  case CK_FloatingToIntegral: {
    APFloat F(0.0);
    if (!EvaluateFloat(SubExpr, F, Info))
      return false;

    return Success(HandleFloatToIntCast(DestType, SrcType, F, Info.Ctx), E);
  }
  }

  llvm_unreachable("unknown cast resulting in integral value");
  return false;
}

bool IntExprEvaluator::VisitUnaryReal(const UnaryOperator *E) {
  if (E->getSubExpr()->getType()->isAnyComplexType()) {
    ComplexValue LV;
    if (!EvaluateComplex(E->getSubExpr(), LV, Info) || !LV.isComplexInt())
      return Error(E->getExprLoc(), diag::note_invalid_subexpr_in_ice, E);
    return Success(LV.getComplexIntReal(), E);
  }

  return Visit(E->getSubExpr());
}

bool IntExprEvaluator::VisitUnaryImag(const UnaryOperator *E) {
  if (E->getSubExpr()->getType()->isComplexIntegerType()) {
    ComplexValue LV;
    if (!EvaluateComplex(E->getSubExpr(), LV, Info) || !LV.isComplexInt())
      return Error(E->getExprLoc(), diag::note_invalid_subexpr_in_ice, E);
    return Success(LV.getComplexIntImag(), E);
  }

  if (!E->getSubExpr()->isEvaluatable(Info.Ctx))
    Info.EvalResult.HasSideEffects = true;
  return Success(0, E);
}

bool IntExprEvaluator::VisitSizeOfPackExpr(const SizeOfPackExpr *E) {
  return Success(E->getPackLength(), E);
}

bool IntExprEvaluator::VisitCXXNoexceptExpr(const CXXNoexceptExpr *E) {
  return Success(E->getValue(), E);
}

//===----------------------------------------------------------------------===//
// Float Evaluation
//===----------------------------------------------------------------------===//

namespace {
class FloatExprEvaluator
  : public ExprEvaluatorBase<FloatExprEvaluator, bool> {
  APFloat &Result;
public:
  FloatExprEvaluator(EvalInfo &info, APFloat &result)
    : ExprEvaluatorBaseTy(info), Result(result) {}

  bool Success(const APValue &V, const Expr *e) {
    Result = V.getFloat();
    return true;
  }
  bool Error(const Stmt *S) {
    return false;
  }

  bool VisitCallExpr(const CallExpr *E);

  bool VisitUnaryOperator(const UnaryOperator *E);
  bool VisitBinaryOperator(const BinaryOperator *E);
  bool VisitFloatingLiteral(const FloatingLiteral *E);
  bool VisitCastExpr(const CastExpr *E);
  bool VisitCXXScalarValueInitExpr(const CXXScalarValueInitExpr *E);

  bool VisitUnaryReal(const UnaryOperator *E);
  bool VisitUnaryImag(const UnaryOperator *E);

  bool VisitDeclRefExpr(const DeclRefExpr *E);

  // FIXME: Missing: array subscript of vector, member of vector,
  //                 ImplicitValueInitExpr
};
} // end anonymous namespace

static bool EvaluateFloat(const Expr* E, APFloat& Result, EvalInfo &Info) {
  assert(E->getType()->isRealFloatingType());
  return FloatExprEvaluator(Info, Result).Visit(E);
}

static bool TryEvaluateBuiltinNaN(const ASTContext &Context,
                                  QualType ResultTy,
                                  const Expr *Arg,
                                  bool SNaN,
                                  llvm::APFloat &Result) {
  const StringLiteral *S = dyn_cast<StringLiteral>(Arg->IgnoreParenCasts());
  if (!S) return false;

  const llvm::fltSemantics &Sem = Context.getFloatTypeSemantics(ResultTy);

  llvm::APInt fill;

  // Treat empty strings as if they were zero.
  if (S->getString().empty())
    fill = llvm::APInt(32, 0);
  else if (S->getString().getAsInteger(0, fill))
    return false;

  if (SNaN)
    Result = llvm::APFloat::getSNaN(Sem, false, &fill);
  else
    Result = llvm::APFloat::getQNaN(Sem, false, &fill);
  return true;
}

bool FloatExprEvaluator::VisitCallExpr(const CallExpr *E) {
  switch (E->isBuiltinCall(Info.Ctx)) {
  default:
    return ExprEvaluatorBaseTy::VisitCallExpr(E);

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

  case Builtin::BI__builtin_nans:
  case Builtin::BI__builtin_nansf:
  case Builtin::BI__builtin_nansl:
    return TryEvaluateBuiltinNaN(Info.Ctx, E->getType(), E->getArg(0),
                                 true, Result);

  case Builtin::BI__builtin_nan:
  case Builtin::BI__builtin_nanf:
  case Builtin::BI__builtin_nanl:
    // If this is __builtin_nan() turn this into a nan, otherwise we
    // can't constant fold it.
    return TryEvaluateBuiltinNaN(Info.Ctx, E->getType(), E->getArg(0),
                                 false, Result);

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

bool FloatExprEvaluator::VisitDeclRefExpr(const DeclRefExpr *E) {
  if (ExprEvaluatorBaseTy::VisitDeclRefExpr(E))
    return true;

  const Decl *D = E->getDecl();
  if (!isa<VarDecl>(D) || isa<ParmVarDecl>(D)) return false;
  const VarDecl *VD = cast<VarDecl>(D);

  // Require the qualifiers to be const and not volatile.
  CanQualType T = Info.Ctx.getCanonicalType(E->getType());
  if (!T.isConstQualified() || T.isVolatileQualified())
    return false;

  const Expr *Init = VD->getAnyInitializer();
  if (!Init) return false;

  if (APValue *V = VD->getEvaluatedValue()) {
    if (V->isFloat()) {
      Result = V->getFloat();
      return true;
    }
    return false;
  }

  if (VD->isEvaluatingValue())
    return false;

  VD->setEvaluatingValue();

  Expr::EvalResult InitResult;
  if (Init->Evaluate(InitResult, Info.Ctx) && !InitResult.HasSideEffects &&
      InitResult.Val.isFloat()) {
    // Cache the evaluated value in the variable declaration.
    Result = InitResult.Val.getFloat();
    VD->setEvaluatedValue(InitResult.Val);
    return true;
  }

  VD->setEvaluatedValue(APValue());
  return false;
}

bool FloatExprEvaluator::VisitUnaryReal(const UnaryOperator *E) {
  if (E->getSubExpr()->getType()->isAnyComplexType()) {
    ComplexValue CV;
    if (!EvaluateComplex(E->getSubExpr(), CV, Info))
      return false;
    Result = CV.FloatReal;
    return true;
  }

  return Visit(E->getSubExpr());
}

bool FloatExprEvaluator::VisitUnaryImag(const UnaryOperator *E) {
  if (E->getSubExpr()->getType()->isAnyComplexType()) {
    ComplexValue CV;
    if (!EvaluateComplex(E->getSubExpr(), CV, Info))
      return false;
    Result = CV.FloatImag;
    return true;
  }

  if (!E->getSubExpr()->isEvaluatable(Info.Ctx))
    Info.EvalResult.HasSideEffects = true;
  const llvm::fltSemantics &Sem = Info.Ctx.getFloatTypeSemantics(E->getType());
  Result = llvm::APFloat::getZero(Sem);
  return true;
}

bool FloatExprEvaluator::VisitUnaryOperator(const UnaryOperator *E) {
  if (E->getOpcode() == UO_Deref)
    return false;

  if (!EvaluateFloat(E->getSubExpr(), Result, Info))
    return false;

  switch (E->getOpcode()) {
  default: return false;
  case UO_Plus:
    return true;
  case UO_Minus:
    Result.changeSign();
    return true;
  }
}

bool FloatExprEvaluator::VisitBinaryOperator(const BinaryOperator *E) {
  if (E->getOpcode() == BO_Comma) {
    if (!EvaluateFloat(E->getRHS(), Result, Info))
      return false;

    // If we can't evaluate the LHS, it might have side effects;
    // conservatively mark it.
    if (!E->getLHS()->isEvaluatable(Info.Ctx))
      Info.EvalResult.HasSideEffects = true;

    return true;
  }

  // We can't evaluate pointer-to-member operations.
  if (E->isPtrMemOp())
    return false;

  // FIXME: Diagnostics?  I really don't understand how the warnings
  // and errors are supposed to work.
  APFloat RHS(0.0);
  if (!EvaluateFloat(E->getLHS(), Result, Info))
    return false;
  if (!EvaluateFloat(E->getRHS(), RHS, Info))
    return false;

  switch (E->getOpcode()) {
  default: return false;
  case BO_Mul:
    Result.multiply(RHS, APFloat::rmNearestTiesToEven);
    return true;
  case BO_Add:
    Result.add(RHS, APFloat::rmNearestTiesToEven);
    return true;
  case BO_Sub:
    Result.subtract(RHS, APFloat::rmNearestTiesToEven);
    return true;
  case BO_Div:
    Result.divide(RHS, APFloat::rmNearestTiesToEven);
    return true;
  }
}

bool FloatExprEvaluator::VisitFloatingLiteral(const FloatingLiteral *E) {
  Result = E->getValue();
  return true;
}

bool FloatExprEvaluator::VisitCastExpr(const CastExpr *E) {
  const Expr* SubExpr = E->getSubExpr();

  switch (E->getCastKind()) {
  default:
    return false;

  case CK_LValueToRValue:
  case CK_NoOp:
    return Visit(SubExpr);

  case CK_IntegralToFloating: {
    APSInt IntResult;
    if (!EvaluateInteger(SubExpr, IntResult, Info))
      return false;
    Result = HandleIntToFloatCast(E->getType(), SubExpr->getType(),
                                  IntResult, Info.Ctx);
    return true;
  }

  case CK_FloatingCast: {
    if (!Visit(SubExpr))
      return false;
    Result = HandleFloatToFloatCast(E->getType(), SubExpr->getType(),
                                    Result, Info.Ctx);
    return true;
  }

  case CK_FloatingComplexToReal: {
    ComplexValue V;
    if (!EvaluateComplex(SubExpr, V, Info))
      return false;
    Result = V.getComplexFloatReal();
    return true;
  }
  }

  return false;
}

bool FloatExprEvaluator::VisitCXXScalarValueInitExpr(const CXXScalarValueInitExpr *E) {
  Result = APFloat::getZero(Info.Ctx.getFloatTypeSemantics(E->getType()));
  return true;
}

//===----------------------------------------------------------------------===//
// Complex Evaluation (for float and integer)
//===----------------------------------------------------------------------===//

namespace {
class ComplexExprEvaluator
  : public ExprEvaluatorBase<ComplexExprEvaluator, bool> {
  ComplexValue &Result;

public:
  ComplexExprEvaluator(EvalInfo &info, ComplexValue &Result)
    : ExprEvaluatorBaseTy(info), Result(Result) {}

  bool Success(const APValue &V, const Expr *e) {
    Result.setFrom(V);
    return true;
  }
  bool Error(const Expr *E) {
    return false;
  }

  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//

  bool VisitImaginaryLiteral(const ImaginaryLiteral *E);

  bool VisitCastExpr(const CastExpr *E);

  bool VisitBinaryOperator(const BinaryOperator *E);
  bool VisitUnaryOperator(const UnaryOperator *E);
  // FIXME Missing: ImplicitValueInitExpr
};
} // end anonymous namespace

static bool EvaluateComplex(const Expr *E, ComplexValue &Result,
                            EvalInfo &Info) {
  assert(E->getType()->isAnyComplexType());
  return ComplexExprEvaluator(Info, Result).Visit(E);
}

bool ComplexExprEvaluator::VisitImaginaryLiteral(const ImaginaryLiteral *E) {
  const Expr* SubExpr = E->getSubExpr();

  if (SubExpr->getType()->isRealFloatingType()) {
    Result.makeComplexFloat();
    APFloat &Imag = Result.FloatImag;
    if (!EvaluateFloat(SubExpr, Imag, Info))
      return false;

    Result.FloatReal = APFloat(Imag.getSemantics());
    return true;
  } else {
    assert(SubExpr->getType()->isIntegerType() &&
           "Unexpected imaginary literal.");

    Result.makeComplexInt();
    APSInt &Imag = Result.IntImag;
    if (!EvaluateInteger(SubExpr, Imag, Info))
      return false;

    Result.IntReal = APSInt(Imag.getBitWidth(), !Imag.isSigned());
    return true;
  }
}

bool ComplexExprEvaluator::VisitCastExpr(const CastExpr *E) {

  switch (E->getCastKind()) {
  case CK_BitCast:
  case CK_BaseToDerived:
  case CK_DerivedToBase:
  case CK_UncheckedDerivedToBase:
  case CK_Dynamic:
  case CK_ToUnion:
  case CK_ArrayToPointerDecay:
  case CK_FunctionToPointerDecay:
  case CK_NullToPointer:
  case CK_NullToMemberPointer:
  case CK_BaseToDerivedMemberPointer:
  case CK_DerivedToBaseMemberPointer:
  case CK_MemberPointerToBoolean:
  case CK_ConstructorConversion:
  case CK_IntegralToPointer:
  case CK_PointerToIntegral:
  case CK_PointerToBoolean:
  case CK_ToVoid:
  case CK_VectorSplat:
  case CK_IntegralCast:
  case CK_IntegralToBoolean:
  case CK_IntegralToFloating:
  case CK_FloatingToIntegral:
  case CK_FloatingToBoolean:
  case CK_FloatingCast:
  case CK_CPointerToObjCPointerCast:
  case CK_BlockPointerToObjCPointerCast:
  case CK_AnyPointerToBlockPointerCast:
  case CK_ObjCObjectLValueCast:
  case CK_FloatingComplexToReal:
  case CK_FloatingComplexToBoolean:
  case CK_IntegralComplexToReal:
  case CK_IntegralComplexToBoolean:
  case CK_ARCProduceObject:
  case CK_ARCConsumeObject:
  case CK_ARCReclaimReturnedObject:
  case CK_ARCExtendBlockObject:
    llvm_unreachable("invalid cast kind for complex value");

  case CK_LValueToRValue:
  case CK_NoOp:
    return Visit(E->getSubExpr());

  case CK_Dependent:
  case CK_GetObjCProperty:
  case CK_LValueBitCast:
  case CK_UserDefinedConversion:
    return false;

  case CK_FloatingRealToComplex: {
    APFloat &Real = Result.FloatReal;
    if (!EvaluateFloat(E->getSubExpr(), Real, Info))
      return false;

    Result.makeComplexFloat();
    Result.FloatImag = APFloat(Real.getSemantics());
    return true;
  }

  case CK_FloatingComplexCast: {
    if (!Visit(E->getSubExpr()))
      return false;

    QualType To = E->getType()->getAs<ComplexType>()->getElementType();
    QualType From
      = E->getSubExpr()->getType()->getAs<ComplexType>()->getElementType();

    Result.FloatReal
      = HandleFloatToFloatCast(To, From, Result.FloatReal, Info.Ctx);
    Result.FloatImag
      = HandleFloatToFloatCast(To, From, Result.FloatImag, Info.Ctx);
    return true;
  }

  case CK_FloatingComplexToIntegralComplex: {
    if (!Visit(E->getSubExpr()))
      return false;

    QualType To = E->getType()->getAs<ComplexType>()->getElementType();
    QualType From
      = E->getSubExpr()->getType()->getAs<ComplexType>()->getElementType();
    Result.makeComplexInt();
    Result.IntReal = HandleFloatToIntCast(To, From, Result.FloatReal, Info.Ctx);
    Result.IntImag = HandleFloatToIntCast(To, From, Result.FloatImag, Info.Ctx);
    return true;
  }

  case CK_IntegralRealToComplex: {
    APSInt &Real = Result.IntReal;
    if (!EvaluateInteger(E->getSubExpr(), Real, Info))
      return false;

    Result.makeComplexInt();
    Result.IntImag = APSInt(Real.getBitWidth(), !Real.isSigned());
    return true;
  }

  case CK_IntegralComplexCast: {
    if (!Visit(E->getSubExpr()))
      return false;

    QualType To = E->getType()->getAs<ComplexType>()->getElementType();
    QualType From
      = E->getSubExpr()->getType()->getAs<ComplexType>()->getElementType();

    Result.IntReal = HandleIntToIntCast(To, From, Result.IntReal, Info.Ctx);
    Result.IntImag = HandleIntToIntCast(To, From, Result.IntImag, Info.Ctx);
    return true;
  }

  case CK_IntegralComplexToFloatingComplex: {
    if (!Visit(E->getSubExpr()))
      return false;

    QualType To = E->getType()->getAs<ComplexType>()->getElementType();
    QualType From
      = E->getSubExpr()->getType()->getAs<ComplexType>()->getElementType();
    Result.makeComplexFloat();
    Result.FloatReal = HandleIntToFloatCast(To, From, Result.IntReal, Info.Ctx);
    Result.FloatImag = HandleIntToFloatCast(To, From, Result.IntImag, Info.Ctx);
    return true;
  }
  }

  llvm_unreachable("unknown cast resulting in complex value");
  return false;
}

bool ComplexExprEvaluator::VisitBinaryOperator(const BinaryOperator *E) {
  if (E->getOpcode() == BO_Comma) {
    if (!Visit(E->getRHS()))
      return false;

    // If we can't evaluate the LHS, it might have side effects;
    // conservatively mark it.
    if (!E->getLHS()->isEvaluatable(Info.Ctx))
      Info.EvalResult.HasSideEffects = true;

    return true;
  }
  if (!Visit(E->getLHS()))
    return false;

  ComplexValue RHS;
  if (!EvaluateComplex(E->getRHS(), RHS, Info))
    return false;

  assert(Result.isComplexFloat() == RHS.isComplexFloat() &&
         "Invalid operands to binary operator.");
  switch (E->getOpcode()) {
  default: return false;
  case BO_Add:
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
  case BO_Sub:
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
  case BO_Mul:
    if (Result.isComplexFloat()) {
      ComplexValue LHS = Result;
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
      ComplexValue LHS = Result;
      Result.getComplexIntReal() =
        (LHS.getComplexIntReal() * RHS.getComplexIntReal() -
         LHS.getComplexIntImag() * RHS.getComplexIntImag());
      Result.getComplexIntImag() =
        (LHS.getComplexIntReal() * RHS.getComplexIntImag() +
         LHS.getComplexIntImag() * RHS.getComplexIntReal());
    }
    break;
  case BO_Div:
    if (Result.isComplexFloat()) {
      ComplexValue LHS = Result;
      APFloat &LHS_r = LHS.getComplexFloatReal();
      APFloat &LHS_i = LHS.getComplexFloatImag();
      APFloat &RHS_r = RHS.getComplexFloatReal();
      APFloat &RHS_i = RHS.getComplexFloatImag();
      APFloat &Res_r = Result.getComplexFloatReal();
      APFloat &Res_i = Result.getComplexFloatImag();

      APFloat Den = RHS_r;
      Den.multiply(RHS_r, APFloat::rmNearestTiesToEven);
      APFloat Tmp = RHS_i;
      Tmp.multiply(RHS_i, APFloat::rmNearestTiesToEven);
      Den.add(Tmp, APFloat::rmNearestTiesToEven);

      Res_r = LHS_r;
      Res_r.multiply(RHS_r, APFloat::rmNearestTiesToEven);
      Tmp = LHS_i;
      Tmp.multiply(RHS_i, APFloat::rmNearestTiesToEven);
      Res_r.add(Tmp, APFloat::rmNearestTiesToEven);
      Res_r.divide(Den, APFloat::rmNearestTiesToEven);

      Res_i = LHS_i;
      Res_i.multiply(RHS_r, APFloat::rmNearestTiesToEven);
      Tmp = LHS_r;
      Tmp.multiply(RHS_i, APFloat::rmNearestTiesToEven);
      Res_i.subtract(Tmp, APFloat::rmNearestTiesToEven);
      Res_i.divide(Den, APFloat::rmNearestTiesToEven);
    } else {
      if (RHS.getComplexIntReal() == 0 && RHS.getComplexIntImag() == 0) {
        // FIXME: what about diagnostics?
        return false;
      }
      ComplexValue LHS = Result;
      APSInt Den = RHS.getComplexIntReal() * RHS.getComplexIntReal() +
        RHS.getComplexIntImag() * RHS.getComplexIntImag();
      Result.getComplexIntReal() =
        (LHS.getComplexIntReal() * RHS.getComplexIntReal() +
         LHS.getComplexIntImag() * RHS.getComplexIntImag()) / Den;
      Result.getComplexIntImag() =
        (LHS.getComplexIntImag() * RHS.getComplexIntReal() -
         LHS.getComplexIntReal() * RHS.getComplexIntImag()) / Den;
    }
    break;
  }

  return true;
}

bool ComplexExprEvaluator::VisitUnaryOperator(const UnaryOperator *E) {
  // Get the operand value into 'Result'.
  if (!Visit(E->getSubExpr()))
    return false;

  switch (E->getOpcode()) {
  default:
    // FIXME: what about diagnostics?
    return false;
  case UO_Extension:
    return true;
  case UO_Plus:
    // The result is always just the subexpr.
    return true;
  case UO_Minus:
    if (Result.isComplexFloat()) {
      Result.getComplexFloatReal().changeSign();
      Result.getComplexFloatImag().changeSign();
    }
    else {
      Result.getComplexIntReal() = -Result.getComplexIntReal();
      Result.getComplexIntImag() = -Result.getComplexIntImag();
    }
    return true;
  case UO_Not:
    if (Result.isComplexFloat())
      Result.getComplexFloatImag().changeSign();
    else
      Result.getComplexIntImag() = -Result.getComplexIntImag();
    return true;
  }
}

//===----------------------------------------------------------------------===//
// Top level Expr::Evaluate method.
//===----------------------------------------------------------------------===//

static bool Evaluate(EvalInfo &Info, const Expr *E) {
  if (E->getType()->isVectorType()) {
    if (!EvaluateVector(E, Info.EvalResult.Val, Info))
      return false;
  } else if (E->getType()->isIntegralOrEnumerationType()) {
    if (!IntExprEvaluator(Info, Info.EvalResult.Val).Visit(E))
      return false;
    if (Info.EvalResult.Val.isLValue() &&
        !IsGlobalLValue(Info.EvalResult.Val.getLValueBase()))
      return false;
  } else if (E->getType()->hasPointerRepresentation()) {
    LValue LV;
    if (!EvaluatePointer(E, LV, Info))
      return false;
    if (!IsGlobalLValue(LV.Base))
      return false;
    LV.moveInto(Info.EvalResult.Val);
  } else if (E->getType()->isRealFloatingType()) {
    llvm::APFloat F(0.0);
    if (!EvaluateFloat(E, F, Info))
      return false;

    Info.EvalResult.Val = APValue(F);
  } else if (E->getType()->isAnyComplexType()) {
    ComplexValue C;
    if (!EvaluateComplex(E, C, Info))
      return false;
    C.moveInto(Info.EvalResult.Val);
  } else
    return false;

  return true;
}

/// Evaluate - Return true if this is a constant which we can fold using
/// any crazy technique (that has nothing to do with language standards) that
/// we want to.  If this function returns true, it returns the folded constant
/// in Result.
bool Expr::Evaluate(EvalResult &Result, const ASTContext &Ctx) const {
  EvalInfo Info(Ctx, Result);
  return ::Evaluate(Info, this);
}

bool Expr::EvaluateAsBooleanCondition(bool &Result,
                                      const ASTContext &Ctx) const {
  EvalResult Scratch;
  EvalInfo Info(Ctx, Scratch);

  return HandleConversionToBool(this, Result, Info);
}

bool Expr::EvaluateAsLValue(EvalResult &Result, const ASTContext &Ctx) const {
  EvalInfo Info(Ctx, Result);

  LValue LV;
  if (EvaluateLValue(this, LV, Info) &&
      !Result.HasSideEffects &&
      IsGlobalLValue(LV.Base)) {
    LV.moveInto(Result.Val);
    return true;
  }
  return false;
}

bool Expr::EvaluateAsAnyLValue(EvalResult &Result,
                               const ASTContext &Ctx) const {
  EvalInfo Info(Ctx, Result);

  LValue LV;
  if (EvaluateLValue(this, LV, Info)) {
    LV.moveInto(Result.Val);
    return true;
  }
  return false;
}

/// isEvaluatable - Call Evaluate to see if this expression can be constant
/// folded, but discard the result.
bool Expr::isEvaluatable(const ASTContext &Ctx) const {
  EvalResult Result;
  return Evaluate(Result, Ctx) && !Result.HasSideEffects;
}

bool Expr::HasSideEffects(const ASTContext &Ctx) const {
  Expr::EvalResult Result;
  EvalInfo Info(Ctx, Result);
  return HasSideEffect(Info).Visit(this);
}

APSInt Expr::EvaluateAsInt(const ASTContext &Ctx) const {
  EvalResult EvalResult;
  bool Result = Evaluate(EvalResult, Ctx);
  (void)Result;
  assert(Result && "Could not evaluate expression");
  assert(EvalResult.Val.isInt() && "Expression did not evaluate to integer");

  return EvalResult.Val.getInt();
}

 bool Expr::EvalResult::isGlobalLValue() const {
   assert(Val.isLValue());
   return IsGlobalLValue(Val.getLValueBase());
 }


/// isIntegerConstantExpr - this recursive routine will test if an expression is
/// an integer constant expression.

/// FIXME: Pass up a reason why! Invalid operation in i-c-e, division by zero,
/// comma, etc
///
/// FIXME: Handle offsetof.  Two things to do:  Handle GCC's __builtin_offsetof
/// to support gcc 4.0+  and handle the idiom GCC recognizes with a null pointer
/// cast+dereference.

// CheckICE - This function does the fundamental ICE checking: the returned
// ICEDiag contains a Val of 0, 1, or 2, and a possibly null SourceLocation.
// Note that to reduce code duplication, this helper does no evaluation
// itself; the caller checks whether the expression is evaluatable, and
// in the rare cases where CheckICE actually cares about the evaluated
// value, it calls into Evalute.
//
// Meanings of Val:
// 0: This expression is an ICE if it can be evaluated by Evaluate.
// 1: This expression is not an ICE, but if it isn't evaluated, it's
//    a legal subexpression for an ICE. This return value is used to handle
//    the comma operator in C99 mode.
// 2: This expression is not an ICE, and is not a legal subexpression for one.

namespace {

struct ICEDiag {
  unsigned Val;
  SourceLocation Loc;

  public:
  ICEDiag(unsigned v, SourceLocation l) : Val(v), Loc(l) {}
  ICEDiag() : Val(0) {}
};

}

static ICEDiag NoDiag() { return ICEDiag(); }

static ICEDiag CheckEvalInICE(const Expr* E, ASTContext &Ctx) {
  Expr::EvalResult EVResult;
  if (!E->Evaluate(EVResult, Ctx) || EVResult.HasSideEffects ||
      !EVResult.Val.isInt()) {
    return ICEDiag(2, E->getLocStart());
  }
  return NoDiag();
}

static ICEDiag CheckICE(const Expr* E, ASTContext &Ctx) {
  assert(!E->isValueDependent() && "Should not see value dependent exprs!");
  if (!E->getType()->isIntegralOrEnumerationType()) {
    return ICEDiag(2, E->getLocStart());
  }

  switch (E->getStmtClass()) {
#define ABSTRACT_STMT(Node)
#define STMT(Node, Base) case Expr::Node##Class:
#define EXPR(Node, Base)
#include "clang/AST/StmtNodes.inc"
  case Expr::PredefinedExprClass:
  case Expr::FloatingLiteralClass:
  case Expr::ImaginaryLiteralClass:
  case Expr::StringLiteralClass:
  case Expr::ArraySubscriptExprClass:
  case Expr::MemberExprClass:
  case Expr::CompoundAssignOperatorClass:
  case Expr::CompoundLiteralExprClass:
  case Expr::ExtVectorElementExprClass:
  case Expr::InitListExprClass:
  case Expr::DesignatedInitExprClass:
  case Expr::ImplicitValueInitExprClass:
  case Expr::ParenListExprClass:
  case Expr::VAArgExprClass:
  case Expr::AddrLabelExprClass:
  case Expr::StmtExprClass:
  case Expr::CXXMemberCallExprClass:
  case Expr::CUDAKernelCallExprClass:
  case Expr::CXXDynamicCastExprClass:
  case Expr::CXXTypeidExprClass:
  case Expr::CXXUuidofExprClass:
  case Expr::CXXNullPtrLiteralExprClass:
  case Expr::CXXThisExprClass:
  case Expr::CXXThrowExprClass:
  case Expr::CXXNewExprClass:
  case Expr::CXXDeleteExprClass:
  case Expr::CXXPseudoDestructorExprClass:
  case Expr::UnresolvedLookupExprClass:
  case Expr::DependentScopeDeclRefExprClass:
  case Expr::CXXConstructExprClass:
  case Expr::CXXBindTemporaryExprClass:
  case Expr::ExprWithCleanupsClass:
  case Expr::CXXTemporaryObjectExprClass:
  case Expr::CXXUnresolvedConstructExprClass:
  case Expr::CXXDependentScopeMemberExprClass:
  case Expr::UnresolvedMemberExprClass:
  case Expr::ObjCStringLiteralClass:
  case Expr::ObjCEncodeExprClass:
  case Expr::ObjCMessageExprClass:
  case Expr::ObjCSelectorExprClass:
  case Expr::ObjCProtocolExprClass:
  case Expr::ObjCIvarRefExprClass:
  case Expr::ObjCPropertyRefExprClass:
  case Expr::ObjCIsaExprClass:
  case Expr::ShuffleVectorExprClass:
  case Expr::BlockExprClass:
  case Expr::BlockDeclRefExprClass:
  case Expr::NoStmtClass:
  case Expr::OpaqueValueExprClass:
  case Expr::PackExpansionExprClass:
  case Expr::SubstNonTypeTemplateParmPackExprClass:
  case Expr::AsTypeExprClass:
  case Expr::ObjCIndirectCopyRestoreExprClass:
  case Expr::MaterializeTemporaryExprClass:
    return ICEDiag(2, E->getLocStart());

  case Expr::SizeOfPackExprClass:
  case Expr::GNUNullExprClass:
    // GCC considers the GNU __null value to be an integral constant expression.
    return NoDiag();

  case Expr::SubstNonTypeTemplateParmExprClass:
    return
      CheckICE(cast<SubstNonTypeTemplateParmExpr>(E)->getReplacement(), Ctx);

  case Expr::ParenExprClass:
    return CheckICE(cast<ParenExpr>(E)->getSubExpr(), Ctx);
  case Expr::GenericSelectionExprClass:
    return CheckICE(cast<GenericSelectionExpr>(E)->getResultExpr(), Ctx);
  case Expr::IntegerLiteralClass:
  case Expr::CharacterLiteralClass:
  case Expr::CXXBoolLiteralExprClass:
  case Expr::CXXScalarValueInitExprClass:
  case Expr::UnaryTypeTraitExprClass:
  case Expr::BinaryTypeTraitExprClass:
  case Expr::ArrayTypeTraitExprClass:
  case Expr::ExpressionTraitExprClass:
  case Expr::CXXNoexceptExprClass:
    return NoDiag();
  case Expr::CallExprClass:
  case Expr::CXXOperatorCallExprClass: {
    const CallExpr *CE = cast<CallExpr>(E);
    if (CE->isBuiltinCall(Ctx))
      return CheckEvalInICE(E, Ctx);
    return ICEDiag(2, E->getLocStart());
  }
  case Expr::DeclRefExprClass:
    if (isa<EnumConstantDecl>(cast<DeclRefExpr>(E)->getDecl()))
      return NoDiag();
    if (Ctx.getLangOptions().CPlusPlus &&
        E->getType().getCVRQualifiers() == Qualifiers::Const) {
      const NamedDecl *D = cast<DeclRefExpr>(E)->getDecl();

      // Parameter variables are never constants.  Without this check,
      // getAnyInitializer() can find a default argument, which leads
      // to chaos.
      if (isa<ParmVarDecl>(D))
        return ICEDiag(2, cast<DeclRefExpr>(E)->getLocation());

      // C++ 7.1.5.1p2
      //   A variable of non-volatile const-qualified integral or enumeration
      //   type initialized by an ICE can be used in ICEs.
      if (const VarDecl *Dcl = dyn_cast<VarDecl>(D)) {
        Qualifiers Quals = Ctx.getCanonicalType(Dcl->getType()).getQualifiers();
        if (Quals.hasVolatile() || !Quals.hasConst())
          return ICEDiag(2, cast<DeclRefExpr>(E)->getLocation());
        
        // Look for a declaration of this variable that has an initializer.
        const VarDecl *ID = 0;
        const Expr *Init = Dcl->getAnyInitializer(ID);
        if (Init) {
          if (ID->isInitKnownICE()) {
            // We have already checked whether this subexpression is an
            // integral constant expression.
            if (ID->isInitICE())
              return NoDiag();
            else
              return ICEDiag(2, cast<DeclRefExpr>(E)->getLocation());
          }

          // It's an ICE whether or not the definition we found is
          // out-of-line.  See DR 721 and the discussion in Clang PR
          // 6206 for details.

          if (Dcl->isCheckingICE()) {
            return ICEDiag(2, cast<DeclRefExpr>(E)->getLocation());
          }

          Dcl->setCheckingICE();
          ICEDiag Result = CheckICE(Init, Ctx);
          // Cache the result of the ICE test.
          Dcl->setInitKnownICE(Result.Val == 0);
          return Result;
        }
      }
    }
    return ICEDiag(2, E->getLocStart());
  case Expr::UnaryOperatorClass: {
    const UnaryOperator *Exp = cast<UnaryOperator>(E);
    switch (Exp->getOpcode()) {
    case UO_PostInc:
    case UO_PostDec:
    case UO_PreInc:
    case UO_PreDec:
    case UO_AddrOf:
    case UO_Deref:
      return ICEDiag(2, E->getLocStart());
    case UO_Extension:
    case UO_LNot:
    case UO_Plus:
    case UO_Minus:
    case UO_Not:
    case UO_Real:
    case UO_Imag:
      return CheckICE(Exp->getSubExpr(), Ctx);
    }
    
    // OffsetOf falls through here.
  }
  case Expr::OffsetOfExprClass: {
      // Note that per C99, offsetof must be an ICE. And AFAIK, using
      // Evaluate matches the proposed gcc behavior for cases like
      // "offsetof(struct s{int x[4];}, x[!.0])".  This doesn't affect
      // compliance: we should warn earlier for offsetof expressions with
      // array subscripts that aren't ICEs, and if the array subscripts
      // are ICEs, the value of the offsetof must be an integer constant.
      return CheckEvalInICE(E, Ctx);
  }
  case Expr::UnaryExprOrTypeTraitExprClass: {
    const UnaryExprOrTypeTraitExpr *Exp = cast<UnaryExprOrTypeTraitExpr>(E);
    if ((Exp->getKind() ==  UETT_SizeOf) &&
        Exp->getTypeOfArgument()->isVariableArrayType())
      return ICEDiag(2, E->getLocStart());
    return NoDiag();
  }
  case Expr::BinaryOperatorClass: {
    const BinaryOperator *Exp = cast<BinaryOperator>(E);
    switch (Exp->getOpcode()) {
    case BO_PtrMemD:
    case BO_PtrMemI:
    case BO_Assign:
    case BO_MulAssign:
    case BO_DivAssign:
    case BO_RemAssign:
    case BO_AddAssign:
    case BO_SubAssign:
    case BO_ShlAssign:
    case BO_ShrAssign:
    case BO_AndAssign:
    case BO_XorAssign:
    case BO_OrAssign:
      return ICEDiag(2, E->getLocStart());

    case BO_Mul:
    case BO_Div:
    case BO_Rem:
    case BO_Add:
    case BO_Sub:
    case BO_Shl:
    case BO_Shr:
    case BO_LT:
    case BO_GT:
    case BO_LE:
    case BO_GE:
    case BO_EQ:
    case BO_NE:
    case BO_And:
    case BO_Xor:
    case BO_Or:
    case BO_Comma: {
      ICEDiag LHSResult = CheckICE(Exp->getLHS(), Ctx);
      ICEDiag RHSResult = CheckICE(Exp->getRHS(), Ctx);
      if (Exp->getOpcode() == BO_Div ||
          Exp->getOpcode() == BO_Rem) {
        // Evaluate gives an error for undefined Div/Rem, so make sure
        // we don't evaluate one.
        if (LHSResult.Val == 0 && RHSResult.Val == 0) {
          llvm::APSInt REval = Exp->getRHS()->EvaluateAsInt(Ctx);
          if (REval == 0)
            return ICEDiag(1, E->getLocStart());
          if (REval.isSigned() && REval.isAllOnesValue()) {
            llvm::APSInt LEval = Exp->getLHS()->EvaluateAsInt(Ctx);
            if (LEval.isMinSignedValue())
              return ICEDiag(1, E->getLocStart());
          }
        }
      }
      if (Exp->getOpcode() == BO_Comma) {
        if (Ctx.getLangOptions().C99) {
          // C99 6.6p3 introduces a strange edge case: comma can be in an ICE
          // if it isn't evaluated.
          if (LHSResult.Val == 0 && RHSResult.Val == 0)
            return ICEDiag(1, E->getLocStart());
        } else {
          // In both C89 and C++, commas in ICEs are illegal.
          return ICEDiag(2, E->getLocStart());
        }
      }
      if (LHSResult.Val >= RHSResult.Val)
        return LHSResult;
      return RHSResult;
    }
    case BO_LAnd:
    case BO_LOr: {
      ICEDiag LHSResult = CheckICE(Exp->getLHS(), Ctx);

      // C++0x [expr.const]p2:
      //   [...] subexpressions of logical AND (5.14), logical OR
      //   (5.15), and condi- tional (5.16) operations that are not
      //   evaluated are not considered.
      if (Ctx.getLangOptions().CPlusPlus0x && LHSResult.Val == 0) {
        if (Exp->getOpcode() == BO_LAnd && 
            Exp->getLHS()->EvaluateAsInt(Ctx) == 0)
          return LHSResult;

        if (Exp->getOpcode() == BO_LOr &&
            Exp->getLHS()->EvaluateAsInt(Ctx) != 0)
          return LHSResult;
      }

      ICEDiag RHSResult = CheckICE(Exp->getRHS(), Ctx);
      if (LHSResult.Val == 0 && RHSResult.Val == 1) {
        // Rare case where the RHS has a comma "side-effect"; we need
        // to actually check the condition to see whether the side
        // with the comma is evaluated.
        if ((Exp->getOpcode() == BO_LAnd) !=
            (Exp->getLHS()->EvaluateAsInt(Ctx) == 0))
          return RHSResult;
        return NoDiag();
      }

      if (LHSResult.Val >= RHSResult.Val)
        return LHSResult;
      return RHSResult;
    }
    }
  }
  case Expr::ImplicitCastExprClass:
  case Expr::CStyleCastExprClass:
  case Expr::CXXFunctionalCastExprClass:
  case Expr::CXXStaticCastExprClass:
  case Expr::CXXReinterpretCastExprClass:
  case Expr::CXXConstCastExprClass: 
  case Expr::ObjCBridgedCastExprClass: {
    const Expr *SubExpr = cast<CastExpr>(E)->getSubExpr();
    if (SubExpr->getType()->isIntegralOrEnumerationType())
      return CheckICE(SubExpr, Ctx);
    if (isa<FloatingLiteral>(SubExpr->IgnoreParens()))
      return NoDiag();
    return ICEDiag(2, E->getLocStart());
  }
  case Expr::BinaryConditionalOperatorClass: {
    const BinaryConditionalOperator *Exp = cast<BinaryConditionalOperator>(E);
    ICEDiag CommonResult = CheckICE(Exp->getCommon(), Ctx);
    if (CommonResult.Val == 2) return CommonResult;
    ICEDiag FalseResult = CheckICE(Exp->getFalseExpr(), Ctx);
    if (FalseResult.Val == 2) return FalseResult;
    if (CommonResult.Val == 1) return CommonResult;
    if (FalseResult.Val == 1 &&
        Exp->getCommon()->EvaluateAsInt(Ctx) == 0) return NoDiag();
    return FalseResult;
  }
  case Expr::ConditionalOperatorClass: {
    const ConditionalOperator *Exp = cast<ConditionalOperator>(E);
    // If the condition (ignoring parens) is a __builtin_constant_p call,
    // then only the true side is actually considered in an integer constant
    // expression, and it is fully evaluated.  This is an important GNU
    // extension.  See GCC PR38377 for discussion.
    if (const CallExpr *CallCE
        = dyn_cast<CallExpr>(Exp->getCond()->IgnoreParenCasts()))
      if (CallCE->isBuiltinCall(Ctx) == Builtin::BI__builtin_constant_p) {
        Expr::EvalResult EVResult;
        if (!E->Evaluate(EVResult, Ctx) || EVResult.HasSideEffects ||
            !EVResult.Val.isInt()) {
          return ICEDiag(2, E->getLocStart());
        }
        return NoDiag();
      }
    ICEDiag CondResult = CheckICE(Exp->getCond(), Ctx);
    if (CondResult.Val == 2)
      return CondResult;

    // C++0x [expr.const]p2:
    //   subexpressions of [...] conditional (5.16) operations that
    //   are not evaluated are not considered
    bool TrueBranch = Ctx.getLangOptions().CPlusPlus0x
      ? Exp->getCond()->EvaluateAsInt(Ctx) != 0
      : false;
    ICEDiag TrueResult = NoDiag();
    if (!Ctx.getLangOptions().CPlusPlus0x || TrueBranch)
      TrueResult = CheckICE(Exp->getTrueExpr(), Ctx);
    ICEDiag FalseResult = NoDiag();
    if (!Ctx.getLangOptions().CPlusPlus0x || !TrueBranch)
      FalseResult = CheckICE(Exp->getFalseExpr(), Ctx);

    if (TrueResult.Val == 2)
      return TrueResult;
    if (FalseResult.Val == 2)
      return FalseResult;
    if (CondResult.Val == 1)
      return CondResult;
    if (TrueResult.Val == 0 && FalseResult.Val == 0)
      return NoDiag();
    // Rare case where the diagnostics depend on which side is evaluated
    // Note that if we get here, CondResult is 0, and at least one of
    // TrueResult and FalseResult is non-zero.
    if (Exp->getCond()->EvaluateAsInt(Ctx) == 0) {
      return FalseResult;
    }
    return TrueResult;
  }
  case Expr::CXXDefaultArgExprClass:
    return CheckICE(cast<CXXDefaultArgExpr>(E)->getExpr(), Ctx);
  case Expr::ChooseExprClass: {
    return CheckICE(cast<ChooseExpr>(E)->getChosenSubExpr(Ctx), Ctx);
  }
  }

  // Silence a GCC warning
  return ICEDiag(2, E->getLocStart());
}

bool Expr::isIntegerConstantExpr(llvm::APSInt &Result, ASTContext &Ctx,
                                 SourceLocation *Loc, bool isEvaluated) const {
  ICEDiag d = CheckICE(this, Ctx);
  if (d.Val != 0) {
    if (Loc) *Loc = d.Loc;
    return false;
  }
  EvalResult EvalResult;
  if (!Evaluate(EvalResult, Ctx))
    llvm_unreachable("ICE cannot be evaluated!");
  assert(!EvalResult.HasSideEffects && "ICE with side effects!");
  assert(EvalResult.Val.isInt() && "ICE that isn't integer!");
  Result = EvalResult.Val.getInt();
  return true;
}
