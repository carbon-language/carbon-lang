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
#include "clang/AST/ASTDiagnostic.h"
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
  
  /// EvalResult - Contains information about the evaluation.
  Expr::EvalResult &EvalResult;

  /// ShortCircuit - will be greater than zero if the current subexpression has
  /// will not be evaluated because it's short-circuited (according to C rules).
  unsigned ShortCircuit;

  EvalInfo(ASTContext &ctx, Expr::EvalResult& evalresult) : Ctx(ctx), 
           EvalResult(evalresult), ShortCircuit(0) {}
};


static bool EvaluateLValue(const Expr *E, APValue &Result, EvalInfo &Info);
static bool EvaluatePointer(const Expr *E, APValue &Result, EvalInfo &Info);
static bool EvaluateInteger(const Expr *E, APSInt  &Result, EvalInfo &Info);
static bool EvaluateFloat(const Expr *E, APFloat &Result, EvalInfo &Info);
static bool EvaluateComplex(const Expr *E, APValue &Result, EvalInfo &Info);

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
  APValue VisitDeclRefExpr(DeclRefExpr *E);
  APValue VisitPredefinedExpr(PredefinedExpr *E) { return APValue(E, 0); }
  APValue VisitCompoundLiteralExpr(CompoundLiteralExpr *E);
  APValue VisitMemberExpr(MemberExpr *E);
  APValue VisitStringLiteral(StringLiteral *E) { return APValue(E, 0); }
  APValue VisitArraySubscriptExpr(ArraySubscriptExpr *E);
};
} // end anonymous namespace

static bool EvaluateLValue(const Expr* E, APValue& Result, EvalInfo &Info) {
  Result = LValueExprEvaluator(Info).Visit(const_cast<Expr*>(E));
  return Result.isLValue();
}

APValue LValueExprEvaluator::VisitDeclRefExpr(DeclRefExpr *E)
{ 
  if (!E->hasGlobalStorage())
    return APValue();
  
  return APValue(E, 0); 
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

  FieldDecl *FD = dyn_cast<FieldDecl>(E->getMemberDecl());
  if (!FD) // FIXME: deal with other kinds of member expressions
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
                   result.getLValueOffset() + RL.getFieldOffset(i) / 8);

  return result;
}

APValue LValueExprEvaluator::VisitArraySubscriptExpr(ArraySubscriptExpr *E)
{
  APValue Result;
  
  if (!EvaluatePointer(E->getBase(), Result, Info))
    return APValue();
  
  APSInt Index;
  if (!EvaluateInteger(E->getIdx(), Index, Info))
    return APValue();

  uint64_t ElementSize = Info.Ctx.getTypeSize(E->getType()) / 8;

  uint64_t Offset = Index.getSExtValue() * ElementSize;
  Result.setLValue(Result.getLValueBase(), 
                   Result.getLValueOffset() + Offset);
  return Result;
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
  APValue VisitAddrLabelExpr(AddrLabelExpr *E)
      { return APValue(E, 0); }
  APValue VisitCallExpr(CallExpr *E);
  APValue VisitBlockExpr(BlockExpr *E) { return APValue(E, 0); }
  APValue VisitConditionalOperator(ConditionalOperator *E);
};
} // end anonymous namespace

static bool EvaluatePointer(const Expr* E, APValue& Result, EvalInfo &Info) {
  if (!E->getType()->isPointerType()
      && !E->getType()->isBlockPointerType())
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
  uint64_t SizeOfPointee;
  
  // Explicitly handle GNU void* and function pointer arithmetic extensions.
  if (PointeeType->isVoidType() || PointeeType->isFunctionType())
    SizeOfPointee = 1;
  else
    SizeOfPointee = Info.Ctx.getTypeSize(PointeeType) / 8;

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

APValue PointerExprEvaluator::VisitCallExpr(CallExpr *E) {
  if (E->isBuiltinCall(Info.Ctx) == 
        Builtin::BI__builtin___CFStringMakeConstantString)
    return APValue(E, 0);
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
  class VISIBILITY_HIDDEN VectorExprEvaluator
  : public StmtVisitor<VectorExprEvaluator, APValue> {
    EvalInfo &Info;
  public:
    
    VectorExprEvaluator(EvalInfo &info) : Info(info) {}
    
    APValue VisitStmt(Stmt *S) {
      return APValue();
    }
    
    APValue VisitParenExpr(ParenExpr *E) { return Visit(E->getSubExpr()); }
    APValue VisitCastExpr(const CastExpr* E);
    APValue VisitCompoundLiteralExpr(const CompoundLiteralExpr *E);
    APValue VisitInitListExpr(const InitListExpr *E);
  };
} // end anonymous namespace

static bool EvaluateVector(const Expr* E, APValue& Result, EvalInfo &Info) {
  if (!E->getType()->isVectorType())
    return false;
  Result = VectorExprEvaluator(Info).Visit(const_cast<Expr*>(E));
  return !Result.isUninit();
}

APValue VectorExprEvaluator::VisitCastExpr(const CastExpr* E) {
  const Expr* SE = E->getSubExpr();

  // Check for vector->vector bitcast.
  if (SE->getType()->isVectorType())
    return this->Visit(const_cast<Expr*>(SE));

  return APValue();
}

APValue 
VectorExprEvaluator::VisitCompoundLiteralExpr(const CompoundLiteralExpr *E) {
  return this->Visit(const_cast<Expr*>(E->getInitializer()));
}

APValue 
VectorExprEvaluator::VisitInitListExpr(const InitListExpr *E) {
  const VectorType *VT = E->getType()->getAsVectorType();
  unsigned NumInits = E->getNumInits();

  if (!VT || VT->getNumElements() != NumInits)
    return APValue();
  
  QualType EltTy = VT->getElementType();
  llvm::SmallVector<APValue, 4> Elements;

  for (unsigned i = 0; i < NumInits; i++) {
    if (EltTy->isIntegerType()) {
      llvm::APSInt sInt(32);
      if (!EvaluateInteger(E->getInit(i), sInt, Info))
        return APValue();
      Elements.push_back(APValue(sInt));
    } else {
      llvm::APFloat f(0.0);
      if (!EvaluateFloat(E->getInit(i), f, Info))
        return APValue();
      Elements.push_back(APValue(f));
    }
  }
  return APValue(&Elements[0], Elements.size());
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

  bool Extension(SourceLocation L, diag::kind D, const Expr *E) {
    Info.EvalResult.DiagLoc = L;
    Info.EvalResult.Diag = D;
    Info.EvalResult.DiagExpr = E;
    return true;  // still a constant.
  }

  bool Success(const llvm::APInt &I, const Expr *E) {
    Result = I;
    Result.setIsUnsigned(E->getType()->isUnsignedIntegerType());
    return true;
  }

  bool Success(uint64_t Value, const Expr *E) {
    Result = Info.Ctx.MakeIntValue(Value, E->getType());
    return true;
  }

  bool Error(SourceLocation L, diag::kind D, const Expr *E) {
    // If this is in an unevaluated portion of the subexpression, ignore the
    // error.
    if (Info.ShortCircuit) {
      // If error is ignored because the value isn't evaluated, get the real
      // type at least to prevent errors downstream.
      return Success(0, E);
    }
    
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
  bool VisitDeclRefExpr(const DeclRefExpr *E);
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

  bool VisitUnaryTypeTraitExpr(const UnaryTypeTraitExpr *E) {
    return Success(E->EvaluateTrait(), E);
  }

private:
  unsigned GetAlignOfExpr(const Expr *E);
  unsigned GetAlignOfType(QualType T);
};
} // end anonymous namespace

static bool EvaluateInteger(const Expr* E, APSInt &Result, EvalInfo &Info) {
  return IntExprEvaluator(Info, Result).Visit(const_cast<Expr*>(E));
}

bool IntExprEvaluator::VisitDeclRefExpr(const DeclRefExpr *E) {
  // Enums are integer constant exprs.
  if (const EnumConstantDecl *D = dyn_cast<EnumConstantDecl>(E->getDecl())) {
    Result = D->getInitVal();
    // FIXME: This is an ugly hack around the fact that enums don't set their
    // signedness consistently; see PR3173
    Result.setIsUnsigned(!E->getType()->isSignedIntegerType());
    return true;
  }

  // In C++, const, non-volatile integers initialized with ICEs are ICEs.
  if (Info.Ctx.getLangOptions().CPlusPlus &&
      E->getType().getCVRQualifiers() == QualType::Const) {
    if (const VarDecl *D = dyn_cast<VarDecl>(E->getDecl())) {
      if (const Expr *Init = D->getInit())
        return Visit(const_cast<Expr*>(Init));
    }
  }

  // Otherwise, random variable references are not constants.
  return Error(E->getLocStart(), diag::note_invalid_subexpr_in_ice, E);
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
  switch (E->isBuiltinCall(Info.Ctx)) {
  default:
    return Error(E->getLocStart(), diag::note_invalid_subexpr_in_ice, E);
  case Builtin::BI__builtin_classify_type:
    return Success(EvaluateBuiltinClassifyType(E), E);
    
  case Builtin::BI__builtin_constant_p:
    // __builtin_constant_p always has one operand: it returns true if that
    // operand can be folded, false otherwise.
    return Success(E->getArg(0)->isEvaluatable(Info.Ctx), E);
  }
}

bool IntExprEvaluator::VisitBinaryOperator(const BinaryOperator *E) {
  if (E->getOpcode() == BinaryOperator::Comma) {
    if (!Visit(E->getRHS()))
      return false;

    if (!Info.ShortCircuit) {
      // If we can't evaluate the LHS, it must be because it has 
      // side effects.
      if (!E->getLHS()->isEvaluatable(Info.Ctx))
        Info.EvalResult.HasSideEffects = true;
      
      return Extension(E->getOperatorLoc(), diag::note_comma_in_ice, E);
    }

    return true;
  }

  if (E->isLogicalOp()) {
    // These need to be handled specially because the operands aren't
    // necessarily integral
    bool lhsResult, rhsResult;
    
    if (HandleConversionToBool(E->getLHS(), lhsResult, Info)) {
      // We were able to evaluate the LHS, see if we can get away with not
      // evaluating the RHS: 0 && X -> 0, 1 || X -> 1
      if (lhsResult == (E->getOpcode() == BinaryOperator::LOr) || 
          !lhsResult == (E->getOpcode() == BinaryOperator::LAnd)) {
        Result = Info.Ctx.MakeIntValue(lhsResult, E->getType());
        
        Info.ShortCircuit++;
        bool rhsEvaluated = HandleConversionToBool(E->getRHS(), rhsResult, Info);
        Info.ShortCircuit--;
        
        if (rhsEvaluated)
          return true;
        
        // FIXME: Return an extension warning saying that the RHS could not be
        // evaluated.
        return true;
      }

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
  
  if (E->getOpcode() == BinaryOperator::Sub) {
    if (LHSTy->isPointerType() && RHSTy->isPointerType()) {
      APValue LHSValue;
      if (!EvaluatePointer(E->getLHS(), LHSValue, Info))
        return false;
      
      APValue RHSValue;
      if (!EvaluatePointer(E->getRHS(), RHSValue, Info))
        return false;
      
      // FIXME: Is this correct? What if only one of the operands has a base?
      if (LHSValue.getLValueBase() || RHSValue.getLValueBase())
        return false;
      
      const QualType Type = E->getLHS()->getType();
      const QualType ElementType = Type->getAsPointerType()->getPointeeType();

      uint64_t D = LHSValue.getLValueOffset() - RHSValue.getLValueOffset();
      D /= Info.Ctx.getTypeSize(ElementType) / 8;
      
      return Success(D, E);
    }
  }
  if (!LHSTy->isIntegralType() ||
      !RHSTy->isIntegralType()) {
    // We can't continue from here for non-integral types, and they
    // could potentially confuse the following operations.
    // FIXME: Deal with EQ and friends.
    return false;
  }

  // The LHS of a constant expr is always evaluated and needed.
  if (!Visit(E->getLHS())) {
    return false; // error in subexpression.
  }

  llvm::APSInt RHS;
  if (!EvaluateInteger(E->getRHS(), RHS, Info))
    return false;

  switch (E->getOpcode()) {
  default:
    return Error(E->getOperatorLoc(), diag::note_invalid_subexpr_in_ice, E);
  case BinaryOperator::Mul: Result *= RHS; return true;
  case BinaryOperator::Add: Result += RHS; return true;
  case BinaryOperator::Sub: Result -= RHS; return true;
  case BinaryOperator::And: Result &= RHS; return true;
  case BinaryOperator::Xor: Result ^= RHS; return true;
  case BinaryOperator::Or:  Result |= RHS; return true;
  case BinaryOperator::Div:
    if (RHS == 0)
      return Error(E->getOperatorLoc(), diag::note_expr_divide_by_zero, E);
    Result /= RHS;
    break;
  case BinaryOperator::Rem:
    if (RHS == 0)
      return Error(E->getOperatorLoc(), diag::note_expr_divide_by_zero, E);
    Result %= RHS;
    break;
  case BinaryOperator::Shl:
    // FIXME: Warn about out of range shift amounts!
    Result <<= (unsigned)RHS.getLimitedValue(Result.getBitWidth()-1);
    break;
  case BinaryOperator::Shr:
    Result >>= (unsigned)RHS.getLimitedValue(Result.getBitWidth()-1);
    break;
      
  case BinaryOperator::LT: return Success(Result < RHS, E);
  case BinaryOperator::GT: return Success(Result > RHS, E);
  case BinaryOperator::LE: return Success(Result <= RHS, E);
  case BinaryOperator::GE: return Success(Result >= RHS, E);
  case BinaryOperator::EQ: return Success(Result == RHS, E);
  case BinaryOperator::NE: return Success(Result != RHS, E);
  }

  Result.setIsUnsigned(E->getType()->isUnsignedIntegerType());
  return true;
}

bool IntExprEvaluator::VisitConditionalOperator(const ConditionalOperator *E) {
  bool Cond;
  if (!HandleConversionToBool(E->getCond(), Cond, Info))
    return false;

  return Visit(Cond ? E->getTrueExpr() : E->getFalseExpr());
}

unsigned IntExprEvaluator::GetAlignOfType(QualType T) {
  const Type *Ty = Info.Ctx.getCanonicalType(T).getTypePtr();
  
  // __alignof__(void) = 1 as a gcc extension.
  if (Ty->isVoidType())
    return 1;
  
  // GCC extension: alignof(function) = 4.
  // FIXME: AlignOf shouldn't be unconditionally 4!  It should listen to the
  // attribute(align) directive.
  if (Ty->isFunctionType())
    return 4;
  
  if (const ExtQualType *EXTQT = dyn_cast<ExtQualType>(Ty))
    return GetAlignOfType(QualType(EXTQT->getBaseType(), 0));

  // alignof VLA/incomplete array.
  if (const ArrayType *VAT = dyn_cast<ArrayType>(Ty))
    return GetAlignOfType(VAT->getElementType());
  
  // sizeof (objc class)?
  if (isa<ObjCInterfaceType>(Ty))
    return 1;  // FIXME: This probably isn't right.

  // Get information about the alignment.
  unsigned CharSize = Info.Ctx.Target.getCharWidth();
  return Info.Ctx.getTypeAlign(Ty) / CharSize;
}

unsigned IntExprEvaluator::GetAlignOfExpr(const Expr *E) {
  E = E->IgnoreParens();

  // alignof decl is always accepted, even if it doesn't make sense: we default
  // to 1 in those cases. 
  if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E))
    return Info.Ctx.getDeclAlignInBytes(DRE->getDecl());
    
  if (const MemberExpr *ME = dyn_cast<MemberExpr>(E))
    return Info.Ctx.getDeclAlignInBytes(ME->getMemberDecl());

  return GetAlignOfType(E->getType());
}


/// VisitSizeAlignOfExpr - Evaluate a sizeof or alignof with a result as the
/// expression's type.
bool IntExprEvaluator::VisitSizeOfAlignOfExpr(const SizeOfAlignOfExpr *E) {
  QualType DstTy = E->getType();

  // Handle alignof separately.
  if (!E->isSizeOf()) {
    if (E->isArgumentType())
      return Success(GetAlignOfType(E->getArgumentType()), E);
    else
      return Success(GetAlignOfExpr(E->getArgumentExpr()), E);
  }
  
  QualType SrcTy = E->getTypeOfArgument();

  // sizeof(void), __alignof__(void), sizeof(function) = 1 as a gcc
  // extension.
  if (SrcTy->isVoidType() || SrcTy->isFunctionType())
    return Success(1, E);
  
  // sizeof(vla) is not a constantexpr: C99 6.5.3.4p2.
  if (!SrcTy->isConstantSizeType())
    return false;

  if (SrcTy->isObjCInterfaceType()) {
    // Slightly unusual case: the size of an ObjC interface type is the
    // size of the class.  This code intentionally falls through to the normal
    // case.
    ObjCInterfaceDecl *OI = SrcTy->getAsObjCInterfaceType()->getDecl();
    RecordDecl *RD = const_cast<RecordDecl*>(Info.Ctx.addRecordToClass(OI));
    SrcTy = Info.Ctx.getTagDeclType(static_cast<TagDecl*>(RD));
  }

  // Get information about the size.
  unsigned CharSize = Info.Ctx.Target.getCharWidth();
  return Success(Info.Ctx.getTypeSize(SrcTy) / CharSize, E);
}

bool IntExprEvaluator::VisitUnaryOperator(const UnaryOperator *E) {
  // Special case unary operators that do not need their subexpression
  // evaluated.  offsetof/sizeof/alignof are all special.
  if (E->isOffsetOfOp())
    return Success(E->evaluateOffsetOf(Info.Ctx), E);

  if (E->getOpcode() == UnaryOperator::LNot) {
    // LNot's operand isn't necessarily an integer, so we handle it specially.
    bool bres;
    if (!HandleConversionToBool(E->getSubExpr(), bres, Info))
      return false;
    return Success(!bres, E);
  }

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
    break;
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
bool IntExprEvaluator::VisitCastExpr(CastExpr *E) {
  Expr *SubExpr = E->getSubExpr();
  QualType DestType = E->getType();

  if (DestType->isBooleanType()) {
    bool BoolResult;
    if (!HandleConversionToBool(SubExpr, BoolResult, Info))
      return false;
    return Success(BoolResult, E);
  }

  // Handle simple integer->integer casts.
  if (SubExpr->getType()->isIntegralType()) {
    if (!Visit(SubExpr))
      return false;

    Result = HandleIntToIntCast(DestType, SubExpr->getType(), Result, Info.Ctx);
    return true;
  }
  
  // FIXME: Clean this up!
  if (SubExpr->getType()->isPointerType()) {
    APValue LV;
    if (!EvaluatePointer(SubExpr, LV, Info))
      return false;

    if (LV.getLValueBase())
      return false;

    return Success(LV.getLValueOffset(), E);
  }

  if (!SubExpr->getType()->isRealFloatingType())
    return Error(E->getExprLoc(), diag::note_invalid_subexpr_in_ice, E);

  APFloat F(0.0);
  if (!EvaluateFloat(SubExpr, F, Info))
    return Error(E->getExprLoc(), diag::note_invalid_subexpr_in_ice, E);
  
  Result = HandleFloatToIntCast(DestType, SubExpr->getType(), F, Info.Ctx);
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
  
  if (SubExpr->getType()->isIntegralType()) {
    APSInt IntResult;
    if (!EvaluateInteger(E, IntResult, Info))
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

  return false;
}

bool FloatExprEvaluator::VisitCXXZeroInitValueExpr(CXXZeroInitValueExpr *E) {
  Result = APFloat::getZero(Info.Ctx.getFloatTypeSemantics(E->getType()));
  return true;
}

//===----------------------------------------------------------------------===//
// Complex Evaluation (for float and integer)
//===----------------------------------------------------------------------===//

namespace {
class VISIBILITY_HIDDEN ComplexExprEvaluator
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
    QualType EltType = E->getType()->getAsComplexType()->getElementType();
    QualType SubType = SubExpr->getType();

    if (SubType->isRealFloatingType()) {
      APFloat Result(0.0);
                     
      if (!EvaluateFloat(SubExpr, Result, Info))
        return APValue();
      
      // Apply float conversion if necessary.
      Result = HandleFloatToFloatCast(EltType, SubType, Result, Info.Ctx);
      return APValue(Result, 
                     APFloat(Result.getSemantics(), APFloat::fcZero, false));
    } else if (SubType->isIntegerType()) {
      APSInt Result;
                     
      if (!EvaluateInteger(SubExpr, Result, Info))
        return APValue();

      // Apply integer conversion if necessary.
      Result = HandleIntToIntCast(EltType, SubType, Result, Info.Ctx);
      llvm::APSInt Zero(Result.getBitWidth(), !Result.isSigned());
      Zero = 0;
      return APValue(Result, Zero);
    } else if (const ComplexType *CT = SubType->getAsComplexType()) {
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

};
} // end anonymous namespace

static bool EvaluateComplex(const Expr *E, APValue &Result, EvalInfo &Info)
{
  Result = ComplexExprEvaluator(Info).Visit(const_cast<Expr*>(E));
  assert((!Result.isComplexFloat() ||
          (&Result.getComplexFloatReal().getSemantics() == 
           &Result.getComplexFloatImag().getSemantics())) && 
         "Invalid complex evaluation.");
  return Result.isComplexFloat() || Result.isComplexInt();
}

APValue ComplexExprEvaluator::VisitBinaryOperator(const BinaryOperator *E)
{
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
    llvm::APSInt sInt(32);
    if (!EvaluateInteger(this, sInt, Info))
      return false;
    
    Result.Val = APValue(sInt);
  } else if (getType()->isPointerType()
             || getType()->isBlockPointerType()) {
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

/// isEvaluatable - Call Evaluate to see if this expression can be constant
/// folded, but discard the result.
bool Expr::isEvaluatable(ASTContext &Ctx) const {
  EvalResult Result;
  return Evaluate(Result, Ctx) && !Result.HasSideEffects;
}

APSInt Expr::EvaluateAsInt(ASTContext &Ctx) const {
  EvalResult EvalResult;
  bool Result = Evaluate(EvalResult, Ctx);
  Result = Result;
  assert(Result && "Could not evaluate expression");
  assert(EvalResult.Val.isInt() && "Expression did not evaluate to integer");

  return EvalResult.Val.getInt();
}
