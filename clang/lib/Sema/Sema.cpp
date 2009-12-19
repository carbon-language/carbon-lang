//===--- Sema.cpp - AST Builder and Semantic Analysis Implementation ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the actions class which performs semantic analysis and
// builds an AST out of a parse stream.
//
//===----------------------------------------------------------------------===//

#include "Sema.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/APFloat.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Expr.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/PartialDiagnostic.h"
#include "clang/Basic/TargetInfo.h"
using namespace clang;

/// Determines whether we should have an a.k.a. clause when
/// pretty-printing a type.  There are three main criteria:
///
/// 1) Some types provide very minimal sugar that doesn't impede the
///    user's understanding --- for example, elaborated type
///    specifiers.  If this is all the sugar we see, we don't want an
///    a.k.a. clause.
/// 2) Some types are technically sugared but are much more familiar
///    when seen in their sugared form --- for example, va_list,
///    vector types, and the magic Objective C types.  We don't
///    want to desugar these, even if we do produce an a.k.a. clause.
/// 3) Some types may have already been desugared previously in this diagnostic.
///    if this is the case, doing another "aka" would just be clutter.
///
static bool ShouldAKA(ASTContext &Context, QualType QT,
                      const Diagnostic::ArgumentValue *PrevArgs,
                      unsigned NumPrevArgs,
                      QualType &DesugaredQT) {
  QualType InputTy = QT;
  
  bool AKA = false;
  QualifierCollector Qc;

  while (true) {
    const Type *Ty = Qc.strip(QT);

    // Don't aka just because we saw an elaborated type...
    if (isa<ElaboratedType>(Ty)) {
      QT = cast<ElaboratedType>(Ty)->desugar();
      continue;
    }

    // ...or a qualified name type...
    if (isa<QualifiedNameType>(Ty)) {
      QT = cast<QualifiedNameType>(Ty)->desugar();
      continue;
    }

    // ...or a substituted template type parameter.
    if (isa<SubstTemplateTypeParmType>(Ty)) {
      QT = cast<SubstTemplateTypeParmType>(Ty)->desugar();
      continue;
    }
      
    // Don't desugar template specializations. 
    if (isa<TemplateSpecializationType>(Ty))
      break;

    // Don't desugar magic Objective-C types.
    if (QualType(Ty,0) == Context.getObjCIdType() ||
        QualType(Ty,0) == Context.getObjCClassType() ||
        QualType(Ty,0) == Context.getObjCSelType() ||
        QualType(Ty,0) == Context.getObjCProtoType())
      break;

    // Don't desugar va_list.
    if (QualType(Ty,0) == Context.getBuiltinVaListType())
      break;

    // Otherwise, do a single-step desugar.
    QualType Underlying;
    bool IsSugar = false;
    switch (Ty->getTypeClass()) {
#define ABSTRACT_TYPE(Class, Base)
#define TYPE(Class, Base) \
    case Type::Class: { \
      const Class##Type *CTy = cast<Class##Type>(Ty); \
      if (CTy->isSugared()) { \
        IsSugar = true; \
        Underlying = CTy->desugar(); \
      } \
      break; \
    }
#include "clang/AST/TypeNodes.def"
    }

    // If it wasn't sugared, we're done.
    if (!IsSugar)
      break;

    // If the desugared type is a vector type, we don't want to expand
    // it, it will turn into an attribute mess. People want their "vec4".
    if (isa<VectorType>(Underlying))
      break;

    // Otherwise, we're tearing through something opaque; note that
    // we'll eventually need an a.k.a. clause and keep going.
    AKA = true;
    QT = Underlying;
    continue;
  }

  // If we never tore through opaque sugar, don't print aka.
  if (!AKA) return false;

  // If we did, check to see if we already desugared this type in this
  // diagnostic.  If so, don't do it again.
  for (unsigned i = 0; i != NumPrevArgs; ++i) {
    // TODO: Handle ak_declcontext case.
    if (PrevArgs[i].first == Diagnostic::ak_qualtype) {
      void *Ptr = (void*)PrevArgs[i].second;
      QualType PrevTy(QualType::getFromOpaquePtr(Ptr));
      if (PrevTy == InputTy)
        return false;
    }
  }
  
  DesugaredQT = Qc.apply(QT);
  return true;
}

/// \brief Convert the given type to a string suitable for printing as part of 
/// a diagnostic. 
///
/// \param Context the context in which the type was allocated
/// \param Ty the type to print
static std::string
ConvertTypeToDiagnosticString(ASTContext &Context, QualType Ty,
                              const Diagnostic::ArgumentValue *PrevArgs,
                              unsigned NumPrevArgs) {
  // FIXME: Playing with std::string is really slow.
  std::string S = Ty.getAsString(Context.PrintingPolicy);
  
  // Consider producing an a.k.a. clause if removing all the direct
  // sugar gives us something "significantly different".

  QualType DesugaredTy;
  if (ShouldAKA(Context, Ty, PrevArgs, NumPrevArgs, DesugaredTy)) {
    S = "'"+S+"' (aka '";
    S += DesugaredTy.getAsString(Context.PrintingPolicy);
    S += "')";
    return S;
  }

  S = "'" + S + "'";
  return S;
}
                                       
/// ConvertQualTypeToStringFn - This function is used to pretty print the
/// specified QualType as a string in diagnostics.
static void ConvertArgToStringFn(Diagnostic::ArgumentKind Kind, intptr_t Val,
                                 const char *Modifier, unsigned ModLen,
                                 const char *Argument, unsigned ArgLen,
                                 const Diagnostic::ArgumentValue *PrevArgs,
                                 unsigned NumPrevArgs,
                                 llvm::SmallVectorImpl<char> &Output,
                                 void *Cookie) {
  ASTContext &Context = *static_cast<ASTContext*>(Cookie);

  std::string S;
  bool NeedQuotes = true;
  
  switch (Kind) {
  default: assert(0 && "unknown ArgumentKind");
  case Diagnostic::ak_qualtype: {
    assert(ModLen == 0 && ArgLen == 0 &&
           "Invalid modifier for QualType argument");

    QualType Ty(QualType::getFromOpaquePtr(reinterpret_cast<void*>(Val)));
    S = ConvertTypeToDiagnosticString(Context, Ty, PrevArgs, NumPrevArgs);
    NeedQuotes = false;
    break;
  }
  case Diagnostic::ak_declarationname: {
    DeclarationName N = DeclarationName::getFromOpaqueInteger(Val);
    S = N.getAsString();

    if (ModLen == 9 && !memcmp(Modifier, "objcclass", 9) && ArgLen == 0)
      S = '+' + S;
    else if (ModLen == 12 && !memcmp(Modifier, "objcinstance", 12) && ArgLen==0)
      S = '-' + S;
    else
      assert(ModLen == 0 && ArgLen == 0 &&
             "Invalid modifier for DeclarationName argument");
    break;
  }
  case Diagnostic::ak_nameddecl: {
    bool Qualified;
    if (ModLen == 1 && Modifier[0] == 'q' && ArgLen == 0)
      Qualified = true;
    else {
      assert(ModLen == 0 && ArgLen == 0 &&
           "Invalid modifier for NamedDecl* argument");
      Qualified = false;
    }
    reinterpret_cast<NamedDecl*>(Val)->
      getNameForDiagnostic(S, Context.PrintingPolicy, Qualified);
    break;
  }
  case Diagnostic::ak_nestednamespec: {
    llvm::raw_string_ostream OS(S);
    reinterpret_cast<NestedNameSpecifier*>(Val)->print(OS,
                                                       Context.PrintingPolicy);
    NeedQuotes = false;
    break;
  }
  case Diagnostic::ak_declcontext: {
    DeclContext *DC = reinterpret_cast<DeclContext *> (Val);
    assert(DC && "Should never have a null declaration context");
    
    if (DC->isTranslationUnit()) {
      // FIXME: Get these strings from some localized place
      if (Context.getLangOptions().CPlusPlus)
        S = "the global namespace";
      else
        S = "the global scope";
    } else if (TypeDecl *Type = dyn_cast<TypeDecl>(DC)) {
      S = ConvertTypeToDiagnosticString(Context, Context.getTypeDeclType(Type),
                                        PrevArgs, NumPrevArgs);
    } else {
      // FIXME: Get these strings from some localized place
      NamedDecl *ND = cast<NamedDecl>(DC);
      if (isa<NamespaceDecl>(ND))
        S += "namespace ";
      else if (isa<ObjCMethodDecl>(ND))
        S += "method ";
      else if (isa<FunctionDecl>(ND))
        S += "function ";

      S += "'";
      ND->getNameForDiagnostic(S, Context.PrintingPolicy, true);
      S += "'";
    }
    NeedQuotes = false;
    break;
  }
  }

  if (NeedQuotes)
    Output.push_back('\'');
  
  Output.append(S.begin(), S.end());
  
  if (NeedQuotes)
    Output.push_back('\'');
}


static inline RecordDecl *CreateStructDecl(ASTContext &C, const char *Name) {
  if (C.getLangOptions().CPlusPlus)
    return CXXRecordDecl::Create(C, TagDecl::TK_struct,
                                 C.getTranslationUnitDecl(),
                                 SourceLocation(), &C.Idents.get(Name));

  return RecordDecl::Create(C, TagDecl::TK_struct,
                            C.getTranslationUnitDecl(),
                            SourceLocation(), &C.Idents.get(Name));
}

void Sema::ActOnTranslationUnitScope(SourceLocation Loc, Scope *S) {
  TUScope = S;
  PushDeclContext(S, Context.getTranslationUnitDecl());

  if (PP.getTargetInfo().getPointerWidth(0) >= 64) {
    TypeSourceInfo *TInfo;

    // Install [u]int128_t for 64-bit targets.
    TInfo = Context.getTrivialTypeSourceInfo(Context.Int128Ty);
    PushOnScopeChains(TypedefDecl::Create(Context, CurContext,
                                          SourceLocation(),
                                          &Context.Idents.get("__int128_t"),
                                          TInfo), TUScope);

    TInfo = Context.getTrivialTypeSourceInfo(Context.UnsignedInt128Ty);
    PushOnScopeChains(TypedefDecl::Create(Context, CurContext,
                                          SourceLocation(),
                                          &Context.Idents.get("__uint128_t"),
                                          TInfo), TUScope);
  }


  if (!PP.getLangOptions().ObjC1) return;

  // Built-in ObjC types may already be set by PCHReader (hence isNull checks).
  if (Context.getObjCSelType().isNull()) {
    // Create the built-in typedef for 'SEL'.
    QualType SelT = Context.getPointerType(Context.ObjCBuiltinSelTy);
    TypeSourceInfo *SelInfo = Context.getTrivialTypeSourceInfo(SelT);
    TypedefDecl *SelTypedef
      = TypedefDecl::Create(Context, CurContext, SourceLocation(),
                            &Context.Idents.get("SEL"), SelInfo);
    PushOnScopeChains(SelTypedef, TUScope);
    Context.setObjCSelType(Context.getTypeDeclType(SelTypedef));
    Context.ObjCSelRedefinitionType = Context.getObjCSelType();
  }

  // Synthesize "@class Protocol;
  if (Context.getObjCProtoType().isNull()) {
    ObjCInterfaceDecl *ProtocolDecl =
      ObjCInterfaceDecl::Create(Context, CurContext, SourceLocation(),
                                &Context.Idents.get("Protocol"),
                                SourceLocation(), true);
    Context.setObjCProtoType(Context.getObjCInterfaceType(ProtocolDecl));
    PushOnScopeChains(ProtocolDecl, TUScope, false);
  }
  // Create the built-in typedef for 'id'.
  if (Context.getObjCIdType().isNull()) {
    QualType IdT = Context.getObjCObjectPointerType(Context.ObjCBuiltinIdTy);
    TypeSourceInfo *IdInfo = Context.getTrivialTypeSourceInfo(IdT);
    TypedefDecl *IdTypedef
      = TypedefDecl::Create(Context, CurContext, SourceLocation(),
                            &Context.Idents.get("id"), IdInfo);
    PushOnScopeChains(IdTypedef, TUScope);
    Context.setObjCIdType(Context.getTypeDeclType(IdTypedef));
    Context.ObjCIdRedefinitionType = Context.getObjCIdType();
  }
  // Create the built-in typedef for 'Class'.
  if (Context.getObjCClassType().isNull()) {
    QualType ClassType
      = Context.getObjCObjectPointerType(Context.ObjCBuiltinClassTy);
    TypeSourceInfo *ClassInfo = Context.getTrivialTypeSourceInfo(ClassType);
    TypedefDecl *ClassTypedef
      = TypedefDecl::Create(Context, CurContext, SourceLocation(),
                            &Context.Idents.get("Class"), ClassInfo);
    PushOnScopeChains(ClassTypedef, TUScope);
    Context.setObjCClassType(Context.getTypeDeclType(ClassTypedef));
    Context.ObjCClassRedefinitionType = Context.getObjCClassType();
  }
}

Sema::Sema(Preprocessor &pp, ASTContext &ctxt, ASTConsumer &consumer,
           bool CompleteTranslationUnit,
           CodeCompleteConsumer *CodeCompleter)
  : LangOpts(pp.getLangOptions()), PP(pp), Context(ctxt), Consumer(consumer),
    Diags(PP.getDiagnostics()), SourceMgr(PP.getSourceManager()),
    ExternalSource(0), CodeCompleter(CodeCompleter), CurContext(0), 
    CurBlock(0), PackContext(0), ParsingDeclDepth(0),
    IdResolver(pp.getLangOptions()), StdNamespace(0), StdBadAlloc(0),
    GlobalNewDeleteDeclared(false), 
    CompleteTranslationUnit(CompleteTranslationUnit),
    NumSFINAEErrors(0), NonInstantiationEntries(0), 
    CurrentInstantiationScope(0) 
{
  TUScope = 0;
  if (getLangOptions().CPlusPlus)
    FieldCollector.reset(new CXXFieldCollector());

  // Tell diagnostics how to render things from the AST library.
  PP.getDiagnostics().SetArgToStringFn(ConvertArgToStringFn, &Context);

  ExprEvalContexts.push_back(
                  ExpressionEvaluationContextRecord(PotentiallyEvaluated, 0));
}

/// Retrieves the width and signedness of the given integer type,
/// or returns false if it is not an integer type.
///
/// \param T must be canonical
static bool getIntProperties(ASTContext &C, const Type *T,
                             unsigned &BitWidth, bool &Signed) {
  assert(T->isCanonicalUnqualified());

  if (const VectorType *VT = dyn_cast<VectorType>(T))
    T = VT->getElementType().getTypePtr();
  if (const ComplexType *CT = dyn_cast<ComplexType>(T))
    T = CT->getElementType().getTypePtr();

  if (const BuiltinType *BT = dyn_cast<BuiltinType>(T)) {
    if (!BT->isInteger()) return false;

    BitWidth = C.getIntWidth(QualType(T, 0));
    Signed = BT->isSignedInteger();
    return true;
  }

  if (const FixedWidthIntType *FWIT = dyn_cast<FixedWidthIntType>(T)) {
    BitWidth = FWIT->getWidth();
    Signed = FWIT->isSigned();
    return true;
  }

  return false;
}

/// Checks whether the given value will have the same value if it it
/// is truncated to the given width, then extended back to the
/// original width.
static bool IsSameIntAfterCast(const llvm::APSInt &value,
                               unsigned TargetWidth) {
  unsigned SourceWidth = value.getBitWidth();
  llvm::APSInt truncated = value;
  truncated.trunc(TargetWidth);
  truncated.extend(SourceWidth);
  return (truncated == value);
}

/// Checks whether the given value will have the same value if it
/// is truncated to the given width, then extended back to the original
/// width.
///
/// The value might be a vector or a complex.
static bool IsSameIntAfterCast(const APValue &value, unsigned TargetWidth) {
  if (value.isInt())
    return IsSameIntAfterCast(value.getInt(), TargetWidth);

  if (value.isVector()) {
    for (unsigned i = 0, e = value.getVectorLength(); i != e; ++i)
      if (!IsSameIntAfterCast(value.getVectorElt(i), TargetWidth))
        return false;
    return true;
  }

  if (value.isComplexInt()) {
    return IsSameIntAfterCast(value.getComplexIntReal(), TargetWidth) &&
           IsSameIntAfterCast(value.getComplexIntImag(), TargetWidth);
  }

  // This can happen with lossless casts to intptr_t of "based" lvalues.
  // Assume it might use arbitrary bits.
  assert(value.isLValue());
  return false;
}
                               

/// Checks whether the given value, which currently has the given
/// source semantics, has the same value when coerced through the
/// target semantics.
static bool IsSameFloatAfterCast(const llvm::APFloat &value,
                                 const llvm::fltSemantics &Src,
                                 const llvm::fltSemantics &Tgt) {
  llvm::APFloat truncated = value;

  bool ignored;
  truncated.convert(Src, llvm::APFloat::rmNearestTiesToEven, &ignored);
  truncated.convert(Tgt, llvm::APFloat::rmNearestTiesToEven, &ignored);

  return truncated.bitwiseIsEqual(value);
}

/// Checks whether the given value, which currently has the given
/// source semantics, has the same value when coerced through the
/// target semantics.
///
/// The value might be a vector of floats (or a complex number).
static bool IsSameFloatAfterCast(const APValue &value,
                                 const llvm::fltSemantics &Src,
                                 const llvm::fltSemantics &Tgt) {
  if (value.isFloat())
    return IsSameFloatAfterCast(value.getFloat(), Src, Tgt);

  if (value.isVector()) {
    for (unsigned i = 0, e = value.getVectorLength(); i != e; ++i)
      if (!IsSameFloatAfterCast(value.getVectorElt(i), Src, Tgt))
        return false;
    return true;
  }

  assert(value.isComplexFloat());
  return (IsSameFloatAfterCast(value.getComplexFloatReal(), Src, Tgt) &&
          IsSameFloatAfterCast(value.getComplexFloatImag(), Src, Tgt));
}

/// Determines if it's reasonable for the given expression to be truncated
/// down to the given integer width.
/// * Boolean expressions are automatically white-listed.
/// * Arithmetic operations on implicitly-promoted operands of the
///   target width or less are okay --- not because the results are
///   actually guaranteed to fit within the width, but because the
///   user is effectively pretending that the operations are closed
///   within the implicitly-promoted type.
static bool IsExprValueWithinWidth(ASTContext &C, Expr *E, unsigned Width) {
  E = E->IgnoreParens();

#ifndef NDEBUG
  {
    const Type *ETy = E->getType()->getCanonicalTypeInternal().getTypePtr();
    unsigned EWidth;
    bool ESigned;

    if (!getIntProperties(C, ETy, EWidth, ESigned))
      assert(0 && "expression not of integer type");

    // The caller should never let this happen.
    assert(EWidth > Width && "called on expr whose type is too small");
  }
#endif

  // Strip implicit casts off.
  while (isa<ImplicitCastExpr>(E)) {
    E = cast<ImplicitCastExpr>(E)->getSubExpr();

    const Type *ETy = E->getType()->getCanonicalTypeInternal().getTypePtr();

    unsigned EWidth;
    bool ESigned;
    if (!getIntProperties(C, ETy, EWidth, ESigned))
      return false;

    if (EWidth <= Width)
      return true;
  }

  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(E)) {
    switch (BO->getOpcode()) {

    // Boolean-valued operations are white-listed.
    case BinaryOperator::LAnd:
    case BinaryOperator::LOr:
    case BinaryOperator::LT:
    case BinaryOperator::GT:
    case BinaryOperator::LE:
    case BinaryOperator::GE:
    case BinaryOperator::EQ:
    case BinaryOperator::NE:
      return true;

    // Operations with opaque sources are black-listed.
    case BinaryOperator::PtrMemD:
    case BinaryOperator::PtrMemI:
      return false;

    // Left shift gets black-listed based on a judgement call.
    case BinaryOperator::Shl:
      return false;

    // Various special cases.
    case BinaryOperator::Shr:
      return IsExprValueWithinWidth(C, BO->getLHS(), Width);
    case BinaryOperator::Comma:
      return IsExprValueWithinWidth(C, BO->getRHS(), Width);
    case BinaryOperator::Sub:
      if (BO->getLHS()->getType()->isPointerType())
        return false;
      // fallthrough
      
    // Any other operator is okay if the operands are
    // promoted from expressions of appropriate size.
    default:
      return IsExprValueWithinWidth(C, BO->getLHS(), Width) &&
             IsExprValueWithinWidth(C, BO->getRHS(), Width);
    }
  }

  if (UnaryOperator *UO = dyn_cast<UnaryOperator>(E)) {
    switch (UO->getOpcode()) {
    // Boolean-valued operations are white-listed.
    case UnaryOperator::LNot:
      return true;

    // Operations with opaque sources are black-listed.
    case UnaryOperator::Deref:
    case UnaryOperator::AddrOf: // should be impossible
      return false;

    case UnaryOperator::OffsetOf:
      return false;

    default:
      return IsExprValueWithinWidth(C, UO->getSubExpr(), Width);
    }
  }

  // Don't diagnose if the expression is an integer constant
  // whose value in the target type is the same as it was
  // in the original type.
  Expr::EvalResult result;
  if (E->Evaluate(result, C))
    if (IsSameIntAfterCast(result.Val, Width))
      return true;

  return false;
}

/// Diagnose an implicit cast;  purely a helper for CheckImplicitConversion.
static void DiagnoseImpCast(Sema &S, Expr *E, QualType T, unsigned diag) {
  S.Diag(E->getExprLoc(), diag) << E->getType() << T << E->getSourceRange();
}

/// Implements -Wconversion.
static void CheckImplicitConversion(Sema &S, Expr *E, QualType T) {
  // Don't diagnose in unevaluated contexts.
  if (S.ExprEvalContexts.back().Context == Sema::Unevaluated)
    return;

  // Don't diagnose for value-dependent expressions.
  if (E->isValueDependent())
    return;

  const Type *Source = S.Context.getCanonicalType(E->getType()).getTypePtr();
  const Type *Target = S.Context.getCanonicalType(T).getTypePtr();

  // Never diagnose implicit casts to bool.
  if (Target->isSpecificBuiltinType(BuiltinType::Bool))
    return;

  // Strip vector types.
  if (isa<VectorType>(Source)) {
    if (!isa<VectorType>(Target))
      return DiagnoseImpCast(S, E, T, diag::warn_impcast_vector_scalar);

    Source = cast<VectorType>(Source)->getElementType().getTypePtr();
    Target = cast<VectorType>(Target)->getElementType().getTypePtr();
  }

  // Strip complex types.
  if (isa<ComplexType>(Source)) {
    if (!isa<ComplexType>(Target))
      return DiagnoseImpCast(S, E, T, diag::warn_impcast_complex_scalar);

    Source = cast<ComplexType>(Source)->getElementType().getTypePtr();
    Target = cast<ComplexType>(Target)->getElementType().getTypePtr();
  }

  const BuiltinType *SourceBT = dyn_cast<BuiltinType>(Source);
  const BuiltinType *TargetBT = dyn_cast<BuiltinType>(Target);

  // If the source is floating point...
  if (SourceBT && SourceBT->isFloatingPoint()) {
    // ...and the target is floating point...
    if (TargetBT && TargetBT->isFloatingPoint()) {
      // ...then warn if we're dropping FP rank.

      // Builtin FP kinds are ordered by increasing FP rank.
      if (SourceBT->getKind() > TargetBT->getKind()) {
        // Don't warn about float constants that are precisely
        // representable in the target type.
        Expr::EvalResult result;
        if (E->Evaluate(result, S.Context)) {
          // Value might be a float, a float vector, or a float complex.
          if (IsSameFloatAfterCast(result.Val,
                     S.Context.getFloatTypeSemantics(QualType(TargetBT, 0)),
                     S.Context.getFloatTypeSemantics(QualType(SourceBT, 0))))
            return;
        }

        DiagnoseImpCast(S, E, T, diag::warn_impcast_float_precision);
      }
      return;
    }

    // If the target is integral, always warn.
    if ((TargetBT && TargetBT->isInteger()) ||
        isa<FixedWidthIntType>(Target))
      // TODO: don't warn for integer values?
      return DiagnoseImpCast(S, E, T, diag::warn_impcast_float_integer);

    return;
  }

  unsigned SourceWidth, TargetWidth;
  bool SourceSigned, TargetSigned;

  if (!getIntProperties(S.Context, Source, SourceWidth, SourceSigned) ||
      !getIntProperties(S.Context, Target, TargetWidth, TargetSigned))
    return;

  if (SourceWidth > TargetWidth) {
    if (IsExprValueWithinWidth(S.Context, E, TargetWidth))
      return;

    // People want to build with -Wshorten-64-to-32 and not -Wconversion
    // and by god we'll let them.
    if (SourceWidth == 64 && TargetWidth == 32)
      return DiagnoseImpCast(S, E, T, diag::warn_impcast_integer_64_32);
    return DiagnoseImpCast(S, E, T, diag::warn_impcast_integer_precision);
  }

  return;
}

/// ImpCastExprToType - If Expr is not of type 'Type', insert an implicit cast.
/// If there is already an implicit cast, merge into the existing one.
/// If isLvalue, the result of the cast is an lvalue.
void Sema::ImpCastExprToType(Expr *&Expr, QualType Ty,
                             CastExpr::CastKind Kind, bool isLvalue) {
  QualType ExprTy = Context.getCanonicalType(Expr->getType());
  QualType TypeTy = Context.getCanonicalType(Ty);

  if (ExprTy == TypeTy)
    return;

  if (Expr->getType()->isPointerType() && Ty->isPointerType()) {
    QualType ExprBaseType = cast<PointerType>(ExprTy)->getPointeeType();
    QualType BaseType = cast<PointerType>(TypeTy)->getPointeeType();
    if (ExprBaseType.getAddressSpace() != BaseType.getAddressSpace()) {
      Diag(Expr->getExprLoc(), diag::err_implicit_pointer_address_space_cast)
        << Expr->getSourceRange();
    }
  }

  CheckImplicitConversion(*this, Expr, Ty);

  if (ImplicitCastExpr *ImpCast = dyn_cast<ImplicitCastExpr>(Expr)) {
    if (ImpCast->getCastKind() == Kind) {
      ImpCast->setType(Ty);
      ImpCast->setLvalueCast(isLvalue);
      return;
    }
  }

  Expr = new (Context) ImplicitCastExpr(Ty, Kind, Expr, isLvalue);
}

void Sema::DeleteExpr(ExprTy *E) {
  if (E) static_cast<Expr*>(E)->Destroy(Context);
}
void Sema::DeleteStmt(StmtTy *S) {
  if (S) static_cast<Stmt*>(S)->Destroy(Context);
}

/// ActOnEndOfTranslationUnit - This is called at the very end of the
/// translation unit when EOF is reached and all but the top-level scope is
/// popped.
void Sema::ActOnEndOfTranslationUnit() {
  
  while (1) {
    // C++: Perform implicit template instantiations.
    //
    // FIXME: When we perform these implicit instantiations, we do not carefully
    // keep track of the point of instantiation (C++ [temp.point]). This means
    // that name lookup that occurs within the template instantiation will
    // always happen at the end of the translation unit, so it will find
    // some names that should not be found. Although this is common behavior
    // for C++ compilers, it is technically wrong. In the future, we either need
    // to be able to filter the results of name lookup or we need to perform
    // template instantiations earlier.
    PerformPendingImplicitInstantiations();
    
    /// If ProcessPendingClassesWithUnmarkedVirtualMembers ends up marking 
    /// any virtual member functions it might lead to more pending template
    /// instantiations, which is why we need to loop here.
    if (!ProcessPendingClassesWithUnmarkedVirtualMembers())
      break;
  }
  
  // Check for #pragma weak identifiers that were never declared
  // FIXME: This will cause diagnostics to be emitted in a non-determinstic
  // order!  Iterating over a densemap like this is bad.
  for (llvm::DenseMap<IdentifierInfo*,WeakInfo>::iterator
       I = WeakUndeclaredIdentifiers.begin(),
       E = WeakUndeclaredIdentifiers.end(); I != E; ++I) {
    if (I->second.getUsed()) continue;

    Diag(I->second.getLocation(), diag::warn_weak_identifier_undeclared)
      << I->first;
  }

  if (!CompleteTranslationUnit)
    return;

  // C99 6.9.2p2:
  //   A declaration of an identifier for an object that has file
  //   scope without an initializer, and without a storage-class
  //   specifier or with the storage-class specifier static,
  //   constitutes a tentative definition. If a translation unit
  //   contains one or more tentative definitions for an identifier,
  //   and the translation unit contains no external definition for
  //   that identifier, then the behavior is exactly as if the
  //   translation unit contains a file scope declaration of that
  //   identifier, with the composite type as of the end of the
  //   translation unit, with an initializer equal to 0.
  for (unsigned i = 0, e = TentativeDefinitionList.size(); i != e; ++i) {
    VarDecl *VD = TentativeDefinitions.lookup(TentativeDefinitionList[i]);

    // If the tentative definition was completed, it will be in the list, but
    // not the map.
    if (VD == 0 || VD->isInvalidDecl() || !VD->isTentativeDefinition(Context))
      continue;

    if (const IncompleteArrayType *ArrayT
        = Context.getAsIncompleteArrayType(VD->getType())) {
      if (RequireCompleteType(VD->getLocation(),
                              ArrayT->getElementType(),
                              diag::err_tentative_def_incomplete_type_arr)) {
        VD->setInvalidDecl();
        continue;
      }

      // Set the length of the array to 1 (C99 6.9.2p5).
      Diag(VD->getLocation(), diag::warn_tentative_incomplete_array);
      llvm::APInt One(Context.getTypeSize(Context.getSizeType()), true);
      QualType T = Context.getConstantArrayType(ArrayT->getElementType(),
                                                One, ArrayType::Normal, 0);
      VD->setType(T);
    } else if (RequireCompleteType(VD->getLocation(), VD->getType(),
                                   diag::err_tentative_def_incomplete_type))
      VD->setInvalidDecl();

    // Notify the consumer that we've completed a tentative definition.
    if (!VD->isInvalidDecl())
      Consumer.CompleteTentativeDefinition(VD);

  }
}


//===----------------------------------------------------------------------===//
// Helper functions.
//===----------------------------------------------------------------------===//

DeclContext *Sema::getFunctionLevelDeclContext() {
  DeclContext *DC = CurContext;

  while (isa<BlockDecl>(DC))
    DC = DC->getParent();

  return DC;
}

/// getCurFunctionDecl - If inside of a function body, this returns a pointer
/// to the function decl for the function being parsed.  If we're currently
/// in a 'block', this returns the containing context.
FunctionDecl *Sema::getCurFunctionDecl() {
  DeclContext *DC = getFunctionLevelDeclContext();
  return dyn_cast<FunctionDecl>(DC);
}

ObjCMethodDecl *Sema::getCurMethodDecl() {
  DeclContext *DC = getFunctionLevelDeclContext();
  return dyn_cast<ObjCMethodDecl>(DC);
}

NamedDecl *Sema::getCurFunctionOrMethodDecl() {
  DeclContext *DC = getFunctionLevelDeclContext();
  if (isa<ObjCMethodDecl>(DC) || isa<FunctionDecl>(DC))
    return cast<NamedDecl>(DC);
  return 0;
}

Sema::SemaDiagnosticBuilder::~SemaDiagnosticBuilder() {
  if (!this->Emit())
    return;

  // If this is not a note, and we're in a template instantiation
  // that is different from the last template instantiation where
  // we emitted an error, print a template instantiation
  // backtrace.
  if (!SemaRef.Diags.isBuiltinNote(DiagID) &&
      !SemaRef.ActiveTemplateInstantiations.empty() &&
      SemaRef.ActiveTemplateInstantiations.back()
        != SemaRef.LastTemplateInstantiationErrorContext) {
    SemaRef.PrintInstantiationStack();
    SemaRef.LastTemplateInstantiationErrorContext
      = SemaRef.ActiveTemplateInstantiations.back();
  }
}

Sema::SemaDiagnosticBuilder
Sema::Diag(SourceLocation Loc, const PartialDiagnostic& PD) {
  SemaDiagnosticBuilder Builder(Diag(Loc, PD.getDiagID()));
  PD.Emit(Builder);

  return Builder;
}

void Sema::ActOnComment(SourceRange Comment) {
  Context.Comments.push_back(Comment);
}

