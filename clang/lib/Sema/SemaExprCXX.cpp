//===--- SemaExprCXX.cpp - Semantic Analysis for Expressions --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for C++ expressions.
//
//===----------------------------------------------------------------------===//

#include "Sema.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ASTContext.h"
#include "clang/Parse/DeclSpec.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
using namespace clang;

/// ActOnCXXNamedCast - Parse {dynamic,static,reinterpret,const}_cast's.
Action::ExprResult
Sema::ActOnCXXNamedCast(SourceLocation OpLoc, tok::TokenKind Kind,
                        SourceLocation LAngleBracketLoc, TypeTy *Ty,
                        SourceLocation RAngleBracketLoc,
                        SourceLocation LParenLoc, ExprTy *E,
                        SourceLocation RParenLoc) {
  Expr *Ex = (Expr*)E;
  QualType DestType = QualType::getFromOpaquePtr(Ty);

  switch (Kind) {
  default: assert(0 && "Unknown C++ cast!");

  case tok::kw_const_cast:
    CheckConstCast(OpLoc, Ex, DestType);
    return new CXXConstCastExpr(DestType.getNonReferenceType(), Ex, 
                                DestType, OpLoc);

  case tok::kw_dynamic_cast:
    return new CXXDynamicCastExpr(DestType.getNonReferenceType(), Ex, 
                                  DestType, OpLoc);

  case tok::kw_reinterpret_cast:
    CheckReinterpretCast(OpLoc, Ex, DestType);
    return new CXXReinterpretCastExpr(DestType.getNonReferenceType(), Ex, 
                                      DestType, OpLoc);

  case tok::kw_static_cast:
    return new CXXStaticCastExpr(DestType.getNonReferenceType(), Ex, 
                                 DestType, OpLoc);
  }
  
  return true;
}

/// CheckConstCast - Check that a const_cast\<DestType\>(SrcExpr) is valid.
/// Refer to C++ 5.2.11 for details. const_cast is typically used in code
/// like this:
/// const char *str = "literal";
/// legacy_function(const_cast\<char*\>(str));
void
Sema::CheckConstCast(SourceLocation OpLoc, Expr *&SrcExpr, QualType DestType)
{
  QualType OrigDestType = DestType, OrigSrcType = SrcExpr->getType();

  DestType = Context.getCanonicalType(DestType);
  QualType SrcType = SrcExpr->getType();
  if (const ReferenceType *DestTypeTmp = DestType->getAsReferenceType()) {
    if (SrcExpr->isLvalue(Context) != Expr::LV_Valid) {
      // Cannot cast non-lvalue to reference type.
      Diag(OpLoc, diag::err_bad_cxx_cast_rvalue,
        "const_cast", OrigDestType.getAsString());
      return;
    }

    // C++ 5.2.11p4: An lvalue of type T1 can be [cast] to an lvalue of type T2
    //   [...] if a pointer to T1 can be [cast] to the type pointer to T2.
    DestType = Context.getPointerType(DestTypeTmp->getPointeeType());
    SrcType = Context.getPointerType(SrcType);
  } else {
    // C++ 5.2.11p1: Otherwise, the result is an rvalue and the
    //   lvalue-to-rvalue, array-to-pointer, and function-to-pointer standard
    //   conversions are performed on the expression.
    DefaultFunctionArrayConversion(SrcExpr);
    SrcType = SrcExpr->getType();
  }

  if (!DestType->isPointerType()) {
    // Cannot cast to non-pointer, non-reference type. Note that, if DestType
    // was a reference type, we converted it to a pointer above.
    // C++ 5.2.11p3: For two pointer types [...]
    Diag(OpLoc, diag::err_bad_const_cast_dest, OrigDestType.getAsString());
    return;
  }
  if (DestType->isFunctionPointerType()) {
    // Cannot cast direct function pointers.
    // C++ 5.2.11p2: [...] where T is any object type or the void type [...]
    // T is the ultimate pointee of source and target type.
    Diag(OpLoc, diag::err_bad_const_cast_dest, OrigDestType.getAsString());
    return;
  }
  SrcType = Context.getCanonicalType(SrcType);

  // Unwrap the pointers. Ignore qualifiers. Terminate early if the types are
  // completely equal.
  // FIXME: const_cast should probably not be able to convert between pointers
  // to different address spaces.
  // C++ 5.2.11p3 describes the core semantics of const_cast. All cv specifiers
  // in multi-level pointers may change, but the level count must be the same,
  // as must be the final pointee type.
  while (SrcType != DestType && UnwrapSimilarPointerTypes(SrcType, DestType)) {
    SrcType = SrcType.getUnqualifiedType();
    DestType = DestType.getUnqualifiedType();
  }

  // Doug Gregor said to disallow this until users complain.
#if 0
  // If we end up with constant arrays of equal size, unwrap those too. A cast
  // from const int [N] to int (&)[N] is invalid by my reading of the
  // standard, but g++ accepts it even with -ansi -pedantic.
  // No more than one level, though, so don't embed this in the unwrap loop
  // above.
  const ConstantArrayType *SrcTypeArr, *DestTypeArr;
  if ((SrcTypeArr = Context.getAsConstantArrayType(SrcType)) &&
     (DestTypeArr = Context.getAsConstantArrayType(DestType)))
  {
    if (SrcTypeArr->getSize() != DestTypeArr->getSize()) {
      // Different array sizes.
      Diag(OpLoc, diag::err_bad_cxx_cast_generic, "const_cast",
        OrigDestType.getAsString(), OrigSrcType.getAsString());
      return;
    }
    SrcType = SrcTypeArr->getElementType().getUnqualifiedType();
    DestType = DestTypeArr->getElementType().getUnqualifiedType();
  }
#endif

  // Since we're dealing in canonical types, the remainder must be the same.
  if (SrcType != DestType) {
    // Cast between unrelated types.
    Diag(OpLoc, diag::err_bad_cxx_cast_generic, "const_cast",
      OrigDestType.getAsString(), OrigSrcType.getAsString());
    return;
  }
}

/// CheckReinterpretCast - Check that a reinterpret_cast\<DestType\>(SrcExpr) is
/// valid.
/// Refer to C++ 5.2.10 for details. reinterpret_cast is typically used in code
/// like this:
/// char *bytes = reinterpret_cast\<char*\>(int_ptr);
void
Sema::CheckReinterpretCast(SourceLocation OpLoc, Expr *&SrcExpr,
                           QualType DestType)
{
  QualType OrigDestType = DestType, OrigSrcType = SrcExpr->getType();

  DestType = Context.getCanonicalType(DestType);
  QualType SrcType = SrcExpr->getType();
  if (const ReferenceType *DestTypeTmp = DestType->getAsReferenceType()) {
    if (SrcExpr->isLvalue(Context) != Expr::LV_Valid) {
      // Cannot cast non-lvalue to reference type.
      Diag(OpLoc, diag::err_bad_cxx_cast_rvalue,
        "reinterpret_cast", OrigDestType.getAsString());
      return;
    }

    // C++ 5.2.10p10: [...] a reference cast reinterpret_cast<T&>(x) has the
    //   same effect as the conversion *reinterpret_cast<T*>(&x) with the
    //   built-in & and * operators.
    // This code does this transformation for the checked types.
    DestType = Context.getPointerType(DestTypeTmp->getPointeeType());
    SrcType = Context.getPointerType(SrcType);
  } else {
    // C++ 5.2.10p1: [...] the lvalue-to-rvalue, array-to-pointer, and
    //   function-to-pointer standard conversions are performed on the
    //   expression v.
    DefaultFunctionArrayConversion(SrcExpr);
    SrcType = SrcExpr->getType();
  }

  // Canonicalize source for comparison.
  SrcType = Context.getCanonicalType(SrcType);

  bool destIsPtr = DestType->isPointerType();
  bool srcIsPtr = SrcType->isPointerType();
  if (!destIsPtr && !srcIsPtr) {
    // Except for std::nullptr_t->integer, which is not supported yet, and
    // lvalue->reference, which is handled above, at least one of the two
    // arguments must be a pointer.
    Diag(OpLoc, diag::err_bad_cxx_cast_generic, "reinterpret_cast",
      OrigDestType.getAsString(), OrigSrcType.getAsString());
    return;
  }

  if (SrcType == DestType) {
    // C++ 5.2.10p2 has a note that mentions that, subject to all other
    // restrictions, a cast to the same type is allowed. The intent is not
    // entirely clear here, since all other paragraphs explicitly forbid casts
    // to the same type. However, the behavior of compilers is pretty consistent
    // on this point: allow same-type conversion if the involved are pointers,
    // disallow otherwise.
    return;
  }

  // Note: Clang treats enumeration types as integral types. If this is ever
  // changed for C++, the additional check here will be redundant.
  if (DestType->isIntegralType() && !DestType->isEnumeralType()) {
    assert(srcIsPtr);
    // C++ 5.2.10p4: A pointer can be explicitly converted to any integral
    //   type large enough to hold it.
    if (Context.getTypeSize(SrcType) > Context.getTypeSize(DestType)) {
      Diag(OpLoc, diag::err_bad_reinterpret_cast_small_int,
        OrigDestType.getAsString());
    }
    return;
  }

  if (SrcType->isIntegralType() || SrcType->isEnumeralType()) {
    assert(destIsPtr);
    // C++ 5.2.10p5: A value of integral or enumeration type can be explicitly
    //   converted to a pointer.
    return;
  }

  if (!destIsPtr || !srcIsPtr) {
    // With the valid non-pointer conversions out of the way, we can be even
    // more stringent.
    Diag(OpLoc, diag::err_bad_cxx_cast_generic, "reinterpret_cast",
      OrigDestType.getAsString(), OrigSrcType.getAsString());
    return;
  }

  // C++ 5.2.10p2: The reinterpret_cast operator shall not cast away constness.
  if (CastsAwayConstness(SrcType, DestType)) {
    Diag(OpLoc, diag::err_bad_cxx_cast_const_away, "reinterpret_cast",
      OrigDestType.getAsString(), OrigSrcType.getAsString());
    return;
  }

  // Not casting away constness, so the only remaining check is for compatible
  // pointer categories.

  if (SrcType->isFunctionPointerType()) {
    if (DestType->isFunctionPointerType()) {
      // C++ 5.2.10p6: A pointer to a function can be explicitly converted to
      // a pointer to a function of a different type.
      return;
    }

    // FIXME: Handle member pointers.

    // C++0x 5.2.10p8: Converting a pointer to a function into a pointer to
    //   an object type or vice versa is conditionally-supported.
    // Compilers support it in C++03 too, though, because it's necessary for
    // casting the return value of dlsym() and GetProcAddress().
    // FIXME: Conditionally-supported behavior should be configurable in the
    // TargetInfo or similar.
    if (!getLangOptions().CPlusPlus0x) {
      Diag(OpLoc, diag::ext_reinterpret_cast_fn_obj);
    }
    return;
  }

  // FIXME: Handle member pointers.

  if (DestType->isFunctionPointerType()) {
    // See above.
    if (!getLangOptions().CPlusPlus0x) {
      Diag(OpLoc, diag::ext_reinterpret_cast_fn_obj);
    }
    return;
  }

  // C++ 5.2.10p7: A pointer to an object can be explicitly converted to
  //   a pointer to an object of different type.
  // Void pointers are not specified, but supported by every compiler out there.
  // So we finish by allowing everything that remains - it's got to be two
  // object pointers.
}

/// CastsAwayConstness - Check if the pointer conversion from SrcType
/// to DestType casts away constness as defined in C++
/// 5.2.11p8ff. This is used by the cast checkers.  Both arguments
/// must denote pointer types.
bool
Sema::CastsAwayConstness(QualType SrcType, QualType DestType)
{
 // Casting away constness is defined in C++ 5.2.11p8 with reference to
  // C++ 4.4.
  // We piggyback on Sema::IsQualificationConversion for this, since the rules
  // are non-trivial. So first we construct Tcv *...cv* as described in
  // C++ 5.2.11p8.
  SrcType  = Context.getCanonicalType(SrcType);
  DestType = Context.getCanonicalType(DestType);

  QualType UnwrappedSrcType = SrcType, UnwrappedDestType = DestType;
  llvm::SmallVector<unsigned, 8> cv1, cv2;

  // Find the qualifications.
  while (UnwrapSimilarPointerTypes(UnwrappedSrcType, UnwrappedDestType)) {
    cv1.push_back(UnwrappedSrcType.getCVRQualifiers());
    cv2.push_back(UnwrappedDestType.getCVRQualifiers());
  }
  assert(cv1.size() > 0 && "Must have at least one pointer level.");

  // Construct void pointers with those qualifiers (in reverse order of
  // unwrapping, of course).
  QualType SrcConstruct = Context.VoidTy;
  QualType DestConstruct = Context.VoidTy;
  for (llvm::SmallVector<unsigned, 8>::reverse_iterator i1 = cv1.rbegin(),
                                                        i2 = cv2.rbegin();
       i1 != cv1.rend(); ++i1, ++i2)
  {
    SrcConstruct = Context.getPointerType(SrcConstruct.getQualifiedType(*i1));
    DestConstruct = Context.getPointerType(DestConstruct.getQualifiedType(*i2));
  }

  // Test if they're compatible.
  return SrcConstruct != DestConstruct &&
    !IsQualificationConversion(SrcConstruct, DestConstruct);
}

/// CheckStaticCast - Check that a static_cast\<DestType\>(SrcExpr) is valid.
void
Sema::CheckStaticCast(SourceLocation OpLoc, Expr *&SrcExpr, QualType DestType)
{
#if 0
  // 5.2.9/1 sets the ground rule of disallowing casting away constness.
  // 5.2.9/2 permits everything allowed for direct-init, deferring to 8.5.
  //   Note: for class destination, that's overload resolution over dest's
  //   constructors. Src's conversions are only considered in overload choice.
  //   For any other destination, that's just the clause 4 standards convs.
  // 5.2.9/4 permits static_cast&lt;cv void>(anything), which is a no-op.
  // 5.2.9/5 permits explicit non-dynamic downcasts for lvalue-to-reference.
  // 5.2.9/6 permits reversing all implicit conversions except lvalue-to-rvalue,
  //   function-to-pointer, array decay and to-bool, with some further
  //   restrictions. Defers to 4.
  // 5.2.9/7 permits integer-to-enum conversion. Interesting note: if the
  //   integer does not correspond to an enum value, the result is unspecified -
  //   but it still has to be some value of the enum. I don't think any compiler
  //   complies with that.
  // 5.2.9/8 is 5.2.9/5 for pointers.
  // 5.2.9/9 messes with member pointers. TODO. No need to think about that yet.
  // 5.2.9/10 permits void* to T*.

  QualType OrigDestType = DestType, OrigSrcType = SrcExpr->getType();
  DestType = Context.getCanonicalType(DestType);
  // Tests are ordered by simplicity and a wild guess at commonness.

  if (const BuiltinType *BuiltinDest = DestType->getAsBuiltinType()) {
    // 5.2.9/4
    if (BuiltinDest->getKind() == BuiltinType::Void) {
      return;
    }

    // Primitive conversions for 5.2.9/2 and 6.
  }
#endif
}

/// ActOnCXXBoolLiteral - Parse {true,false} literals.
Action::ExprResult
Sema::ActOnCXXBoolLiteral(SourceLocation OpLoc, tok::TokenKind Kind) {
  assert((Kind == tok::kw_true || Kind == tok::kw_false) &&
         "Unknown C++ Boolean value!");
  return new CXXBoolLiteralExpr(Kind == tok::kw_true, Context.BoolTy, OpLoc);
}

/// ActOnCXXThrow - Parse throw expressions.
Action::ExprResult
Sema::ActOnCXXThrow(SourceLocation OpLoc, ExprTy *E) {
  return new CXXThrowExpr((Expr*)E, Context.VoidTy, OpLoc);
}

Action::ExprResult Sema::ActOnCXXThis(SourceLocation ThisLoc) {
  /// C++ 9.3.2: In the body of a non-static member function, the keyword this
  /// is a non-lvalue expression whose value is the address of the object for
  /// which the function is called.

  if (!isa<FunctionDecl>(CurContext)) {
    Diag(ThisLoc, diag::err_invalid_this_use);
    return ExprResult(true);
  }

  if (CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(CurContext))
    if (MD->isInstance())
      return new PredefinedExpr(ThisLoc, MD->getThisType(Context),
                                PredefinedExpr::CXXThis);

  return Diag(ThisLoc, diag::err_invalid_this_use);
}

/// ActOnCXXTypeConstructExpr - Parse construction of a specified type.
/// Can be interpreted either as function-style casting ("int(x)")
/// or class type construction ("ClassType(x,y,z)")
/// or creation of a value-initialized type ("int()").
Action::ExprResult
Sema::ActOnCXXTypeConstructExpr(SourceRange TypeRange, TypeTy *TypeRep,
                                SourceLocation LParenLoc,
                                ExprTy **ExprTys, unsigned NumExprs,
                                SourceLocation *CommaLocs,
                                SourceLocation RParenLoc) {
  assert(TypeRep && "Missing type!");
  QualType Ty = QualType::getFromOpaquePtr(TypeRep);
  Expr **Exprs = (Expr**)ExprTys;
  SourceLocation TyBeginLoc = TypeRange.getBegin();
  SourceRange FullRange = SourceRange(TyBeginLoc, RParenLoc);

  if (const RecordType *RT = Ty->getAsRecordType()) {
    // C++ 5.2.3p1:
    // If the simple-type-specifier specifies a class type, the class type shall
    // be complete.
    //
    if (!RT->getDecl()->isDefinition())
      return Diag(TyBeginLoc, diag::err_invalid_incomplete_type_use,
                  Ty.getAsString(), FullRange);

    unsigned DiagID = PP.getDiagnostics().getCustomDiagID(Diagnostic::Error,
                                    "class constructors are not supported yet");
    return Diag(TyBeginLoc, DiagID);
  }

  // C++ 5.2.3p1:
  // If the expression list is a single expression, the type conversion
  // expression is equivalent (in definedness, and if defined in meaning) to the
  // corresponding cast expression.
  //
  if (NumExprs == 1) {
    if (CheckCastTypes(TypeRange, Ty, Exprs[0]))
      return true;
    return new CXXFunctionalCastExpr(Ty.getNonReferenceType(), Ty, TyBeginLoc, 
                                     Exprs[0], RParenLoc);
  }

  // C++ 5.2.3p1:
  // If the expression list specifies more than a single value, the type shall
  // be a class with a suitably declared constructor.
  //
  if (NumExprs > 1)
    return Diag(CommaLocs[0], diag::err_builtin_func_cast_more_than_one_arg,
                FullRange);

  assert(NumExprs == 0 && "Expected 0 expressions");

  // C++ 5.2.3p2:
  // The expression T(), where T is a simple-type-specifier for a non-array
  // complete object type or the (possibly cv-qualified) void type, creates an
  // rvalue of the specified type, which is value-initialized.
  //
  if (Ty->isArrayType())
    return Diag(TyBeginLoc, diag::err_value_init_for_array_type, FullRange);
  if (Ty->isIncompleteType() && !Ty->isVoidType())
    return Diag(TyBeginLoc, diag::err_invalid_incomplete_type_use,
                Ty.getAsString(), FullRange);

  return new CXXZeroInitValueExpr(Ty, TyBeginLoc, RParenLoc);
}


/// ActOnCXXConditionDeclarationExpr - Parsed a condition declaration of a
/// C++ if/switch/while/for statement.
/// e.g: "if (int x = f()) {...}"
Action::ExprResult
Sema::ActOnCXXConditionDeclarationExpr(Scope *S, SourceLocation StartLoc,
                                       Declarator &D,
                                       SourceLocation EqualLoc,
                                       ExprTy *AssignExprVal) {
  assert(AssignExprVal && "Null assignment expression");

  // C++ 6.4p2:
  // The declarator shall not specify a function or an array.
  // The type-specifier-seq shall not contain typedef and shall not declare a
  // new class or enumeration.

  assert(D.getDeclSpec().getStorageClassSpec() != DeclSpec::SCS_typedef &&
         "Parser allowed 'typedef' as storage class of condition decl.");

  QualType Ty = GetTypeForDeclarator(D, S);
  
  if (Ty->isFunctionType()) { // The declarator shall not specify a function...
    // We exit without creating a CXXConditionDeclExpr because a FunctionDecl
    // would be created and CXXConditionDeclExpr wants a VarDecl.
    return Diag(StartLoc, diag::err_invalid_use_of_function_type,
                SourceRange(StartLoc, EqualLoc));
  } else if (Ty->isArrayType()) { // ...or an array.
    Diag(StartLoc, diag::err_invalid_use_of_array_type,
         SourceRange(StartLoc, EqualLoc));
  } else if (const RecordType *RT = Ty->getAsRecordType()) {
    RecordDecl *RD = RT->getDecl();
    // The type-specifier-seq shall not declare a new class...
    if (RD->isDefinition() && (RD->getIdentifier() == 0 || S->isDeclScope(RD)))
      Diag(RD->getLocation(), diag::err_type_defined_in_condition);
  } else if (const EnumType *ET = Ty->getAsEnumType()) {
    EnumDecl *ED = ET->getDecl();
    // ...or enumeration.
    if (ED->isDefinition() && (ED->getIdentifier() == 0 || S->isDeclScope(ED)))
      Diag(ED->getLocation(), diag::err_type_defined_in_condition);
  }

  DeclTy *Dcl = ActOnDeclarator(S, D, 0);
  if (!Dcl)
    return true;
  AddInitializerToDecl(Dcl, AssignExprVal);

  return new CXXConditionDeclExpr(StartLoc, EqualLoc,
                                       cast<VarDecl>(static_cast<Decl *>(Dcl)));
}

/// CheckCXXBooleanCondition - Returns true if a conversion to bool is invalid.
bool Sema::CheckCXXBooleanCondition(Expr *&CondExpr) {
  // C++ 6.4p4:
  // The value of a condition that is an initialized declaration in a statement
  // other than a switch statement is the value of the declared variable
  // implicitly converted to type bool. If that conversion is ill-formed, the
  // program is ill-formed.
  // The value of a condition that is an expression is the value of the
  // expression, implicitly converted to bool.
  //
  QualType Ty = CondExpr->getType(); // Save the type.
  AssignConvertType
    ConvTy = CheckSingleAssignmentConstraints(Context.BoolTy, CondExpr);
  if (ConvTy == Incompatible)
    return Diag(CondExpr->getLocStart(), diag::err_typecheck_bool_condition,
                Ty.getAsString(), CondExpr->getSourceRange());
  return false;
}

/// Helper function to determine whether this is the (deprecated) C++
/// conversion from a string literal to a pointer to non-const char or
/// non-const wchar_t (for narrow and wide string literals,
/// respectively).
bool 
Sema::IsStringLiteralToNonConstPointerConversion(Expr *From, QualType ToType) {
  // Look inside the implicit cast, if it exists.
  if (ImplicitCastExpr *Cast = dyn_cast<ImplicitCastExpr>(From))
    From = Cast->getSubExpr();

  // A string literal (2.13.4) that is not a wide string literal can
  // be converted to an rvalue of type "pointer to char"; a wide
  // string literal can be converted to an rvalue of type "pointer
  // to wchar_t" (C++ 4.2p2).
  if (StringLiteral *StrLit = dyn_cast<StringLiteral>(From))
    if (const PointerType *ToPtrType = ToType->getAsPointerType())
      if (const BuiltinType *ToPointeeType 
          = ToPtrType->getPointeeType()->getAsBuiltinType()) {
        // This conversion is considered only when there is an
        // explicit appropriate pointer target type (C++ 4.2p2).
        if (ToPtrType->getPointeeType().getCVRQualifiers() == 0 &&
            ((StrLit->isWide() && ToPointeeType->isWideCharType()) ||
             (!StrLit->isWide() &&
              (ToPointeeType->getKind() == BuiltinType::Char_U ||
               ToPointeeType->getKind() == BuiltinType::Char_S))))
          return true;
      }

  return false;
}

/// PerformImplicitConversion - Perform an implicit conversion of the
/// expression From to the type ToType. Returns true if there was an
/// error, false otherwise. The expression From is replaced with the
/// converted expression.
bool 
Sema::PerformImplicitConversion(Expr *&From, QualType ToType)
{
  ImplicitConversionSequence ICS = TryCopyInitialization(From, ToType);
  switch (ICS.ConversionKind) {
  case ImplicitConversionSequence::StandardConversion:
    if (PerformImplicitConversion(From, ToType, ICS.Standard))
      return true;
    break;

  case ImplicitConversionSequence::UserDefinedConversion:
    // FIXME: This is, of course, wrong. We'll need to actually call
    // the constructor or conversion operator, and then cope with the
    // standard conversions.
    ImpCastExprToType(From, ToType);
    break;

  case ImplicitConversionSequence::EllipsisConversion:
    assert(false && "Cannot perform an ellipsis conversion");
    break;

  case ImplicitConversionSequence::BadConversion:
    return true;
  }

  // Everything went well.
  return false;
}

/// PerformImplicitConversion - Perform an implicit conversion of the
/// expression From to the type ToType by following the standard
/// conversion sequence SCS. Returns true if there was an error, false
/// otherwise. The expression From is replaced with the converted
/// expression.
bool 
Sema::PerformImplicitConversion(Expr *&From, QualType ToType,
                                const StandardConversionSequence& SCS)
{
  // Overall FIXME: we are recomputing too many types here and doing
  // far too much extra work. What this means is that we need to keep
  // track of more information that is computed when we try the
  // implicit conversion initially, so that we don't need to recompute
  // anything here.
  QualType FromType = From->getType();

  // Perform the first implicit conversion.
  switch (SCS.First) {
  case ICK_Identity:
  case ICK_Lvalue_To_Rvalue:
    // Nothing to do.
    break;

  case ICK_Array_To_Pointer:
    FromType = Context.getArrayDecayedType(FromType);
    ImpCastExprToType(From, FromType);
    break;

  case ICK_Function_To_Pointer:
    FromType = Context.getPointerType(FromType);
    ImpCastExprToType(From, FromType);
    break;

  default:
    assert(false && "Improper first standard conversion");
    break;
  }

  // Perform the second implicit conversion
  switch (SCS.Second) {
  case ICK_Identity:
    // Nothing to do.
    break;

  case ICK_Integral_Promotion:
  case ICK_Floating_Promotion:
  case ICK_Integral_Conversion:
  case ICK_Floating_Conversion:
  case ICK_Floating_Integral:
    FromType = ToType.getUnqualifiedType();
    ImpCastExprToType(From, FromType);
    break;

  case ICK_Pointer_Conversion:
    if (CheckPointerConversion(From, ToType))
      return true;
    ImpCastExprToType(From, ToType);
    break;

  case ICK_Pointer_Member:
    // FIXME: Implement pointer-to-member conversions.
    assert(false && "Pointer-to-member conversions are unsupported");
    break;

  case ICK_Boolean_Conversion:
    FromType = Context.BoolTy;
    ImpCastExprToType(From, FromType);
    break;

  default:
    assert(false && "Improper second standard conversion");
    break;
  }

  switch (SCS.Third) {
  case ICK_Identity:
    // Nothing to do.
    break;

  case ICK_Qualification:
    ImpCastExprToType(From, ToType);
    break;

  default:
    assert(false && "Improper second standard conversion");
    break;
  }

  return false;
}

