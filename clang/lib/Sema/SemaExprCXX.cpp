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
#include "SemaInherit.h"
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
    CheckDynamicCast(OpLoc, Ex, DestType,
      SourceRange(LAngleBracketLoc, RAngleBracketLoc));
    return new CXXDynamicCastExpr(DestType.getNonReferenceType(), Ex, 
                                  DestType, OpLoc);

  case tok::kw_reinterpret_cast:
    CheckReinterpretCast(OpLoc, Ex, DestType);
    return new CXXReinterpretCastExpr(DestType.getNonReferenceType(), Ex, 
                                      DestType, OpLoc);

  case tok::kw_static_cast:
    CheckStaticCast(OpLoc, Ex, DestType);
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
/// Refer to C++ 5.2.9 for details. Static casts are mostly used for making
/// implicit conversions explicit and getting rid of data loss warnings.
void
Sema::CheckStaticCast(SourceLocation OpLoc, Expr *&SrcExpr, QualType DestType)
{
  QualType OrigDestType = DestType, OrigSrcType = SrcExpr->getType();

  // Conversions are tried roughly in the order the standard specifies them.
  // This is necessary because there are some conversions that can be
  // interpreted in more than one way, and the order disambiguates.
  // DR 427 specifies that paragraph 5 is to be applied before paragraph 2.

  // This option is unambiguous and simple, so put it here.
  // C++ 5.2.9p4: Any expression can be explicitly converted to type "cv void".
  if (DestType->isVoidType()) {
    return;
  }

  DestType = Context.getCanonicalType(DestType);

  // C++ 5.2.9p5, reference downcast.
  // See the function for details.
  if (IsStaticReferenceDowncast(SrcExpr, DestType)) {
    return;
  }

  // C++ 5.2.9p2: An expression e can be explicitly converted to a type T
  //   [...] if the declaration "T t(e);" is well-formed, [...].
  ImplicitConversionSequence ICS = TryDirectInitialization(SrcExpr, DestType);
  if (ICS.ConversionKind != ImplicitConversionSequence::BadConversion) {
    if (ICS.ConversionKind == ImplicitConversionSequence::StandardConversion &&
        ICS.Standard.First != ICK_Identity)
    {
      DefaultFunctionArrayConversion(SrcExpr);
    }
    return;
  }
  // FIXME: Missing the validation of the conversion, e.g. for an accessible
  // base.

  // C++ 5.2.9p6: May apply the reverse of any standard conversion, except
  // lvalue-to-rvalue, array-to-pointer, function-to-pointer, and boolean
  // conversions, subject to further restrictions.
  // Also, C++ 5.2.9p1 forbids casting away constness, which makes reversal
  // of qualification conversions impossible.

  // The lvalue-to-rvalue, array-to-pointer and function-to-pointer conversions
  // are applied to the expression.
  DefaultFunctionArrayConversion(SrcExpr);

  QualType SrcType = Context.getCanonicalType(SrcExpr->getType());

  // Reverse integral promotion/conversion. All such conversions are themselves
  // again integral promotions or conversions and are thus already handled by
  // p2 (TryDirectInitialization above).
  // (Note: any data loss warnings should be suppressed.)
  // The exception is the reverse of enum->integer, i.e. integer->enum (and
  // enum->enum). See also C++ 5.2.9p7.
  // The same goes for reverse floating point promotion/conversion and
  // floating-integral conversions. Again, only floating->enum is relevant.
  if (DestType->isEnumeralType()) {
    if (SrcType->isComplexType() || SrcType->isVectorType()) {
      // Fall through - these cannot be converted.
    } else if (SrcType->isArithmeticType() || SrcType->isEnumeralType()) {
      return;
    }
  }

  // Reverse pointer upcast. C++ 4.10p3 specifies pointer upcast.
  // C++ 5.2.9p8 additionally disallows a cast path through virtual inheritance.
  if (IsStaticPointerDowncast(SrcType, DestType)) {
    return;
  }

  // Reverse member pointer conversion. C++ 5.11 specifies member pointer
  // conversion. C++ 5.2.9p9 has additional information.
  // DR54's access restrictions apply here also.
  // FIXME: Don't have member pointers yet.

  // Reverse pointer conversion to void*. C++ 4.10.p2 specifies conversion to
  // void*. C++ 5.2.9p10 specifies additional restrictions, which really is
  // just the usual constness stuff.
  if (const PointerType *SrcPointer = SrcType->getAsPointerType()) {
    QualType SrcPointee = SrcPointer->getPointeeType();
    if (SrcPointee->isVoidType()) {
      if (const PointerType *DestPointer = DestType->getAsPointerType()) {
        QualType DestPointee = DestPointer->getPointeeType();
        if (DestPointee->isObjectType() &&
            DestPointee.isAtLeastAsQualifiedAs(SrcPointee))
        {
          return;
        }
      }
    }
  }

  // We tried everything. Everything! Nothing works! :-(
  // FIXME: Error reporting could be a lot better. Should store the reason
  // why every substep failed and, at the end, select the most specific and
  // report that.
  Diag(OpLoc, diag::err_bad_cxx_cast_generic, "static_cast",
    OrigDestType.getAsString(), OrigSrcType.getAsString());
}

/// Tests whether a conversion according to C++ 5.2.9p5 is valid.
bool
Sema::IsStaticReferenceDowncast(Expr *SrcExpr, QualType DestType)
{
  // C++ 5.2.9p5: An lvalue of type "cv1 B", where B is a class type, can be
  //   cast to type "reference to cv2 D", where D is a class derived from B,
  //   if a valid standard conversion from "pointer to D" to "pointer to B"
  //   exists, cv2 >= cv1, and B is not a virtual base class of D.
  // In addition, DR54 clarifies that the base must be accessible in the
  // current context. Although the wording of DR54 only applies to the pointer
  // variant of this rule, the intent is clearly for it to apply to the this
  // conversion as well.

  if (SrcExpr->isLvalue(Context) != Expr::LV_Valid) {
    return false;
  }

  DestType = Context.getCanonicalType(DestType);
  const ReferenceType *DestReference = DestType->getAsReferenceType();
  if (!DestReference) {
    return false;
  }
  QualType DestPointee = DestReference->getPointeeType();

  QualType SrcType = Context.getCanonicalType(SrcExpr->getType());

  return IsStaticDowncast(SrcType, DestPointee);
}

/// Tests whether a conversion according to C++ 5.2.9p8 is valid.
bool
Sema::IsStaticPointerDowncast(QualType SrcType, QualType DestType)
{
  // C++ 5.2.9p8: An rvalue of type "pointer to cv1 B", where B is a class
  //   type, can be converted to an rvalue of type "pointer to cv2 D", where D
  //   is a class derived from B, if a valid standard conversion from "pointer
  //   to D" to "pointer to B" exists, cv2 >= cv1, and B is not a virtual base
  //   class of D.
  // In addition, DR54 clarifies that the base must be accessible in the
  // current context.

  SrcType = Context.getCanonicalType(SrcType);
  const PointerType *SrcPointer = SrcType->getAsPointerType();
  if (!SrcPointer) {
    return false;
  }

  DestType = Context.getCanonicalType(DestType);
  const PointerType *DestPointer = DestType->getAsPointerType();
  if (!DestPointer) {
    return false;
  }

  return IsStaticDowncast(SrcPointer->getPointeeType(),
                          DestPointer->getPointeeType());
}

/// IsStaticDowncast - Common functionality of IsStaticReferenceDowncast and
/// IsStaticPointerDowncast. Tests whether a static downcast from SrcType to
/// DestType, both of which must be canonical, is possible and allowed.
bool
Sema::IsStaticDowncast(QualType SrcType, QualType DestType)
{
  assert(SrcType->isCanonical());
  assert(DestType->isCanonical());

  if (!DestType->isRecordType()) {
    return false;
  }

  if (!SrcType->isRecordType()) {
    return false;
  }

  // Comparing cv is cheaper, so do it first.
  if (!DestType.isAtLeastAsQualifiedAs(SrcType)) {
    return false;
  }

  BasePaths Paths(/*FindAmbiguities=*/true, /*RecordPaths=*/false,
                  /*DetectVirtual=*/true);
  if (!IsDerivedFrom(DestType, SrcType, Paths)) {
    return false;
  }

  if (Paths.isAmbiguous(SrcType.getUnqualifiedType())) {
    return false;
  }

  if (Paths.getDetectedVirtual() != 0) {
    return false;
  }

  // FIXME: Test accessibility.

  return true;
}

/// TryDirectInitialization - Attempt to direct-initialize a value of the
/// given type (DestType) from the given expression (SrcExpr), as one would
/// do when creating an object with new with parameters. This function returns
/// an implicit conversion sequence that can be used to perform the
/// initialization.
/// This routine is very similar to TryCopyInitialization; the differences
/// between the two (C++ 8.5p12 and C++ 8.5p14) are:
/// 1) In direct-initialization, all constructors of the target type are
///    considered, including those marked as explicit.
/// 2) In direct-initialization, overload resolution is performed over the
///    constructors of the target type. In copy-initialization, overload
///    resolution is performed over all conversion functions that result in
///    the target type. This can lead to different functions used.
ImplicitConversionSequence
Sema::TryDirectInitialization(Expr *SrcExpr, QualType DestType)
{
  if (!DestType->isRecordType()) {
    // For non-class types, copy and direct initialization are identical.
    // C++ 8.5p11
    // FIXME: Those parts should be in a common function, actually.
    return TryCopyInitialization(SrcExpr, DestType);
  }

  // Not enough support for the rest yet, actually.
  ImplicitConversionSequence ICS;
  ICS.ConversionKind = ImplicitConversionSequence::BadConversion;
  return ICS;
}

/// CheckDynamicCast - Check that a dynamic_cast\<DestType\>(SrcExpr) is valid.
/// Refer to C++ 5.2.7 for details. Dynamic casts are used mostly for runtime-
/// checked downcasts in class hierarchies.
void
Sema::CheckDynamicCast(SourceLocation OpLoc, Expr *&SrcExpr, QualType DestType,
  const SourceRange &DestRange)
{
  QualType OrigDestType = DestType, OrigSrcType = SrcExpr->getType();
  DestType = Context.getCanonicalType(DestType);

  // C++ 5.2.7p1: T shall be a pointer or reference to a complete class type,
  //   or "pointer to cv void".

  QualType DestPointee;
  const PointerType *DestPointer = DestType->getAsPointerType();
  const ReferenceType *DestReference = DestType->getAsReferenceType();
  if (DestPointer) {
    DestPointee = DestPointer->getPointeeType();
  } else if (DestReference) {
    DestPointee = DestReference->getPointeeType();
  } else {
    Diag(OpLoc, diag::err_bad_dynamic_cast_operand,
      OrigDestType.getAsString(), "not a reference or pointer", DestRange);
    return;
  }

  const RecordType *DestRecord = DestPointee->getAsRecordType();
  if (DestPointee->isVoidType()) {
    assert(DestPointer && "Reference to void is not possible");
  } else if (DestRecord) {
    if (!DestRecord->getDecl()->isDefinition()) {
      Diag(OpLoc, diag::err_bad_dynamic_cast_operand,
        DestPointee.getUnqualifiedType().getAsString(),
        "incomplete", DestRange);
      return;
    }
  } else {
    Diag(OpLoc, diag::err_bad_dynamic_cast_operand,
      DestPointee.getUnqualifiedType().getAsString(),
      "not a class", DestRange);
    return;
  }

  // C++ 5.2.7p2: If T is a pointer type, v shall be an rvalue of a pointer to
  //   complete class type, [...]. If T is a reference type, v shall be an
  //   lvalue of a complete class type, [...].

  QualType SrcType = Context.getCanonicalType(OrigSrcType);
  QualType SrcPointee;
  if (DestPointer) {
    if (const PointerType *SrcPointer = SrcType->getAsPointerType()) {
      SrcPointee = SrcPointer->getPointeeType();
    } else {
      Diag(OpLoc, diag::err_bad_dynamic_cast_operand,
        OrigSrcType.getAsString(), "not a pointer", SrcExpr->getSourceRange());
      return;
    }
  } else {
    if (SrcExpr->isLvalue(Context) != Expr::LV_Valid) {
      Diag(OpLoc, diag::err_bad_dynamic_cast_operand,
        OrigDestType.getAsString(), "not an lvalue", SrcExpr->getSourceRange());
    }
    SrcPointee = SrcType;
  }

  const RecordType *SrcRecord = SrcPointee->getAsRecordType();
  if (SrcRecord) {
    if (!SrcRecord->getDecl()->isDefinition()) {
      Diag(OpLoc, diag::err_bad_dynamic_cast_operand,
        SrcPointee.getUnqualifiedType().getAsString(), "incomplete",
        SrcExpr->getSourceRange());
      return;
    }
  } else {
    Diag(OpLoc, diag::err_bad_dynamic_cast_operand,
      SrcPointee.getUnqualifiedType().getAsString(), "not a class",
      SrcExpr->getSourceRange());
    return;
  }

  // Assumptions to this point.
  assert(DestPointer || DestReference);
  assert(DestRecord || DestPointee->isVoidType());
  assert(SrcRecord);

  // C++ 5.2.7p1: The dynamic_cast operator shall not cast away constness.
  if (!DestPointee.isAtLeastAsQualifiedAs(SrcPointee)) {
    Diag(OpLoc, diag::err_bad_cxx_cast_const_away, "dynamic_cast",
      OrigDestType.getAsString(), OrigSrcType.getAsString());
    return;
  }

  // C++ 5.2.7p3: If the type of v is the same as the required result type,
  //   [except for cv].
  if (DestRecord == SrcRecord) {
    return;
  }

  // C++ 5.2.7p5
  // Upcasts are resolved statically.
  if (DestRecord && IsDerivedFrom(SrcPointee, DestPointee)) {
    CheckDerivedToBaseConversion(SrcPointee, DestPointee, OpLoc, SourceRange());
    // Diagnostic already emitted on error.
    return;
  }

  // C++ 5.2.7p6: Otherwise, v shall be [polymorphic].
  // FIXME: Information not yet available.

  // Done. Everything else is run-time checks.
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
  ImplicitConversionSequence ICS = TryImplicitConversion(From, ToType);
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
    return false;

  case ImplicitConversionSequence::EllipsisConversion:
    assert(false && "Cannot perform an ellipsis conversion");
    return false;

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

