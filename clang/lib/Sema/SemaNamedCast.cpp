//===--- SemaNamedCast.cpp - Semantic Analysis for Named Casts ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for C++ named casts.
//
//===----------------------------------------------------------------------===//

#include "Sema.h"
#include "SemaInherit.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/SmallVector.h"
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
  SourceRange OpRange(OpLoc, RParenLoc);
  SourceRange DestRange(LAngleBracketLoc, RAngleBracketLoc);

  switch (Kind) {
  default: assert(0 && "Unknown C++ cast!");

  case tok::kw_const_cast:
    CheckConstCast(Ex, DestType, OpRange, DestRange);
    return new CXXConstCastExpr(DestType.getNonReferenceType(), Ex, 
                                DestType, OpLoc);

  case tok::kw_dynamic_cast:
    CheckDynamicCast(Ex, DestType, OpRange, DestRange);
    return new CXXDynamicCastExpr(DestType.getNonReferenceType(), Ex, 
                                  DestType, OpLoc);

  case tok::kw_reinterpret_cast:
    CheckReinterpretCast(Ex, DestType, OpRange, DestRange);
    return new CXXReinterpretCastExpr(DestType.getNonReferenceType(), Ex, 
                                      DestType, OpLoc);

  case tok::kw_static_cast:
    CheckStaticCast(Ex, DestType, OpRange);
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
Sema::CheckConstCast(Expr *&SrcExpr, QualType DestType,
                     const SourceRange &OpRange, const SourceRange &DestRange)
{
  QualType OrigDestType = DestType, OrigSrcType = SrcExpr->getType();

  DestType = Context.getCanonicalType(DestType);
  QualType SrcType = SrcExpr->getType();
  if (const ReferenceType *DestTypeTmp = DestType->getAsReferenceType()) {
    if (SrcExpr->isLvalue(Context) != Expr::LV_Valid) {
      // Cannot cast non-lvalue to reference type.
      Diag(OpRange.getBegin(), diag::err_bad_cxx_cast_rvalue,
        "const_cast", OrigDestType.getAsString(), SrcExpr->getSourceRange());
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
    Diag(OpRange.getBegin(), diag::err_bad_const_cast_dest,
      OrigDestType.getAsString(), DestRange);
    return;
  }
  if (DestType->isFunctionPointerType()) {
    // Cannot cast direct function pointers.
    // C++ 5.2.11p2: [...] where T is any object type or the void type [...]
    // T is the ultimate pointee of source and target type.
    Diag(OpRange.getBegin(), diag::err_bad_const_cast_dest,
      OrigDestType.getAsString(), DestRange);
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
      Diag(OpRange.getBegin(), diag::err_bad_cxx_cast_generic, "const_cast",
        OrigDestType.getAsString(), OrigSrcType.getAsString(), OpRange);
      return;
    }
    SrcType = SrcTypeArr->getElementType().getUnqualifiedType();
    DestType = DestTypeArr->getElementType().getUnqualifiedType();
  }
#endif

  // Since we're dealing in canonical types, the remainder must be the same.
  if (SrcType != DestType) {
    // Cast between unrelated types.
    Diag(OpRange.getBegin(), diag::err_bad_cxx_cast_generic, "const_cast",
      OrigDestType.getAsString(), OrigSrcType.getAsString(), OpRange);
    return;
  }
}

/// CheckReinterpretCast - Check that a reinterpret_cast\<DestType\>(SrcExpr) is
/// valid.
/// Refer to C++ 5.2.10 for details. reinterpret_cast is typically used in code
/// like this:
/// char *bytes = reinterpret_cast\<char*\>(int_ptr);
void
Sema::CheckReinterpretCast(Expr *&SrcExpr, QualType DestType,
                           const SourceRange &OpRange,
                           const SourceRange &DestRange)
{
  QualType OrigDestType = DestType, OrigSrcType = SrcExpr->getType();

  DestType = Context.getCanonicalType(DestType);
  QualType SrcType = SrcExpr->getType();
  if (const ReferenceType *DestTypeTmp = DestType->getAsReferenceType()) {
    if (SrcExpr->isLvalue(Context) != Expr::LV_Valid) {
      // Cannot cast non-lvalue to reference type.
      Diag(OpRange.getBegin(), diag::err_bad_cxx_cast_rvalue,
        "reinterpret_cast", OrigDestType.getAsString(),
        SrcExpr->getSourceRange());
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
    Diag(OpRange.getBegin(), diag::err_bad_cxx_cast_generic, "reinterpret_cast",
      OrigDestType.getAsString(), OrigSrcType.getAsString(), OpRange);
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
    assert(srcIsPtr && "One type must be a pointer");
    // C++ 5.2.10p4: A pointer can be explicitly converted to any integral
    //   type large enough to hold it.
    if (Context.getTypeSize(SrcType) > Context.getTypeSize(DestType)) {
      Diag(OpRange.getBegin(), diag::err_bad_reinterpret_cast_small_int,
        OrigDestType.getAsString(), DestRange);
    }
    return;
  }

  if (SrcType->isIntegralType() || SrcType->isEnumeralType()) {
    assert(destIsPtr && "One type must be a pointer");
    // C++ 5.2.10p5: A value of integral or enumeration type can be explicitly
    //   converted to a pointer.
    return;
  }

  if (!destIsPtr || !srcIsPtr) {
    // With the valid non-pointer conversions out of the way, we can be even
    // more stringent.
    Diag(OpRange.getBegin(), diag::err_bad_cxx_cast_generic, "reinterpret_cast",
      OrigDestType.getAsString(), OrigSrcType.getAsString(), OpRange);
    return;
  }

  // C++ 5.2.10p2: The reinterpret_cast operator shall not cast away constness.
  if (CastsAwayConstness(SrcType, DestType)) {
    Diag(OpRange.getBegin(), diag::err_bad_cxx_cast_const_away,
      "reinterpret_cast", OrigDestType.getAsString(), OrigSrcType.getAsString(),
      OpRange);
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
      Diag(OpRange.getBegin(), diag::ext_reinterpret_cast_fn_obj, OpRange);
    }
    return;
  }

  // FIXME: Handle member pointers.

  if (DestType->isFunctionPointerType()) {
    // See above.
    if (!getLangOptions().CPlusPlus0x) {
      Diag(OpRange.getBegin(), diag::ext_reinterpret_cast_fn_obj, OpRange);
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
Sema::CheckStaticCast(Expr *&SrcExpr, QualType DestType,
                      const SourceRange &OpRange)
{
  // The order the tests is not entirely arbitrary. There is one conversion
  // that can be handled in two different ways. Given:
  // struct A {};
  // struct B : public A {
  //   B(); B(const A&);
  // };
  // const A &a = B();
  // the cast static_cast<const B&>(a) could be seen as either a static
  // reference downcast, or an explicit invocation of the user-defined
  // conversion using B's conversion constructor.
  // DR 427 specifies that the downcast is to be applied here.

  // C++ 5.2.9p4: Any expression can be explicitly converted to type "cv void".
  if (DestType->isVoidType()) {
    return;
  }

  // C++ 5.2.9p5, reference downcast.
  // See the function for details.
  // DR 427 specifies that this is to be applied before paragraph 2.
  if (IsStaticReferenceDowncast(SrcExpr, DestType)) {
    return;
  }

  // C++ 5.2.9p2: An expression e can be explicitly converted to a type T
  //   [...] if the declaration "T t(e);" is well-formed, [...].
  ImplicitConversionSequence ICS = TryDirectInitialization(SrcExpr, DestType);
  if (ICS.ConversionKind != ImplicitConversionSequence::BadConversion) {
    assert(ICS.ConversionKind != ImplicitConversionSequence::EllipsisConversion
      && "Direct initialization cannot result in ellipsis conversion");
    // UserDefinedConversionSequence has a StandardConversionSequence as a
    // prefix. Accessing Standard is therefore safe.
    // FIXME: Of course, this is definitely not enough.
    if(ICS.Standard.First != ICK_Identity) {
      DefaultFunctionArrayConversion(SrcExpr);
    }
    // FIXME: Test the details, such as accessible base.
    return;
  }

  // C++ 5.2.9p6: May apply the reverse of any standard conversion, except
  // lvalue-to-rvalue, array-to-pointer, function-to-pointer, and boolean
  // conversions, subject to further restrictions.
  // Also, C++ 5.2.9p1 forbids casting away constness, which makes reversal
  // of qualification conversions impossible.

  // The lvalue-to-rvalue, array-to-pointer and function-to-pointer conversions
  // are applied to the expression.
  QualType OrigSrcType = SrcExpr->getType();
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
        if (DestPointee->isObjectType()) {
          // This is definitely the intended conversion, but it might fail due
          // to a const violation.
          if (!DestPointee.isAtLeastAsQualifiedAs(SrcPointee)) {
            Diag(OpRange.getBegin(), diag::err_bad_cxx_cast_const_away,
              "static_cast", DestType.getAsString(),
              OrigSrcType.getAsString(), OpRange);
          }
          return;
        }
      }
    }
  }

  // We tried everything. Everything! Nothing works! :-(
  // FIXME: Error reporting could be a lot better. Should store the reason
  // why every substep failed and, at the end, select the most specific and
  // report that.
  Diag(OpRange.getBegin(), diag::err_bad_cxx_cast_generic, "static_cast",
    DestType.getAsString(), OrigSrcType.getAsString(), OpRange);
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

  const ReferenceType *DestReference = DestType->getAsReferenceType();
  if (!DestReference) {
    return false;
  }
  QualType DestPointee = DestReference->getPointeeType();

  return IsStaticDowncast(SrcExpr->getType(), DestPointee);
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

  const PointerType *SrcPointer = SrcType->getAsPointerType();
  if (!SrcPointer) {
    return false;
  }

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
  // Downcast can only happen in class hierarchies, so we need classes.
  if (!DestType->isRecordType() || !SrcType->isRecordType()) {
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

  // FIXME: Not enough support for the rest yet, actually.
  ImplicitConversionSequence ICS;
  ICS.ConversionKind = ImplicitConversionSequence::BadConversion;
  return ICS;
}

/// CheckDynamicCast - Check that a dynamic_cast\<DestType\>(SrcExpr) is valid.
/// Refer to C++ 5.2.7 for details. Dynamic casts are used mostly for runtime-
/// checked downcasts in class hierarchies.
void
Sema::CheckDynamicCast(Expr *&SrcExpr, QualType DestType,
                       const SourceRange &OpRange,
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
    Diag(OpRange.getBegin(), diag::err_bad_dynamic_cast_not_ref_or_ptr,
      OrigDestType.getAsString(), DestRange);
    return;
  }

  const RecordType *DestRecord = DestPointee->getAsRecordType();
  if (DestPointee->isVoidType()) {
    assert(DestPointer && "Reference to void is not possible");
  } else if (DestRecord) {
    if (!DestRecord->getDecl()->isDefinition()) {
      Diag(OpRange.getBegin(), diag::err_bad_dynamic_cast_incomplete,
        DestPointee.getUnqualifiedType().getAsString(), DestRange);
      return;
    }
  } else {
    Diag(OpRange.getBegin(), diag::err_bad_dynamic_cast_not_class,
      DestPointee.getUnqualifiedType().getAsString(), DestRange);
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
      Diag(OpRange.getBegin(), diag::err_bad_dynamic_cast_not_ptr,
        OrigSrcType.getAsString(), SrcExpr->getSourceRange());
      return;
    }
  } else {
    if (SrcExpr->isLvalue(Context) != Expr::LV_Valid) {
      Diag(OpRange.getBegin(), diag::err_bad_cxx_cast_rvalue, "dynamic_cast",
        OrigDestType.getAsString(), OpRange);
    }
    SrcPointee = SrcType;
  }

  const RecordType *SrcRecord = SrcPointee->getAsRecordType();
  if (SrcRecord) {
    if (!SrcRecord->getDecl()->isDefinition()) {
      Diag(OpRange.getBegin(), diag::err_bad_dynamic_cast_incomplete,
        SrcPointee.getUnqualifiedType().getAsString(),
        SrcExpr->getSourceRange());
      return;
    }
  } else {
    Diag(OpRange.getBegin(), diag::err_bad_dynamic_cast_not_class,
      SrcPointee.getUnqualifiedType().getAsString(),
      SrcExpr->getSourceRange());
    return;
  }

  assert((DestPointer || DestReference) &&
    "Bad destination non-ptr/ref slipped through.");
  assert((DestRecord || DestPointee->isVoidType()) &&
    "Bad destination pointee slipped through.");
  assert(SrcRecord && "Bad source pointee slipped through.");

  // C++ 5.2.7p1: The dynamic_cast operator shall not cast away constness.
  if (!DestPointee.isAtLeastAsQualifiedAs(SrcPointee)) {
    Diag(OpRange.getBegin(), diag::err_bad_cxx_cast_const_away, "dynamic_cast",
      OrigDestType.getAsString(), OrigSrcType.getAsString(), OpRange);
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
    CheckDerivedToBaseConversion(SrcPointee, DestPointee, OpRange.getBegin(),
      OpRange);
    // Diagnostic already emitted on error.
    return;
  }

  // C++ 5.2.7p6: Otherwise, v shall be [polymorphic].
  // FIXME: Information not yet available.

  // Done. Everything else is run-time checks.
}
