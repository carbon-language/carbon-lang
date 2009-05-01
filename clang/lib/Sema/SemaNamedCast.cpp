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
#include "llvm/ADT/SmallVector.h"
#include <set>
using namespace clang;

enum TryStaticCastResult {
  TSC_NotApplicable, ///< The cast method is not applicable.
  TSC_Success,       ///< The cast method is appropriate and successful.
  TSC_Failed         ///< The cast method is appropriate, but failed. A
                     ///< diagnostic has been emitted.
};

static void CheckConstCast(Sema &Self, Expr *&SrcExpr, QualType DestType,
                           const SourceRange &OpRange,
                           const SourceRange &DestRange);
static void CheckReinterpretCast(Sema &Self, Expr *&SrcExpr, QualType DestType,
                                 const SourceRange &OpRange,
                                 const SourceRange &DestRange);
static void CheckStaticCast(Sema &Self, Expr *&SrcExpr, QualType DestType,
                            const SourceRange &OpRange);
static void CheckDynamicCast(Sema &Self, Expr *&SrcExpr, QualType DestType,
                             const SourceRange &OpRange,
                             const SourceRange &DestRange);

static bool CastsAwayConstness(Sema &Self, QualType SrcType, QualType DestType);
static TryStaticCastResult TryLValueToRValueCast(
  Sema &Self, Expr *SrcExpr, QualType DestType, const SourceRange &OpRange);
static TryStaticCastResult TryStaticReferenceDowncast(
  Sema &Self, Expr *SrcExpr, QualType DestType, const SourceRange &OpRange);
static TryStaticCastResult TryStaticPointerDowncast(
  Sema &Self, QualType SrcType, QualType DestType, const SourceRange &OpRange);
static TryStaticCastResult TryStaticMemberPointerUpcast(
  Sema &Self, QualType SrcType, QualType DestType, const SourceRange &OpRange);
static TryStaticCastResult TryStaticDowncast(Sema &Self, QualType SrcType,
                                             QualType DestType,
                                             const SourceRange &OpRange,
                                             QualType OrigSrcType,
                                             QualType OrigDestType);
static TryStaticCastResult TryStaticImplicitCast(Sema &Self, Expr *SrcExpr,
                                                 QualType DestType,
                                                 const SourceRange &OpRange);

/// ActOnCXXNamedCast - Parse {dynamic,static,reinterpret,const}_cast's.
Action::OwningExprResult
Sema::ActOnCXXNamedCast(SourceLocation OpLoc, tok::TokenKind Kind,
                        SourceLocation LAngleBracketLoc, TypeTy *Ty,
                        SourceLocation RAngleBracketLoc,
                        SourceLocation LParenLoc, ExprArg E,
                        SourceLocation RParenLoc) {
  Expr *Ex = E.takeAs<Expr>();
  QualType DestType = QualType::getFromOpaquePtr(Ty);
  SourceRange OpRange(OpLoc, RParenLoc);
  SourceRange DestRange(LAngleBracketLoc, RAngleBracketLoc);

  // If the type is dependent, we won't do the semantic analysis now.
  // FIXME: should we check this in a more fine-grained manner?
  bool TypeDependent = DestType->isDependentType() || Ex->isTypeDependent();

  switch (Kind) {
  default: assert(0 && "Unknown C++ cast!");

  case tok::kw_const_cast:
    if (!TypeDependent)
      CheckConstCast(*this, Ex, DestType, OpRange, DestRange);
    return Owned(new (Context) CXXConstCastExpr(DestType.getNonReferenceType(),
                                                Ex, DestType, OpLoc));

  case tok::kw_dynamic_cast:
    if (!TypeDependent)
      CheckDynamicCast(*this, Ex, DestType, OpRange, DestRange);
    return Owned(new (Context)CXXDynamicCastExpr(DestType.getNonReferenceType(),
                                                 Ex, DestType, OpLoc));

  case tok::kw_reinterpret_cast:
    if (!TypeDependent)
      CheckReinterpretCast(*this, Ex, DestType, OpRange, DestRange);
    return Owned(new (Context) CXXReinterpretCastExpr(
                                  DestType.getNonReferenceType(),
                                  Ex, DestType, OpLoc));

  case tok::kw_static_cast:
    if (!TypeDependent)
      CheckStaticCast(*this, Ex, DestType, OpRange);
    return Owned(new (Context) CXXStaticCastExpr(DestType.getNonReferenceType(),
                                                 Ex, DestType, OpLoc));
  }

  return ExprError();
}

/// CheckConstCast - Check that a const_cast\<DestType\>(SrcExpr) is valid.
/// Refer to C++ 5.2.11 for details. const_cast is typically used in code
/// like this:
/// const char *str = "literal";
/// legacy_function(const_cast\<char*\>(str));
void
CheckConstCast(Sema &Self, Expr *&SrcExpr, QualType DestType,
               const SourceRange &OpRange, const SourceRange &DestRange)
{
  QualType OrigDestType = DestType, OrigSrcType = SrcExpr->getType();

  DestType = Self.Context.getCanonicalType(DestType);
  QualType SrcType = SrcExpr->getType();
  if (const LValueReferenceType *DestTypeTmp =
        DestType->getAsLValueReferenceType()) {
    if (SrcExpr->isLvalue(Self.Context) != Expr::LV_Valid) {
      // Cannot cast non-lvalue to lvalue reference type.
      Self.Diag(OpRange.getBegin(), diag::err_bad_cxx_cast_rvalue)
        << "const_cast" << OrigDestType << SrcExpr->getSourceRange();
      return;
    }

    // C++ 5.2.11p4: An lvalue of type T1 can be [cast] to an lvalue of type T2
    //   [...] if a pointer to T1 can be [cast] to the type pointer to T2.
    DestType = Self.Context.getPointerType(DestTypeTmp->getPointeeType());
    SrcType = Self.Context.getPointerType(SrcType);
  } else {
    // C++ 5.2.11p1: Otherwise, the result is an rvalue and the
    //   lvalue-to-rvalue, array-to-pointer, and function-to-pointer standard
    //   conversions are performed on the expression.
    Self.DefaultFunctionArrayConversion(SrcExpr);
    SrcType = SrcExpr->getType();
  }

  // C++ 5.2.11p5: For a const_cast involving pointers to data members [...]
  //   the rules for const_cast are the same as those used for pointers.

  if (!DestType->isPointerType() && !DestType->isMemberPointerType()) {
    // Cannot cast to non-pointer, non-reference type. Note that, if DestType
    // was a reference type, we converted it to a pointer above.
    // The status of rvalue references isn't entirely clear, but it looks like
    // conversion to them is simply invalid.
    // C++ 5.2.11p3: For two pointer types [...]
    Self.Diag(OpRange.getBegin(), diag::err_bad_const_cast_dest)
      << OrigDestType << DestRange;
    return;
  }
  if (DestType->isFunctionPointerType() ||
      DestType->isMemberFunctionPointerType()) {
    // Cannot cast direct function pointers.
    // C++ 5.2.11p2: [...] where T is any object type or the void type [...]
    // T is the ultimate pointee of source and target type.
    Self.Diag(OpRange.getBegin(), diag::err_bad_const_cast_dest)
      << OrigDestType << DestRange;
    return;
  }
  SrcType = Self.Context.getCanonicalType(SrcType);

  // Unwrap the pointers. Ignore qualifiers. Terminate early if the types are
  // completely equal.
  // FIXME: const_cast should probably not be able to convert between pointers
  // to different address spaces.
  // C++ 5.2.11p3 describes the core semantics of const_cast. All cv specifiers
  // in multi-level pointers may change, but the level count must be the same,
  // as must be the final pointee type.
  while (SrcType != DestType &&
         Self.UnwrapSimilarPointerTypes(SrcType, DestType)) {
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
  if ((SrcTypeArr = Self.Context.getAsConstantArrayType(SrcType)) &&
     (DestTypeArr = Self.Context.getAsConstantArrayType(DestType)))
  {
    if (SrcTypeArr->getSize() != DestTypeArr->getSize()) {
      // Different array sizes.
      Self.Diag(OpRange.getBegin(), diag::err_bad_cxx_cast_generic)
        << "const_cast" << OrigDestType << OrigSrcType << OpRange;
      return;
    }
    SrcType = SrcTypeArr->getElementType().getUnqualifiedType();
    DestType = DestTypeArr->getElementType().getUnqualifiedType();
  }
#endif

  // Since we're dealing in canonical types, the remainder must be the same.
  if (SrcType != DestType) {
    // Cast between unrelated types.
    Self.Diag(OpRange.getBegin(), diag::err_bad_cxx_cast_generic)
      << "const_cast" << OrigDestType << OrigSrcType << OpRange;
    return;
  }
}

/// CheckReinterpretCast - Check that a reinterpret_cast\<DestType\>(SrcExpr) is
/// valid.
/// Refer to C++ 5.2.10 for details. reinterpret_cast is typically used in code
/// like this:
/// char *bytes = reinterpret_cast\<char*\>(int_ptr);
void
CheckReinterpretCast(Sema &Self, Expr *&SrcExpr, QualType DestType,
                     const SourceRange &OpRange, const SourceRange &DestRange)
{
  QualType OrigDestType = DestType, OrigSrcType = SrcExpr->getType();

  DestType = Self.Context.getCanonicalType(DestType);
  QualType SrcType = SrcExpr->getType();
  if (const LValueReferenceType *DestTypeTmp =
        DestType->getAsLValueReferenceType()) {
    if (SrcExpr->isLvalue(Self.Context) != Expr::LV_Valid) {
      // Cannot cast non-lvalue to reference type.
      Self.Diag(OpRange.getBegin(), diag::err_bad_cxx_cast_rvalue)
        << "reinterpret_cast" << OrigDestType << SrcExpr->getSourceRange();
      return;
    }

    // C++ 5.2.10p10: [...] a reference cast reinterpret_cast<T&>(x) has the
    //   same effect as the conversion *reinterpret_cast<T*>(&x) with the
    //   built-in & and * operators.
    // This code does this transformation for the checked types.
    DestType = Self.Context.getPointerType(DestTypeTmp->getPointeeType());
    SrcType = Self.Context.getPointerType(SrcType);
  } else if (const RValueReferenceType *DestTypeTmp =
               DestType->getAsRValueReferenceType()) {
    // Both the reference conversion and the rvalue rules apply.
    Self.DefaultFunctionArrayConversion(SrcExpr);
    SrcType = SrcExpr->getType();

    DestType = Self.Context.getPointerType(DestTypeTmp->getPointeeType());
    SrcType = Self.Context.getPointerType(SrcType);
  } else {
    // C++ 5.2.10p1: [...] the lvalue-to-rvalue, array-to-pointer, and
    //   function-to-pointer standard conversions are performed on the
    //   expression v.
    Self.DefaultFunctionArrayConversion(SrcExpr);
    SrcType = SrcExpr->getType();
  }

  // Canonicalize source for comparison.
  SrcType = Self.Context.getCanonicalType(SrcType);

  const MemberPointerType *DestMemPtr = DestType->getAsMemberPointerType(),
                          *SrcMemPtr = SrcType->getAsMemberPointerType();
  if (DestMemPtr && SrcMemPtr) {
    // C++ 5.2.10p9: An rvalue of type "pointer to member of X of type T1"
    //   can be explicitly converted to an rvalue of type "pointer to member
    //   of Y of type T2" if T1 and T2 are both function types or both object
    //   types.
    if (DestMemPtr->getPointeeType()->isFunctionType() !=
        SrcMemPtr->getPointeeType()->isFunctionType()) {
      Self.Diag(OpRange.getBegin(), diag::err_bad_cxx_cast_generic)
        << "reinterpret_cast" << OrigDestType << OrigSrcType << OpRange;
      return;
    }

    // C++ 5.2.10p2: The reinterpret_cast operator shall not cast away
    //   constness.
    if (CastsAwayConstness(Self, SrcType, DestType)) {
      Self.Diag(OpRange.getBegin(), diag::err_bad_cxx_cast_const_away)
        << "reinterpret_cast" << OrigDestType << OrigSrcType << OpRange;
      return;
    }

    // A valid member pointer cast.
    return;
  }

  bool destIsPtr = DestType->isPointerType();
  bool srcIsPtr = SrcType->isPointerType();
  if (!destIsPtr && !srcIsPtr) {
    // Except for std::nullptr_t->integer, which is not supported yet, and
    // lvalue->reference, which is handled above, at least one of the two
    // arguments must be a pointer.
    Self.Diag(OpRange.getBegin(), diag::err_bad_cxx_cast_generic)
      << "reinterpret_cast" << OrigDestType << OrigSrcType << OpRange;
    return;
  }

  if (SrcType == DestType) {
    // C++ 5.2.10p2 has a note that mentions that, subject to all other
    // restrictions, a cast to the same type is allowed. The intent is not
    // entirely clear here, since all other paragraphs explicitly forbid casts
    // to the same type. However, the behavior of compilers is pretty consistent
    // on this point: allow same-type conversion if the involved types are
    // pointers, disallow otherwise.
    return;
  }

  // Note: Clang treats enumeration types as integral types. If this is ever
  // changed for C++, the additional check here will be redundant.
  if (DestType->isIntegralType() && !DestType->isEnumeralType()) {
    assert(srcIsPtr && "One type must be a pointer");
    // C++ 5.2.10p4: A pointer can be explicitly converted to any integral
    //   type large enough to hold it.
    if (Self.Context.getTypeSize(SrcType) >
        Self.Context.getTypeSize(DestType)) {
      Self.Diag(OpRange.getBegin(), diag::err_bad_reinterpret_cast_small_int)
        << OrigDestType << DestRange;
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
    Self.Diag(OpRange.getBegin(), diag::err_bad_cxx_cast_generic)
      << "reinterpret_cast" << OrigDestType << OrigSrcType << OpRange;
    return;
  }

  // C++ 5.2.10p2: The reinterpret_cast operator shall not cast away constness.
  if (CastsAwayConstness(Self, SrcType, DestType)) {
    Self.Diag(OpRange.getBegin(), diag::err_bad_cxx_cast_const_away)
      << "reinterpret_cast" << OrigDestType << OrigSrcType << OpRange;
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

    // C++0x 5.2.10p8: Converting a pointer to a function into a pointer to
    //   an object type or vice versa is conditionally-supported.
    // Compilers support it in C++03 too, though, because it's necessary for
    // casting the return value of dlsym() and GetProcAddress().
    // FIXME: Conditionally-supported behavior should be configurable in the
    // TargetInfo or similar.
    if (!Self.getLangOptions().CPlusPlus0x) {
      Self.Diag(OpRange.getBegin(), diag::ext_reinterpret_cast_fn_obj)
        << OpRange;
    }
    return;
  }

  if (DestType->isFunctionPointerType()) {
    // See above.
    if (!Self.getLangOptions().CPlusPlus0x) {
      Self.Diag(OpRange.getBegin(), diag::ext_reinterpret_cast_fn_obj)
        << OpRange;
    }
    return;
  }

  // C++ 5.2.10p7: A pointer to an object can be explicitly converted to
  //   a pointer to an object of different type.
  // Void pointers are not specified, but supported by every compiler out there.
  // So we finish by allowing everything that remains - it's got to be two
  // object pointers.
}

/// CastsAwayConstness - Check if the pointer conversion from SrcType to
/// DestType casts away constness as defined in C++ 5.2.11p8ff. This is used by
/// the cast checkers.  Both arguments must denote pointer (possibly to member)
/// types.
bool
CastsAwayConstness(Sema &Self, QualType SrcType, QualType DestType)
{
  // Casting away constness is defined in C++ 5.2.11p8 with reference to
  // C++ 4.4. We piggyback on Sema::IsQualificationConversion for this, since
  // the rules are non-trivial. So first we construct Tcv *...cv* as described
  // in C++ 5.2.11p8.
  assert((SrcType->isPointerType() || SrcType->isMemberPointerType()) &&
         "Source type is not pointer or pointer to member.");
  assert((DestType->isPointerType() || DestType->isMemberPointerType()) &&
         "Destination type is not pointer or pointer to member.");

  QualType UnwrappedSrcType = SrcType, UnwrappedDestType = DestType;
  llvm::SmallVector<unsigned, 8> cv1, cv2;

  // Find the qualifications.
  while (Self.UnwrapSimilarPointerTypes(UnwrappedSrcType, UnwrappedDestType)) {
    cv1.push_back(UnwrappedSrcType.getCVRQualifiers());
    cv2.push_back(UnwrappedDestType.getCVRQualifiers());
  }
  assert(cv1.size() > 0 && "Must have at least one pointer level.");

  // Construct void pointers with those qualifiers (in reverse order of
  // unwrapping, of course).
  QualType SrcConstruct = Self.Context.VoidTy;
  QualType DestConstruct = Self.Context.VoidTy;
  for (llvm::SmallVector<unsigned, 8>::reverse_iterator i1 = cv1.rbegin(),
                                                        i2 = cv2.rbegin();
       i1 != cv1.rend(); ++i1, ++i2)
  {
    SrcConstruct = Self.Context.getPointerType(
      SrcConstruct.getQualifiedType(*i1));
    DestConstruct = Self.Context.getPointerType(
      DestConstruct.getQualifiedType(*i2));
  }

  // Test if they're compatible.
  return SrcConstruct != DestConstruct &&
    !Self.IsQualificationConversion(SrcConstruct, DestConstruct);
}

/// CheckStaticCast - Check that a static_cast\<DestType\>(SrcExpr) is valid.
/// Refer to C++ 5.2.9 for details. Static casts are mostly used for making
/// implicit conversions explicit and getting rid of data loss warnings.
void
CheckStaticCast(Sema &Self, Expr *&SrcExpr, QualType DestType,
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

  // FIXME: With N2812, casts to rvalue refs will change.

  // C++ 5.2.9p4: Any expression can be explicitly converted to type "cv void".
  if (DestType->isVoidType()) {
    return;
  }

  // C++ 5.2.9p5, reference downcast.
  // See the function for details.
  // DR 427 specifies that this is to be applied before paragraph 2.
  if (TryStaticReferenceDowncast(Self, SrcExpr, DestType, OpRange)
      > TSC_NotApplicable) {
    return;
  }

  // N2844 5.2.9p3: An lvalue of type "cv1 T1" can be cast to type "rvalue
  //   reference to cv2 T2" if "cv2 T2" is reference-compatible with "cv1 T1".
  if (TryLValueToRValueCast(Self, SrcExpr, DestType, OpRange) >
      TSC_NotApplicable) {
    return;
  }

  // C++ 5.2.9p2: An expression e can be explicitly converted to a type T
  //   [...] if the declaration "T t(e);" is well-formed, [...].
  if (TryStaticImplicitCast(Self, SrcExpr, DestType, OpRange) >
      TSC_NotApplicable) {
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
  Self.DefaultFunctionArrayConversion(SrcExpr);

  QualType SrcType = Self.Context.getCanonicalType(SrcExpr->getType());

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
  if (TryStaticPointerDowncast(Self, SrcType, DestType, OpRange)
      > TSC_NotApplicable) {
    return;
  }

  // Reverse member pointer conversion. C++ 4.11 specifies member pointer
  // conversion. C++ 5.2.9p9 has additional information.
  // DR54's access restrictions apply here also.
  if (TryStaticMemberPointerUpcast(Self, SrcType, DestType, OpRange)
      > TSC_NotApplicable) {
    return;
  }

  // Reverse pointer conversion to void*. C++ 4.10.p2 specifies conversion to
  // void*. C++ 5.2.9p10 specifies additional restrictions, which really is
  // just the usual constness stuff.
  if (const PointerType *SrcPointer = SrcType->getAsPointerType()) {
    QualType SrcPointee = SrcPointer->getPointeeType();
    if (SrcPointee->isVoidType()) {
      if (const PointerType *DestPointer = DestType->getAsPointerType()) {
        QualType DestPointee = DestPointer->getPointeeType();
        if (DestPointee->isIncompleteOrObjectType()) {
          // This is definitely the intended conversion, but it might fail due
          // to a const violation.
          if (!DestPointee.isAtLeastAsQualifiedAs(SrcPointee)) {
            Self.Diag(OpRange.getBegin(), diag::err_bad_cxx_cast_const_away)
              << "static_cast" << DestType << OrigSrcType << OpRange;
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
  Self.Diag(OpRange.getBegin(), diag::err_bad_cxx_cast_generic)
    << "static_cast" << DestType << OrigSrcType
    << OpRange;
}

/// Tests whether a conversion according to N2844 is valid.
TryStaticCastResult
TryLValueToRValueCast(Sema &Self, Expr *SrcExpr, QualType DestType,
                      const SourceRange &OpRange)
{
  // N2844 5.2.9p3: An lvalue of type "cv1 T1" can be cast to type "rvalue
  //   reference to cv2 T2" if "cv2 T2" is reference-compatible with "cv1 T1".
  const RValueReferenceType *R = DestType->getAsRValueReferenceType();
  if (!R)
    return TSC_NotApplicable;

  if (SrcExpr->isLvalue(Self.Context) != Expr::LV_Valid)
    return TSC_NotApplicable;

  // Because we try the reference downcast before this function, from now on
  // this is the only cast possibility, so we issue an error if we fail now.
  bool DerivedToBase;
  if (Self.CompareReferenceRelationship(SrcExpr->getType(), R->getPointeeType(),
                                        DerivedToBase) <
        Sema::Ref_Compatible_With_Added_Qualification) {
    Self.Diag(OpRange.getBegin(), diag::err_bad_lvalue_to_rvalue_cast)
      << SrcExpr->getType() << R->getPointeeType() << OpRange;
    return TSC_Failed;
  }

  // FIXME: Similar to CheckReferenceInit, we actually need more AST annotation
  // than nothing.
  return TSC_Success;
}

/// Tests whether a conversion according to C++ 5.2.9p5 is valid.
TryStaticCastResult
TryStaticReferenceDowncast(Sema &Self, Expr *SrcExpr, QualType DestType,
                           const SourceRange &OpRange)
{
  // C++ 5.2.9p5: An lvalue of type "cv1 B", where B is a class type, can be
  //   cast to type "reference to cv2 D", where D is a class derived from B,
  //   if a valid standard conversion from "pointer to D" to "pointer to B"
  //   exists, cv2 >= cv1, and B is not a virtual base class of D.
  // In addition, DR54 clarifies that the base must be accessible in the
  // current context. Although the wording of DR54 only applies to the pointer
  // variant of this rule, the intent is clearly for it to apply to the this
  // conversion as well.

  if (SrcExpr->isLvalue(Self.Context) != Expr::LV_Valid) {
    return TSC_NotApplicable;
  }

  const ReferenceType *DestReference = DestType->getAsReferenceType();
  if (!DestReference) {
    return TSC_NotApplicable;
  }
  QualType DestPointee = DestReference->getPointeeType();

  return TryStaticDowncast(Self, SrcExpr->getType(), DestPointee, OpRange,
                          SrcExpr->getType(), DestType);
}

/// Tests whether a conversion according to C++ 5.2.9p8 is valid.
TryStaticCastResult
TryStaticPointerDowncast(Sema &Self, QualType SrcType, QualType DestType,
                         const SourceRange &OpRange)
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
    return TSC_NotApplicable;
  }

  const PointerType *DestPointer = DestType->getAsPointerType();
  if (!DestPointer) {
    return TSC_NotApplicable;
  }

  return TryStaticDowncast(Self, SrcPointer->getPointeeType(),
                          DestPointer->getPointeeType(),
                          OpRange, SrcType, DestType);
}

/// TryStaticDowncast - Common functionality of TryStaticReferenceDowncast and
/// TryStaticPointerDowncast. Tests whether a static downcast from SrcType to
/// DestType, both of which must be canonical, is possible and allowed.
TryStaticCastResult
TryStaticDowncast(Sema &Self, QualType SrcType, QualType DestType,
                  const SourceRange &OpRange, QualType OrigSrcType,
                  QualType OrigDestType)
{
  // Downcast can only happen in class hierarchies, so we need classes.
  if (!DestType->isRecordType() || !SrcType->isRecordType()) {
    return TSC_NotApplicable;
  }

  BasePaths Paths(/*FindAmbiguities=*/true, /*RecordPaths=*/false,
                  /*DetectVirtual=*/true);
  if (!Self.IsDerivedFrom(DestType, SrcType, Paths)) {
    return TSC_NotApplicable;
  }

  // Target type does derive from source type. Now we're serious. If an error
  // appears now, it's not ignored.
  // This may not be entirely in line with the standard. Take for example:
  // struct A {};
  // struct B : virtual A {
  //   B(A&);
  // };
  // 
  // void f()
  // {
  //   (void)static_cast<const B&>(*((A*)0));
  // }
  // As far as the standard is concerned, p5 does not apply (A is virtual), so
  // p2 should be used instead - "const B& t(*((A*)0));" is perfectly valid.
  // However, both GCC and Comeau reject this example, and accepting it would
  // mean more complex code if we're to preserve the nice error message.
  // FIXME: Being 100% compliant here would be nice to have.

  // Must preserve cv, as always.
  if (!DestType.isAtLeastAsQualifiedAs(SrcType)) {
    Self.Diag(OpRange.getBegin(), diag::err_bad_cxx_cast_const_away)
      << "static_cast" << OrigDestType << OrigSrcType << OpRange;
    return TSC_Failed;
  }

  if (Paths.isAmbiguous(SrcType.getUnqualifiedType())) {
    // This code is analoguous to that in CheckDerivedToBaseConversion, except
    // that it builds the paths in reverse order.
    // To sum up: record all paths to the base and build a nice string from
    // them. Use it to spice up the error message.
    Paths.clear();
    Paths.setRecordingPaths(true);
    Self.IsDerivedFrom(DestType, SrcType, Paths);
    std::string PathDisplayStr;
    std::set<unsigned> DisplayedPaths;
    for (BasePaths::paths_iterator Path = Paths.begin();
         Path != Paths.end(); ++Path) {
      if (DisplayedPaths.insert(Path->back().SubobjectNumber).second) {
        // We haven't displayed a path to this particular base
        // class subobject yet.
        PathDisplayStr += "\n    ";
        for (BasePath::const_reverse_iterator Element = Path->rbegin();
             Element != Path->rend(); ++Element)
          PathDisplayStr += Element->Base->getType().getAsString() + " -> ";
        PathDisplayStr += DestType.getAsString();
      }
    }

    Self.Diag(OpRange.getBegin(), diag::err_ambiguous_base_to_derived_cast)
      << SrcType.getUnqualifiedType() << DestType.getUnqualifiedType()
      << PathDisplayStr << OpRange;
    return TSC_Failed;
  }

  if (Paths.getDetectedVirtual() != 0) {
    QualType VirtualBase(Paths.getDetectedVirtual(), 0);
    Self.Diag(OpRange.getBegin(), diag::err_static_downcast_via_virtual)
      << OrigSrcType << OrigDestType << VirtualBase << OpRange;
    return TSC_Failed;
  }

  // FIXME: Test accessibility.

  return TSC_Success;
}

/// TryStaticMemberPointerUpcast - Tests whether a conversion according to
/// C++ 5.2.9p9 is valid:
///
///   An rvalue of type "pointer to member of D of type cv1 T" can be
///   converted to an rvalue of type "pointer to member of B of type cv2 T",
///   where B is a base class of D [...].
///
TryStaticCastResult
TryStaticMemberPointerUpcast(Sema &Self, QualType SrcType, QualType DestType,
                             const SourceRange &OpRange)
{
  const MemberPointerType *SrcMemPtr = SrcType->getAsMemberPointerType();
  if (!SrcMemPtr)
    return TSC_NotApplicable;
  const MemberPointerType *DestMemPtr = DestType->getAsMemberPointerType();
  if (!DestMemPtr)
    return TSC_NotApplicable;

  // T == T, modulo cv
  if (Self.Context.getCanonicalType(
        SrcMemPtr->getPointeeType().getUnqualifiedType()) !=
      Self.Context.getCanonicalType(DestMemPtr->getPointeeType().
                                    getUnqualifiedType()))
    return TSC_NotApplicable;

  // B base of D
  QualType SrcClass(SrcMemPtr->getClass(), 0);
  QualType DestClass(DestMemPtr->getClass(), 0);
  BasePaths Paths(/*FindAmbiguities=*/true, /*RecordPaths=*/false,
                  /*DetectVirtual=*/true);
  if (!Self.IsDerivedFrom(SrcClass, DestClass, Paths)) {
    return TSC_NotApplicable;
  }

  // B is a base of D. But is it an allowed base? If not, it's a hard error.
  if (Paths.isAmbiguous(DestClass)) {
    Paths.clear();
    Paths.setRecordingPaths(true);
    bool StillOkay = Self.IsDerivedFrom(SrcClass, DestClass, Paths);
    assert(StillOkay);
    StillOkay = StillOkay;
    std::string PathDisplayStr = Self.getAmbiguousPathsDisplayString(Paths);
    Self.Diag(OpRange.getBegin(), diag::err_ambiguous_memptr_conv)
      << 1 << SrcClass << DestClass << PathDisplayStr << OpRange;
    return TSC_Failed;
  }

  if (const RecordType *VBase = Paths.getDetectedVirtual()) {
    Self.Diag(OpRange.getBegin(), diag::err_memptr_conv_via_virtual)
      << SrcClass << DestClass << QualType(VBase, 0) << OpRange;
    return TSC_Failed;
  }

  // FIXME: Test accessibility.

  return TSC_Success;
}

/// TryStaticImplicitCast - Tests whether a conversion according to C++ 5.2.9p2
/// is valid:
///
///   An expression e can be explicitly converted to a type T using a
///   @c static_cast if the declaration "T t(e);" is well-formed [...].
TryStaticCastResult
TryStaticImplicitCast(Sema &Self, Expr *SrcExpr, QualType DestType,
                      const SourceRange &OpRange)
{
  if (DestType->isReferenceType()) {
    // At this point of CheckStaticCast, if the destination is a reference,
    // this has to work. There is no other way that works.
    return Self.CheckReferenceInit(SrcExpr, DestType) ?
      TSC_Failed : TSC_Success;
  }
  if (DestType->isRecordType()) {
    // FIXME: Use an implementation of C++ [over.match.ctor] for this.
    return TSC_NotApplicable;
  }

  // FIXME: To get a proper error from invalid conversions here, we need to
  // reimplement more of this.
  ImplicitConversionSequence ICS = Self.TryImplicitConversion(
    SrcExpr, DestType);
  return ICS.ConversionKind == ImplicitConversionSequence::BadConversion ?
    TSC_NotApplicable : TSC_Success;
}

/// CheckDynamicCast - Check that a dynamic_cast\<DestType\>(SrcExpr) is valid.
/// Refer to C++ 5.2.7 for details. Dynamic casts are used mostly for runtime-
/// checked downcasts in class hierarchies.
void
CheckDynamicCast(Sema &Self, Expr *&SrcExpr, QualType DestType,
                 const SourceRange &OpRange,
                 const SourceRange &DestRange)
{
  QualType OrigDestType = DestType, OrigSrcType = SrcExpr->getType();
  DestType = Self.Context.getCanonicalType(DestType);

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
    Self.Diag(OpRange.getBegin(), diag::err_bad_dynamic_cast_not_ref_or_ptr)
      << OrigDestType << DestRange;
    return;
  }

  const RecordType *DestRecord = DestPointee->getAsRecordType();
  if (DestPointee->isVoidType()) {
    assert(DestPointer && "Reference to void is not possible");
  } else if (DestRecord) {
    if (Self.RequireCompleteType(OpRange.getBegin(), DestPointee, 
                                    diag::err_bad_dynamic_cast_incomplete,
                                    DestRange))
      return;
  } else {
    Self.Diag(OpRange.getBegin(), diag::err_bad_dynamic_cast_not_class)
      << DestPointee.getUnqualifiedType() << DestRange;
    return;
  }

  // C++0x 5.2.7p2: If T is a pointer type, v shall be an rvalue of a pointer to
  //   complete class type, [...]. If T is an lvalue reference type, v shall be
  //   an lvalue of a complete class type, [...]. If T is an rvalue reference
  //   type, v shall be an expression having a complete effective class type,
  //   [...]

  QualType SrcType = Self.Context.getCanonicalType(OrigSrcType);
  QualType SrcPointee;
  if (DestPointer) {
    if (const PointerType *SrcPointer = SrcType->getAsPointerType()) {
      SrcPointee = SrcPointer->getPointeeType();
    } else {
      Self.Diag(OpRange.getBegin(), diag::err_bad_dynamic_cast_not_ptr)
        << OrigSrcType << SrcExpr->getSourceRange();
      return;
    }
  } else if (DestReference->isLValueReferenceType()) {
    if (SrcExpr->isLvalue(Self.Context) != Expr::LV_Valid) {
      Self.Diag(OpRange.getBegin(), diag::err_bad_cxx_cast_rvalue)
        << "dynamic_cast" << OrigDestType << OpRange;
    }
    SrcPointee = SrcType;
  } else {
    SrcPointee = SrcType;
  }

  const RecordType *SrcRecord = SrcPointee->getAsRecordType();
  if (SrcRecord) {
    if (Self.RequireCompleteType(OpRange.getBegin(), SrcPointee,
                                    diag::err_bad_dynamic_cast_incomplete,
                                    SrcExpr->getSourceRange()))
      return;
  } else {
    Self.Diag(OpRange.getBegin(), diag::err_bad_dynamic_cast_not_class)
      << SrcPointee.getUnqualifiedType() << SrcExpr->getSourceRange();
    return;
  }

  assert((DestPointer || DestReference) &&
    "Bad destination non-ptr/ref slipped through.");
  assert((DestRecord || DestPointee->isVoidType()) &&
    "Bad destination pointee slipped through.");
  assert(SrcRecord && "Bad source pointee slipped through.");

  // C++ 5.2.7p1: The dynamic_cast operator shall not cast away constness.
  if (!DestPointee.isAtLeastAsQualifiedAs(SrcPointee)) {
    Self.Diag(OpRange.getBegin(), diag::err_bad_cxx_cast_const_away)
      << "dynamic_cast" << OrigDestType << OrigSrcType << OpRange;
    return;
  }

  // C++ 5.2.7p3: If the type of v is the same as the required result type,
  //   [except for cv].
  if (DestRecord == SrcRecord) {
    return;
  }

  // C++ 5.2.7p5
  // Upcasts are resolved statically.
  if (DestRecord && Self.IsDerivedFrom(SrcPointee, DestPointee)) {
    Self.CheckDerivedToBaseConversion(SrcPointee, DestPointee,
                                      OpRange.getBegin(), OpRange);
    // Diagnostic already emitted on error.
    return;
  }

  // C++ 5.2.7p6: Otherwise, v shall be [polymorphic].
  const RecordDecl *SrcDecl = SrcRecord->getDecl()->getDefinition(Self.Context);
  assert(SrcDecl && "Definition missing");
  if (!cast<CXXRecordDecl>(SrcDecl)->isPolymorphic()) {
    Self.Diag(OpRange.getBegin(), diag::err_bad_dynamic_cast_not_polymorphic)
      << SrcPointee.getUnqualifiedType() << SrcExpr->getSourceRange();
  }

  // Done. Everything else is run-time checks.
}
