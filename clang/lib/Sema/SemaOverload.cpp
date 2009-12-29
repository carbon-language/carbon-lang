//===--- SemaOverload.cpp - C++ Overloading ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides Sema routines for C++ overloading.
//
//===----------------------------------------------------------------------===//

#include "Sema.h"
#include "Lookup.h"
#include "SemaInit.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/TypeOrdering.h"
#include "clang/Basic/PartialDiagnostic.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <cstdio>

namespace clang {

/// GetConversionCategory - Retrieve the implicit conversion
/// category corresponding to the given implicit conversion kind.
ImplicitConversionCategory
GetConversionCategory(ImplicitConversionKind Kind) {
  static const ImplicitConversionCategory
    Category[(int)ICK_Num_Conversion_Kinds] = {
    ICC_Identity,
    ICC_Lvalue_Transformation,
    ICC_Lvalue_Transformation,
    ICC_Lvalue_Transformation,
    ICC_Identity,
    ICC_Qualification_Adjustment,
    ICC_Promotion,
    ICC_Promotion,
    ICC_Promotion,
    ICC_Conversion,
    ICC_Conversion,
    ICC_Conversion,
    ICC_Conversion,
    ICC_Conversion,
    ICC_Conversion,
    ICC_Conversion,
    ICC_Conversion,
    ICC_Conversion,
    ICC_Conversion
  };
  return Category[(int)Kind];
}

/// GetConversionRank - Retrieve the implicit conversion rank
/// corresponding to the given implicit conversion kind.
ImplicitConversionRank GetConversionRank(ImplicitConversionKind Kind) {
  static const ImplicitConversionRank
    Rank[(int)ICK_Num_Conversion_Kinds] = {
    ICR_Exact_Match,
    ICR_Exact_Match,
    ICR_Exact_Match,
    ICR_Exact_Match,
    ICR_Exact_Match,
    ICR_Exact_Match,
    ICR_Promotion,
    ICR_Promotion,
    ICR_Promotion,
    ICR_Conversion,
    ICR_Conversion,
    ICR_Conversion,
    ICR_Conversion,
    ICR_Conversion,
    ICR_Conversion,
    ICR_Conversion,
    ICR_Conversion,
    ICR_Conversion,
    ICR_Conversion
  };
  return Rank[(int)Kind];
}

/// GetImplicitConversionName - Return the name of this kind of
/// implicit conversion.
const char* GetImplicitConversionName(ImplicitConversionKind Kind) {
  static const char* const Name[(int)ICK_Num_Conversion_Kinds] = {
    "No conversion",
    "Lvalue-to-rvalue",
    "Array-to-pointer",
    "Function-to-pointer",
    "Noreturn adjustment",
    "Qualification",
    "Integral promotion",
    "Floating point promotion",
    "Complex promotion",
    "Integral conversion",
    "Floating conversion",
    "Complex conversion",
    "Floating-integral conversion",
    "Complex-real conversion",
    "Pointer conversion",
    "Pointer-to-member conversion",
    "Boolean conversion",
    "Compatible-types conversion",
    "Derived-to-base conversion"
  };
  return Name[Kind];
}

/// StandardConversionSequence - Set the standard conversion
/// sequence to the identity conversion.
void StandardConversionSequence::setAsIdentityConversion() {
  First = ICK_Identity;
  Second = ICK_Identity;
  Third = ICK_Identity;
  Deprecated = false;
  ReferenceBinding = false;
  DirectBinding = false;
  RRefBinding = false;
  CopyConstructor = 0;
}

/// getRank - Retrieve the rank of this standard conversion sequence
/// (C++ 13.3.3.1.1p3). The rank is the largest rank of each of the
/// implicit conversions.
ImplicitConversionRank StandardConversionSequence::getRank() const {
  ImplicitConversionRank Rank = ICR_Exact_Match;
  if  (GetConversionRank(First) > Rank)
    Rank = GetConversionRank(First);
  if  (GetConversionRank(Second) > Rank)
    Rank = GetConversionRank(Second);
  if  (GetConversionRank(Third) > Rank)
    Rank = GetConversionRank(Third);
  return Rank;
}

/// isPointerConversionToBool - Determines whether this conversion is
/// a conversion of a pointer or pointer-to-member to bool. This is
/// used as part of the ranking of standard conversion sequences
/// (C++ 13.3.3.2p4).
bool StandardConversionSequence::isPointerConversionToBool() const {
  QualType FromType = QualType::getFromOpaquePtr(FromTypePtr);
  QualType ToType = QualType::getFromOpaquePtr(ToTypePtr);

  // Note that FromType has not necessarily been transformed by the
  // array-to-pointer or function-to-pointer implicit conversions, so
  // check for their presence as well as checking whether FromType is
  // a pointer.
  if (ToType->isBooleanType() &&
      (FromType->isPointerType() || FromType->isBlockPointerType() ||
       First == ICK_Array_To_Pointer || First == ICK_Function_To_Pointer))
    return true;

  return false;
}

/// isPointerConversionToVoidPointer - Determines whether this
/// conversion is a conversion of a pointer to a void pointer. This is
/// used as part of the ranking of standard conversion sequences (C++
/// 13.3.3.2p4).
bool
StandardConversionSequence::
isPointerConversionToVoidPointer(ASTContext& Context) const {
  QualType FromType = QualType::getFromOpaquePtr(FromTypePtr);
  QualType ToType = QualType::getFromOpaquePtr(ToTypePtr);

  // Note that FromType has not necessarily been transformed by the
  // array-to-pointer implicit conversion, so check for its presence
  // and redo the conversion to get a pointer.
  if (First == ICK_Array_To_Pointer)
    FromType = Context.getArrayDecayedType(FromType);

  if (Second == ICK_Pointer_Conversion && FromType->isPointerType())
    if (const PointerType* ToPtrType = ToType->getAs<PointerType>())
      return ToPtrType->getPointeeType()->isVoidType();

  return false;
}

/// DebugPrint - Print this standard conversion sequence to standard
/// error. Useful for debugging overloading issues.
void StandardConversionSequence::DebugPrint() const {
  bool PrintedSomething = false;
  if (First != ICK_Identity) {
    fprintf(stderr, "%s", GetImplicitConversionName(First));
    PrintedSomething = true;
  }

  if (Second != ICK_Identity) {
    if (PrintedSomething) {
      fprintf(stderr, " -> ");
    }
    fprintf(stderr, "%s", GetImplicitConversionName(Second));

    if (CopyConstructor) {
      fprintf(stderr, " (by copy constructor)");
    } else if (DirectBinding) {
      fprintf(stderr, " (direct reference binding)");
    } else if (ReferenceBinding) {
      fprintf(stderr, " (reference binding)");
    }
    PrintedSomething = true;
  }

  if (Third != ICK_Identity) {
    if (PrintedSomething) {
      fprintf(stderr, " -> ");
    }
    fprintf(stderr, "%s", GetImplicitConversionName(Third));
    PrintedSomething = true;
  }

  if (!PrintedSomething) {
    fprintf(stderr, "No conversions required");
  }
}

/// DebugPrint - Print this user-defined conversion sequence to standard
/// error. Useful for debugging overloading issues.
void UserDefinedConversionSequence::DebugPrint() const {
  if (Before.First || Before.Second || Before.Third) {
    Before.DebugPrint();
    fprintf(stderr, " -> ");
  }
  fprintf(stderr, "'%s'", ConversionFunction->getNameAsString().c_str());
  if (After.First || After.Second || After.Third) {
    fprintf(stderr, " -> ");
    After.DebugPrint();
  }
}

/// DebugPrint - Print this implicit conversion sequence to standard
/// error. Useful for debugging overloading issues.
void ImplicitConversionSequence::DebugPrint() const {
  switch (ConversionKind) {
  case StandardConversion:
    fprintf(stderr, "Standard conversion: ");
    Standard.DebugPrint();
    break;
  case UserDefinedConversion:
    fprintf(stderr, "User-defined conversion: ");
    UserDefined.DebugPrint();
    break;
  case EllipsisConversion:
    fprintf(stderr, "Ellipsis conversion");
    break;
  case BadConversion:
    fprintf(stderr, "Bad conversion");
    break;
  }

  fprintf(stderr, "\n");
}

// IsOverload - Determine whether the given New declaration is an
// overload of the declarations in Old. This routine returns false if
// New and Old cannot be overloaded, e.g., if New has the same
// signature as some function in Old (C++ 1.3.10) or if the Old
// declarations aren't functions (or function templates) at all. When
// it does return false, MatchedDecl will point to the decl that New
// cannot be overloaded with.  This decl may be a UsingShadowDecl on
// top of the underlying declaration.
//
// Example: Given the following input:
//
//   void f(int, float); // #1
//   void f(int, int); // #2
//   int f(int, int); // #3
//
// When we process #1, there is no previous declaration of "f",
// so IsOverload will not be used.
//
// When we process #2, Old contains only the FunctionDecl for #1.  By
// comparing the parameter types, we see that #1 and #2 are overloaded
// (since they have different signatures), so this routine returns
// false; MatchedDecl is unchanged.
//
// When we process #3, Old is an overload set containing #1 and #2. We
// compare the signatures of #3 to #1 (they're overloaded, so we do
// nothing) and then #3 to #2. Since the signatures of #3 and #2 are
// identical (return types of functions are not part of the
// signature), IsOverload returns false and MatchedDecl will be set to
// point to the FunctionDecl for #2.
Sema::OverloadKind
Sema::CheckOverload(FunctionDecl *New, const LookupResult &Old,
                    NamedDecl *&Match) {
  for (LookupResult::iterator I = Old.begin(), E = Old.end();
         I != E; ++I) {
    NamedDecl *OldD = (*I)->getUnderlyingDecl();
    if (FunctionTemplateDecl *OldT = dyn_cast<FunctionTemplateDecl>(OldD)) {
      if (!IsOverload(New, OldT->getTemplatedDecl())) {
        Match = *I;
        return Ovl_Match;
      }
    } else if (FunctionDecl *OldF = dyn_cast<FunctionDecl>(OldD)) {
      if (!IsOverload(New, OldF)) {
        Match = *I;
        return Ovl_Match;
      }
    } else if (isa<UsingDecl>(OldD) || isa<TagDecl>(OldD)) {
      // We can overload with these, which can show up when doing
      // redeclaration checks for UsingDecls.
      assert(Old.getLookupKind() == LookupUsingDeclName);
    } else if (isa<UnresolvedUsingValueDecl>(OldD)) {
      // Optimistically assume that an unresolved using decl will
      // overload; if it doesn't, we'll have to diagnose during
      // template instantiation.
    } else {
      // (C++ 13p1):
      //   Only function declarations can be overloaded; object and type
      //   declarations cannot be overloaded.
      Match = *I;
      return Ovl_NonFunction;
    }
  }

  return Ovl_Overload;
}

bool Sema::IsOverload(FunctionDecl *New, FunctionDecl *Old) {
  FunctionTemplateDecl *OldTemplate = Old->getDescribedFunctionTemplate();
  FunctionTemplateDecl *NewTemplate = New->getDescribedFunctionTemplate();

  // C++ [temp.fct]p2:
  //   A function template can be overloaded with other function templates
  //   and with normal (non-template) functions.
  if ((OldTemplate == 0) != (NewTemplate == 0))
    return true;

  // Is the function New an overload of the function Old?
  QualType OldQType = Context.getCanonicalType(Old->getType());
  QualType NewQType = Context.getCanonicalType(New->getType());

  // Compare the signatures (C++ 1.3.10) of the two functions to
  // determine whether they are overloads. If we find any mismatch
  // in the signature, they are overloads.

  // If either of these functions is a K&R-style function (no
  // prototype), then we consider them to have matching signatures.
  if (isa<FunctionNoProtoType>(OldQType.getTypePtr()) ||
      isa<FunctionNoProtoType>(NewQType.getTypePtr()))
    return false;

  FunctionProtoType* OldType = cast<FunctionProtoType>(OldQType);
  FunctionProtoType* NewType = cast<FunctionProtoType>(NewQType);

  // The signature of a function includes the types of its
  // parameters (C++ 1.3.10), which includes the presence or absence
  // of the ellipsis; see C++ DR 357).
  if (OldQType != NewQType &&
      (OldType->getNumArgs() != NewType->getNumArgs() ||
       OldType->isVariadic() != NewType->isVariadic() ||
       !std::equal(OldType->arg_type_begin(), OldType->arg_type_end(),
                   NewType->arg_type_begin())))
    return true;

  // C++ [temp.over.link]p4:
  //   The signature of a function template consists of its function
  //   signature, its return type and its template parameter list. The names
  //   of the template parameters are significant only for establishing the
  //   relationship between the template parameters and the rest of the
  //   signature.
  //
  // We check the return type and template parameter lists for function
  // templates first; the remaining checks follow.
  if (NewTemplate &&
      (!TemplateParameterListsAreEqual(NewTemplate->getTemplateParameters(),
                                       OldTemplate->getTemplateParameters(),
                                       false, TPL_TemplateMatch) ||
       OldType->getResultType() != NewType->getResultType()))
    return true;

  // If the function is a class member, its signature includes the
  // cv-qualifiers (if any) on the function itself.
  //
  // As part of this, also check whether one of the member functions
  // is static, in which case they are not overloads (C++
  // 13.1p2). While not part of the definition of the signature,
  // this check is important to determine whether these functions
  // can be overloaded.
  CXXMethodDecl* OldMethod = dyn_cast<CXXMethodDecl>(Old);
  CXXMethodDecl* NewMethod = dyn_cast<CXXMethodDecl>(New);
  if (OldMethod && NewMethod &&
      !OldMethod->isStatic() && !NewMethod->isStatic() &&
      OldMethod->getTypeQualifiers() != NewMethod->getTypeQualifiers())
    return true;
  
  // The signatures match; this is not an overload.
  return false;
}

/// TryImplicitConversion - Attempt to perform an implicit conversion
/// from the given expression (Expr) to the given type (ToType). This
/// function returns an implicit conversion sequence that can be used
/// to perform the initialization. Given
///
///   void f(float f);
///   void g(int i) { f(i); }
///
/// this routine would produce an implicit conversion sequence to
/// describe the initialization of f from i, which will be a standard
/// conversion sequence containing an lvalue-to-rvalue conversion (C++
/// 4.1) followed by a floating-integral conversion (C++ 4.9).
//
/// Note that this routine only determines how the conversion can be
/// performed; it does not actually perform the conversion. As such,
/// it will not produce any diagnostics if no conversion is available,
/// but will instead return an implicit conversion sequence of kind
/// "BadConversion".
///
/// If @p SuppressUserConversions, then user-defined conversions are
/// not permitted.
/// If @p AllowExplicit, then explicit user-defined conversions are
/// permitted.
/// If @p ForceRValue, then overloading is performed as if From was an rvalue,
/// no matter its actual lvalueness.
/// If @p UserCast, the implicit conversion is being done for a user-specified
/// cast.
ImplicitConversionSequence
Sema::TryImplicitConversion(Expr* From, QualType ToType,
                            bool SuppressUserConversions,
                            bool AllowExplicit, bool ForceRValue,
                            bool InOverloadResolution,
                            bool UserCast) {
  ImplicitConversionSequence ICS;
  OverloadCandidateSet Conversions;
  OverloadingResult UserDefResult = OR_Success;
  if (IsStandardConversion(From, ToType, InOverloadResolution, ICS.Standard))
    ICS.ConversionKind = ImplicitConversionSequence::StandardConversion;
  else if (getLangOptions().CPlusPlus &&
           (UserDefResult = IsUserDefinedConversion(From, ToType, 
                                   ICS.UserDefined,
                                   Conversions,
                                   !SuppressUserConversions, AllowExplicit,
				   ForceRValue, UserCast)) == OR_Success) {
    ICS.ConversionKind = ImplicitConversionSequence::UserDefinedConversion;
    // C++ [over.ics.user]p4:
    //   A conversion of an expression of class type to the same class
    //   type is given Exact Match rank, and a conversion of an
    //   expression of class type to a base class of that type is
    //   given Conversion rank, in spite of the fact that a copy
    //   constructor (i.e., a user-defined conversion function) is
    //   called for those cases.
    if (CXXConstructorDecl *Constructor
          = dyn_cast<CXXConstructorDecl>(ICS.UserDefined.ConversionFunction)) {
      QualType FromCanon
        = Context.getCanonicalType(From->getType().getUnqualifiedType());
      QualType ToCanon = Context.getCanonicalType(ToType).getUnqualifiedType();
      if (Constructor->isCopyConstructor() &&
          (FromCanon == ToCanon || IsDerivedFrom(FromCanon, ToCanon))) {
        // Turn this into a "standard" conversion sequence, so that it
        // gets ranked with standard conversion sequences.
        ICS.ConversionKind = ImplicitConversionSequence::StandardConversion;
        ICS.Standard.setAsIdentityConversion();
        ICS.Standard.FromTypePtr = From->getType().getAsOpaquePtr();
        ICS.Standard.ToTypePtr = ToType.getAsOpaquePtr();
        ICS.Standard.CopyConstructor = Constructor;
        if (ToCanon != FromCanon)
          ICS.Standard.Second = ICK_Derived_To_Base;
      }
    }

    // C++ [over.best.ics]p4:
    //   However, when considering the argument of a user-defined
    //   conversion function that is a candidate by 13.3.1.3 when
    //   invoked for the copying of the temporary in the second step
    //   of a class copy-initialization, or by 13.3.1.4, 13.3.1.5, or
    //   13.3.1.6 in all cases, only standard conversion sequences and
    //   ellipsis conversion sequences are allowed.
    if (SuppressUserConversions &&
        ICS.ConversionKind == ImplicitConversionSequence::UserDefinedConversion)
      ICS.ConversionKind = ImplicitConversionSequence::BadConversion;
  } else {
    ICS.ConversionKind = ImplicitConversionSequence::BadConversion;
    if (UserDefResult == OR_Ambiguous) {
      for (OverloadCandidateSet::iterator Cand = Conversions.begin();
           Cand != Conversions.end(); ++Cand)
        if (Cand->Viable)
          ICS.ConversionFunctionSet.push_back(Cand->Function);
    }
  }

  return ICS;
}

/// \brief Determine whether the conversion from FromType to ToType is a valid 
/// conversion that strips "noreturn" off the nested function type.
static bool IsNoReturnConversion(ASTContext &Context, QualType FromType, 
                                 QualType ToType, QualType &ResultTy) {
  if (Context.hasSameUnqualifiedType(FromType, ToType))
    return false;
  
  // Strip the noreturn off the type we're converting from; noreturn can
  // safely be removed.
  FromType = Context.getNoReturnType(FromType, false);
  if (!Context.hasSameUnqualifiedType(FromType, ToType))
    return false;

  ResultTy = FromType;
  return true;
}
  
/// IsStandardConversion - Determines whether there is a standard
/// conversion sequence (C++ [conv], C++ [over.ics.scs]) from the
/// expression From to the type ToType. Standard conversion sequences
/// only consider non-class types; for conversions that involve class
/// types, use TryImplicitConversion. If a conversion exists, SCS will
/// contain the standard conversion sequence required to perform this
/// conversion and this routine will return true. Otherwise, this
/// routine will return false and the value of SCS is unspecified.
bool
Sema::IsStandardConversion(Expr* From, QualType ToType,
                           bool InOverloadResolution,
                           StandardConversionSequence &SCS) {
  QualType FromType = From->getType();

  // Standard conversions (C++ [conv])
  SCS.setAsIdentityConversion();
  SCS.Deprecated = false;
  SCS.IncompatibleObjC = false;
  SCS.FromTypePtr = FromType.getAsOpaquePtr();
  SCS.CopyConstructor = 0;

  // There are no standard conversions for class types in C++, so
  // abort early. When overloading in C, however, we do permit
  if (FromType->isRecordType() || ToType->isRecordType()) {
    if (getLangOptions().CPlusPlus)
      return false;

    // When we're overloading in C, we allow, as standard conversions,
  }

  // The first conversion can be an lvalue-to-rvalue conversion,
  // array-to-pointer conversion, or function-to-pointer conversion
  // (C++ 4p1).

  // Lvalue-to-rvalue conversion (C++ 4.1):
  //   An lvalue (3.10) of a non-function, non-array type T can be
  //   converted to an rvalue.
  Expr::isLvalueResult argIsLvalue = From->isLvalue(Context);
  if (argIsLvalue == Expr::LV_Valid &&
      !FromType->isFunctionType() && !FromType->isArrayType() &&
      Context.getCanonicalType(FromType) != Context.OverloadTy) {
    SCS.First = ICK_Lvalue_To_Rvalue;

    // If T is a non-class type, the type of the rvalue is the
    // cv-unqualified version of T. Otherwise, the type of the rvalue
    // is T (C++ 4.1p1). C++ can't get here with class types; in C, we
    // just strip the qualifiers because they don't matter.
    FromType = FromType.getUnqualifiedType();
  } else if (FromType->isArrayType()) {
    // Array-to-pointer conversion (C++ 4.2)
    SCS.First = ICK_Array_To_Pointer;

    // An lvalue or rvalue of type "array of N T" or "array of unknown
    // bound of T" can be converted to an rvalue of type "pointer to
    // T" (C++ 4.2p1).
    FromType = Context.getArrayDecayedType(FromType);

    if (IsStringLiteralToNonConstPointerConversion(From, ToType)) {
      // This conversion is deprecated. (C++ D.4).
      SCS.Deprecated = true;

      // For the purpose of ranking in overload resolution
      // (13.3.3.1.1), this conversion is considered an
      // array-to-pointer conversion followed by a qualification
      // conversion (4.4). (C++ 4.2p2)
      SCS.Second = ICK_Identity;
      SCS.Third = ICK_Qualification;
      SCS.ToTypePtr = ToType.getAsOpaquePtr();
      return true;
    }
  } else if (FromType->isFunctionType() && argIsLvalue == Expr::LV_Valid) {
    // Function-to-pointer conversion (C++ 4.3).
    SCS.First = ICK_Function_To_Pointer;

    // An lvalue of function type T can be converted to an rvalue of
    // type "pointer to T." The result is a pointer to the
    // function. (C++ 4.3p1).
    FromType = Context.getPointerType(FromType);
  } else if (FunctionDecl *Fn
               = ResolveAddressOfOverloadedFunction(From, ToType, false)) {
    // Address of overloaded function (C++ [over.over]).
    SCS.First = ICK_Function_To_Pointer;

    // We were able to resolve the address of the overloaded function,
    // so we can convert to the type of that function.
    FromType = Fn->getType();
    if (ToType->isLValueReferenceType())
      FromType = Context.getLValueReferenceType(FromType);
    else if (ToType->isRValueReferenceType())
      FromType = Context.getRValueReferenceType(FromType);
    else if (ToType->isMemberPointerType()) {
      // Resolve address only succeeds if both sides are member pointers,
      // but it doesn't have to be the same class. See DR 247.
      // Note that this means that the type of &Derived::fn can be
      // Ret (Base::*)(Args) if the fn overload actually found is from the
      // base class, even if it was brought into the derived class via a
      // using declaration. The standard isn't clear on this issue at all.
      CXXMethodDecl *M = cast<CXXMethodDecl>(Fn);
      FromType = Context.getMemberPointerType(FromType,
                    Context.getTypeDeclType(M->getParent()).getTypePtr());
    } else
      FromType = Context.getPointerType(FromType);
  } else {
    // We don't require any conversions for the first step.
    SCS.First = ICK_Identity;
  }

  // The second conversion can be an integral promotion, floating
  // point promotion, integral conversion, floating point conversion,
  // floating-integral conversion, pointer conversion,
  // pointer-to-member conversion, or boolean conversion (C++ 4p1).
  // For overloading in C, this can also be a "compatible-type"
  // conversion.
  bool IncompatibleObjC = false;
  if (Context.hasSameUnqualifiedType(FromType, ToType)) {
    // The unqualified versions of the types are the same: there's no
    // conversion to do.
    SCS.Second = ICK_Identity;
  } else if (IsIntegralPromotion(From, FromType, ToType)) {
    // Integral promotion (C++ 4.5).
    SCS.Second = ICK_Integral_Promotion;
    FromType = ToType.getUnqualifiedType();
  } else if (IsFloatingPointPromotion(FromType, ToType)) {
    // Floating point promotion (C++ 4.6).
    SCS.Second = ICK_Floating_Promotion;
    FromType = ToType.getUnqualifiedType();
  } else if (IsComplexPromotion(FromType, ToType)) {
    // Complex promotion (Clang extension)
    SCS.Second = ICK_Complex_Promotion;
    FromType = ToType.getUnqualifiedType();
  } else if ((FromType->isIntegralType() || FromType->isEnumeralType()) &&
           (ToType->isIntegralType() && !ToType->isEnumeralType())) {
    // Integral conversions (C++ 4.7).
    // FIXME: isIntegralType shouldn't be true for enums in C++.
    SCS.Second = ICK_Integral_Conversion;
    FromType = ToType.getUnqualifiedType();
  } else if (FromType->isFloatingType() && ToType->isFloatingType()) {
    // Floating point conversions (C++ 4.8).
    SCS.Second = ICK_Floating_Conversion;
    FromType = ToType.getUnqualifiedType();
  } else if (FromType->isComplexType() && ToType->isComplexType()) {
    // Complex conversions (C99 6.3.1.6)
    SCS.Second = ICK_Complex_Conversion;
    FromType = ToType.getUnqualifiedType();
  } else if ((FromType->isFloatingType() &&
              ToType->isIntegralType() && (!ToType->isBooleanType() &&
                                           !ToType->isEnumeralType())) ||
             ((FromType->isIntegralType() || FromType->isEnumeralType()) &&
              ToType->isFloatingType())) {
    // Floating-integral conversions (C++ 4.9).
    // FIXME: isIntegralType shouldn't be true for enums in C++.
    SCS.Second = ICK_Floating_Integral;
    FromType = ToType.getUnqualifiedType();
  } else if ((FromType->isComplexType() && ToType->isArithmeticType()) ||
             (ToType->isComplexType() && FromType->isArithmeticType())) {
    // Complex-real conversions (C99 6.3.1.7)
    SCS.Second = ICK_Complex_Real;
    FromType = ToType.getUnqualifiedType();
  } else if (IsPointerConversion(From, FromType, ToType, InOverloadResolution,
                                 FromType, IncompatibleObjC)) {
    // Pointer conversions (C++ 4.10).
    SCS.Second = ICK_Pointer_Conversion;
    SCS.IncompatibleObjC = IncompatibleObjC;
  } else if (IsMemberPointerConversion(From, FromType, ToType, 
                                       InOverloadResolution, FromType)) {
    // Pointer to member conversions (4.11).
    SCS.Second = ICK_Pointer_Member;
  } else if (ToType->isBooleanType() &&
             (FromType->isArithmeticType() ||
              FromType->isEnumeralType() ||
              FromType->isAnyPointerType() ||
              FromType->isBlockPointerType() ||
              FromType->isMemberPointerType() ||
              FromType->isNullPtrType())) {
    // Boolean conversions (C++ 4.12).
    SCS.Second = ICK_Boolean_Conversion;
    FromType = Context.BoolTy;
  } else if (!getLangOptions().CPlusPlus &&
             Context.typesAreCompatible(ToType, FromType)) {
    // Compatible conversions (Clang extension for C function overloading)
    SCS.Second = ICK_Compatible_Conversion;
  } else if (IsNoReturnConversion(Context, FromType, ToType, FromType)) {
    // Treat a conversion that strips "noreturn" as an identity conversion.
    SCS.Second = ICK_NoReturn_Adjustment;
  } else {
    // No second conversion required.
    SCS.Second = ICK_Identity;
  }

  QualType CanonFrom;
  QualType CanonTo;
  // The third conversion can be a qualification conversion (C++ 4p1).
  if (IsQualificationConversion(FromType, ToType)) {
    SCS.Third = ICK_Qualification;
    FromType = ToType;
    CanonFrom = Context.getCanonicalType(FromType);
    CanonTo = Context.getCanonicalType(ToType);
  } else {
    // No conversion required
    SCS.Third = ICK_Identity;

    // C++ [over.best.ics]p6:
    //   [...] Any difference in top-level cv-qualification is
    //   subsumed by the initialization itself and does not constitute
    //   a conversion. [...]
    CanonFrom = Context.getCanonicalType(FromType);
    CanonTo = Context.getCanonicalType(ToType);
    if (CanonFrom.getLocalUnqualifiedType() 
                                       == CanonTo.getLocalUnqualifiedType() &&
        CanonFrom.getLocalCVRQualifiers() != CanonTo.getLocalCVRQualifiers()) {
      FromType = ToType;
      CanonFrom = CanonTo;
    }
  }

  // If we have not converted the argument type to the parameter type,
  // this is a bad conversion sequence.
  if (CanonFrom != CanonTo)
    return false;

  SCS.ToTypePtr = FromType.getAsOpaquePtr();
  return true;
}

/// IsIntegralPromotion - Determines whether the conversion from the
/// expression From (whose potentially-adjusted type is FromType) to
/// ToType is an integral promotion (C++ 4.5). If so, returns true and
/// sets PromotedType to the promoted type.
bool Sema::IsIntegralPromotion(Expr *From, QualType FromType, QualType ToType) {
  const BuiltinType *To = ToType->getAs<BuiltinType>();
  // All integers are built-in.
  if (!To) {
    return false;
  }

  // An rvalue of type char, signed char, unsigned char, short int, or
  // unsigned short int can be converted to an rvalue of type int if
  // int can represent all the values of the source type; otherwise,
  // the source rvalue can be converted to an rvalue of type unsigned
  // int (C++ 4.5p1).
  if (FromType->isPromotableIntegerType() && !FromType->isBooleanType()) {
    if (// We can promote any signed, promotable integer type to an int
        (FromType->isSignedIntegerType() ||
         // We can promote any unsigned integer type whose size is
         // less than int to an int.
         (!FromType->isSignedIntegerType() &&
          Context.getTypeSize(FromType) < Context.getTypeSize(ToType)))) {
      return To->getKind() == BuiltinType::Int;
    }

    return To->getKind() == BuiltinType::UInt;
  }

  // An rvalue of type wchar_t (3.9.1) or an enumeration type (7.2)
  // can be converted to an rvalue of the first of the following types
  // that can represent all the values of its underlying type: int,
  // unsigned int, long, or unsigned long (C++ 4.5p2).

  // We pre-calculate the promotion type for enum types.
  if (const EnumType *FromEnumType = FromType->getAs<EnumType>())
    if (ToType->isIntegerType())
      return Context.hasSameUnqualifiedType(ToType,
                                FromEnumType->getDecl()->getPromotionType());

  if (FromType->isWideCharType() && ToType->isIntegerType()) {
    // Determine whether the type we're converting from is signed or
    // unsigned.
    bool FromIsSigned;
    uint64_t FromSize = Context.getTypeSize(FromType);
    
    // FIXME: Is wchar_t signed or unsigned? We assume it's signed for now.
    FromIsSigned = true;

    // The types we'll try to promote to, in the appropriate
    // order. Try each of these types.
    QualType PromoteTypes[6] = {
      Context.IntTy, Context.UnsignedIntTy,
      Context.LongTy, Context.UnsignedLongTy ,
      Context.LongLongTy, Context.UnsignedLongLongTy
    };
    for (int Idx = 0; Idx < 6; ++Idx) {
      uint64_t ToSize = Context.getTypeSize(PromoteTypes[Idx]);
      if (FromSize < ToSize ||
          (FromSize == ToSize &&
           FromIsSigned == PromoteTypes[Idx]->isSignedIntegerType())) {
        // We found the type that we can promote to. If this is the
        // type we wanted, we have a promotion. Otherwise, no
        // promotion.
        return Context.hasSameUnqualifiedType(ToType, PromoteTypes[Idx]);
      }
    }
  }

  // An rvalue for an integral bit-field (9.6) can be converted to an
  // rvalue of type int if int can represent all the values of the
  // bit-field; otherwise, it can be converted to unsigned int if
  // unsigned int can represent all the values of the bit-field. If
  // the bit-field is larger yet, no integral promotion applies to
  // it. If the bit-field has an enumerated type, it is treated as any
  // other value of that type for promotion purposes (C++ 4.5p3).
  // FIXME: We should delay checking of bit-fields until we actually perform the
  // conversion.
  using llvm::APSInt;
  if (From)
    if (FieldDecl *MemberDecl = From->getBitField()) {
      APSInt BitWidth;
      if (FromType->isIntegralType() && !FromType->isEnumeralType() &&
          MemberDecl->getBitWidth()->isIntegerConstantExpr(BitWidth, Context)) {
        APSInt ToSize(BitWidth.getBitWidth(), BitWidth.isUnsigned());
        ToSize = Context.getTypeSize(ToType);

        // Are we promoting to an int from a bitfield that fits in an int?
        if (BitWidth < ToSize ||
            (FromType->isSignedIntegerType() && BitWidth <= ToSize)) {
          return To->getKind() == BuiltinType::Int;
        }

        // Are we promoting to an unsigned int from an unsigned bitfield
        // that fits into an unsigned int?
        if (FromType->isUnsignedIntegerType() && BitWidth <= ToSize) {
          return To->getKind() == BuiltinType::UInt;
        }

        return false;
      }
    }

  // An rvalue of type bool can be converted to an rvalue of type int,
  // with false becoming zero and true becoming one (C++ 4.5p4).
  if (FromType->isBooleanType() && To->getKind() == BuiltinType::Int) {
    return true;
  }

  return false;
}

/// IsFloatingPointPromotion - Determines whether the conversion from
/// FromType to ToType is a floating point promotion (C++ 4.6). If so,
/// returns true and sets PromotedType to the promoted type.
bool Sema::IsFloatingPointPromotion(QualType FromType, QualType ToType) {
  /// An rvalue of type float can be converted to an rvalue of type
  /// double. (C++ 4.6p1).
  if (const BuiltinType *FromBuiltin = FromType->getAs<BuiltinType>())
    if (const BuiltinType *ToBuiltin = ToType->getAs<BuiltinType>()) {
      if (FromBuiltin->getKind() == BuiltinType::Float &&
          ToBuiltin->getKind() == BuiltinType::Double)
        return true;

      // C99 6.3.1.5p1:
      //   When a float is promoted to double or long double, or a
      //   double is promoted to long double [...].
      if (!getLangOptions().CPlusPlus &&
          (FromBuiltin->getKind() == BuiltinType::Float ||
           FromBuiltin->getKind() == BuiltinType::Double) &&
          (ToBuiltin->getKind() == BuiltinType::LongDouble))
        return true;
    }

  return false;
}

/// \brief Determine if a conversion is a complex promotion.
///
/// A complex promotion is defined as a complex -> complex conversion
/// where the conversion between the underlying real types is a
/// floating-point or integral promotion.
bool Sema::IsComplexPromotion(QualType FromType, QualType ToType) {
  const ComplexType *FromComplex = FromType->getAs<ComplexType>();
  if (!FromComplex)
    return false;

  const ComplexType *ToComplex = ToType->getAs<ComplexType>();
  if (!ToComplex)
    return false;

  return IsFloatingPointPromotion(FromComplex->getElementType(),
                                  ToComplex->getElementType()) ||
    IsIntegralPromotion(0, FromComplex->getElementType(),
                        ToComplex->getElementType());
}

/// BuildSimilarlyQualifiedPointerType - In a pointer conversion from
/// the pointer type FromPtr to a pointer to type ToPointee, with the
/// same type qualifiers as FromPtr has on its pointee type. ToType,
/// if non-empty, will be a pointer to ToType that may or may not have
/// the right set of qualifiers on its pointee.
static QualType
BuildSimilarlyQualifiedPointerType(const PointerType *FromPtr,
                                   QualType ToPointee, QualType ToType,
                                   ASTContext &Context) {
  QualType CanonFromPointee = Context.getCanonicalType(FromPtr->getPointeeType());
  QualType CanonToPointee = Context.getCanonicalType(ToPointee);
  Qualifiers Quals = CanonFromPointee.getQualifiers();

  // Exact qualifier match -> return the pointer type we're converting to.
  if (CanonToPointee.getLocalQualifiers() == Quals) {
    // ToType is exactly what we need. Return it.
    if (!ToType.isNull())
      return ToType;

    // Build a pointer to ToPointee. It has the right qualifiers
    // already.
    return Context.getPointerType(ToPointee);
  }

  // Just build a canonical type that has the right qualifiers.
  return Context.getPointerType(
         Context.getQualifiedType(CanonToPointee.getLocalUnqualifiedType(), 
                                  Quals));
}

/// BuildSimilarlyQualifiedObjCObjectPointerType - In a pointer conversion from
/// the FromType, which is an objective-c pointer, to ToType, which may or may
/// not have the right set of qualifiers.
static QualType
BuildSimilarlyQualifiedObjCObjectPointerType(QualType FromType,
                                             QualType ToType,
                                             ASTContext &Context) {
  QualType CanonFromType = Context.getCanonicalType(FromType);
  QualType CanonToType = Context.getCanonicalType(ToType);
  Qualifiers Quals = CanonFromType.getQualifiers();
    
  // Exact qualifier match -> return the pointer type we're converting to.
  if (CanonToType.getLocalQualifiers() == Quals)
    return ToType;
  
  // Just build a canonical type that has the right qualifiers.
  return Context.getQualifiedType(CanonToType.getLocalUnqualifiedType(), Quals);
}
  
static bool isNullPointerConstantForConversion(Expr *Expr,
                                               bool InOverloadResolution,
                                               ASTContext &Context) {
  // Handle value-dependent integral null pointer constants correctly.
  // http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_active.html#903
  if (Expr->isValueDependent() && !Expr->isTypeDependent() &&
      Expr->getType()->isIntegralType())
    return !InOverloadResolution;

  return Expr->isNullPointerConstant(Context,
                    InOverloadResolution? Expr::NPC_ValueDependentIsNotNull
                                        : Expr::NPC_ValueDependentIsNull);
}

/// IsPointerConversion - Determines whether the conversion of the
/// expression From, which has the (possibly adjusted) type FromType,
/// can be converted to the type ToType via a pointer conversion (C++
/// 4.10). If so, returns true and places the converted type (that
/// might differ from ToType in its cv-qualifiers at some level) into
/// ConvertedType.
///
/// This routine also supports conversions to and from block pointers
/// and conversions with Objective-C's 'id', 'id<protocols...>', and
/// pointers to interfaces. FIXME: Once we've determined the
/// appropriate overloading rules for Objective-C, we may want to
/// split the Objective-C checks into a different routine; however,
/// GCC seems to consider all of these conversions to be pointer
/// conversions, so for now they live here. IncompatibleObjC will be
/// set if the conversion is an allowed Objective-C conversion that
/// should result in a warning.
bool Sema::IsPointerConversion(Expr *From, QualType FromType, QualType ToType,
                               bool InOverloadResolution,
                               QualType& ConvertedType,
                               bool &IncompatibleObjC) {
  IncompatibleObjC = false;
  if (isObjCPointerConversion(FromType, ToType, ConvertedType, IncompatibleObjC))
    return true;

  // Conversion from a null pointer constant to any Objective-C pointer type.
  if (ToType->isObjCObjectPointerType() &&
      isNullPointerConstantForConversion(From, InOverloadResolution, Context)) {
    ConvertedType = ToType;
    return true;
  }

  // Blocks: Block pointers can be converted to void*.
  if (FromType->isBlockPointerType() && ToType->isPointerType() &&
      ToType->getAs<PointerType>()->getPointeeType()->isVoidType()) {
    ConvertedType = ToType;
    return true;
  }
  // Blocks: A null pointer constant can be converted to a block
  // pointer type.
  if (ToType->isBlockPointerType() &&
      isNullPointerConstantForConversion(From, InOverloadResolution, Context)) {
    ConvertedType = ToType;
    return true;
  }

  // If the left-hand-side is nullptr_t, the right side can be a null
  // pointer constant.
  if (ToType->isNullPtrType() &&
      isNullPointerConstantForConversion(From, InOverloadResolution, Context)) {
    ConvertedType = ToType;
    return true;
  }

  const PointerType* ToTypePtr = ToType->getAs<PointerType>();
  if (!ToTypePtr)
    return false;

  // A null pointer constant can be converted to a pointer type (C++ 4.10p1).
  if (isNullPointerConstantForConversion(From, InOverloadResolution, Context)) {
    ConvertedType = ToType;
    return true;
  }

  // Beyond this point, both types need to be pointers 
  // , including objective-c pointers.
  QualType ToPointeeType = ToTypePtr->getPointeeType();
  if (FromType->isObjCObjectPointerType() && ToPointeeType->isVoidType()) {
    ConvertedType = BuildSimilarlyQualifiedObjCObjectPointerType(FromType,
                                                       ToType, Context);
    return true;
    
  }
  const PointerType *FromTypePtr = FromType->getAs<PointerType>();
  if (!FromTypePtr)
    return false;

  QualType FromPointeeType = FromTypePtr->getPointeeType();

  // An rvalue of type "pointer to cv T," where T is an object type,
  // can be converted to an rvalue of type "pointer to cv void" (C++
  // 4.10p2).
  if (FromPointeeType->isObjectType() && ToPointeeType->isVoidType()) {
    ConvertedType = BuildSimilarlyQualifiedPointerType(FromTypePtr,
                                                       ToPointeeType,
                                                       ToType, Context);
    return true;
  }

  // When we're overloading in C, we allow a special kind of pointer
  // conversion for compatible-but-not-identical pointee types.
  if (!getLangOptions().CPlusPlus &&
      Context.typesAreCompatible(FromPointeeType, ToPointeeType)) {
    ConvertedType = BuildSimilarlyQualifiedPointerType(FromTypePtr,
                                                       ToPointeeType,
                                                       ToType, Context);
    return true;
  }

  // C++ [conv.ptr]p3:
  //
  //   An rvalue of type "pointer to cv D," where D is a class type,
  //   can be converted to an rvalue of type "pointer to cv B," where
  //   B is a base class (clause 10) of D. If B is an inaccessible
  //   (clause 11) or ambiguous (10.2) base class of D, a program that
  //   necessitates this conversion is ill-formed. The result of the
  //   conversion is a pointer to the base class sub-object of the
  //   derived class object. The null pointer value is converted to
  //   the null pointer value of the destination type.
  //
  // Note that we do not check for ambiguity or inaccessibility
  // here. That is handled by CheckPointerConversion.
  if (getLangOptions().CPlusPlus &&
      FromPointeeType->isRecordType() && ToPointeeType->isRecordType() &&
      !RequireCompleteType(From->getLocStart(), FromPointeeType, PDiag()) &&
      IsDerivedFrom(FromPointeeType, ToPointeeType)) {
    ConvertedType = BuildSimilarlyQualifiedPointerType(FromTypePtr,
                                                       ToPointeeType,
                                                       ToType, Context);
    return true;
  }

  return false;
}

/// isObjCPointerConversion - Determines whether this is an
/// Objective-C pointer conversion. Subroutine of IsPointerConversion,
/// with the same arguments and return values.
bool Sema::isObjCPointerConversion(QualType FromType, QualType ToType,
                                   QualType& ConvertedType,
                                   bool &IncompatibleObjC) {
  if (!getLangOptions().ObjC1)
    return false;

  // First, we handle all conversions on ObjC object pointer types.
  const ObjCObjectPointerType* ToObjCPtr = ToType->getAs<ObjCObjectPointerType>();
  const ObjCObjectPointerType *FromObjCPtr =
    FromType->getAs<ObjCObjectPointerType>();

  if (ToObjCPtr && FromObjCPtr) {
    // Objective C++: We're able to convert between "id" or "Class" and a
    // pointer to any interface (in both directions).
    if (ToObjCPtr->isObjCBuiltinType() && FromObjCPtr->isObjCBuiltinType()) {
      ConvertedType = ToType;
      return true;
    }
    // Conversions with Objective-C's id<...>.
    if ((FromObjCPtr->isObjCQualifiedIdType() ||
         ToObjCPtr->isObjCQualifiedIdType()) &&
        Context.ObjCQualifiedIdTypesAreCompatible(ToType, FromType,
                                                  /*compare=*/false)) {
      ConvertedType = ToType;
      return true;
    }
    // Objective C++: We're able to convert from a pointer to an
    // interface to a pointer to a different interface.
    if (Context.canAssignObjCInterfaces(ToObjCPtr, FromObjCPtr)) {
      ConvertedType = ToType;
      return true;
    }

    if (Context.canAssignObjCInterfaces(FromObjCPtr, ToObjCPtr)) {
      // Okay: this is some kind of implicit downcast of Objective-C
      // interfaces, which is permitted. However, we're going to
      // complain about it.
      IncompatibleObjC = true;
      ConvertedType = FromType;
      return true;
    }
  }
  // Beyond this point, both types need to be C pointers or block pointers.
  QualType ToPointeeType;
  if (const PointerType *ToCPtr = ToType->getAs<PointerType>())
    ToPointeeType = ToCPtr->getPointeeType();
  else if (const BlockPointerType *ToBlockPtr = ToType->getAs<BlockPointerType>())
    ToPointeeType = ToBlockPtr->getPointeeType();
  else
    return false;

  QualType FromPointeeType;
  if (const PointerType *FromCPtr = FromType->getAs<PointerType>())
    FromPointeeType = FromCPtr->getPointeeType();
  else if (const BlockPointerType *FromBlockPtr = FromType->getAs<BlockPointerType>())
    FromPointeeType = FromBlockPtr->getPointeeType();
  else
    return false;

  // If we have pointers to pointers, recursively check whether this
  // is an Objective-C conversion.
  if (FromPointeeType->isPointerType() && ToPointeeType->isPointerType() &&
      isObjCPointerConversion(FromPointeeType, ToPointeeType, ConvertedType,
                              IncompatibleObjC)) {
    // We always complain about this conversion.
    IncompatibleObjC = true;
    ConvertedType = ToType;
    return true;
  }
  // If we have pointers to functions or blocks, check whether the only
  // differences in the argument and result types are in Objective-C
  // pointer conversions. If so, we permit the conversion (but
  // complain about it).
  const FunctionProtoType *FromFunctionType
    = FromPointeeType->getAs<FunctionProtoType>();
  const FunctionProtoType *ToFunctionType
    = ToPointeeType->getAs<FunctionProtoType>();
  if (FromFunctionType && ToFunctionType) {
    // If the function types are exactly the same, this isn't an
    // Objective-C pointer conversion.
    if (Context.getCanonicalType(FromPointeeType)
          == Context.getCanonicalType(ToPointeeType))
      return false;

    // Perform the quick checks that will tell us whether these
    // function types are obviously different.
    if (FromFunctionType->getNumArgs() != ToFunctionType->getNumArgs() ||
        FromFunctionType->isVariadic() != ToFunctionType->isVariadic() ||
        FromFunctionType->getTypeQuals() != ToFunctionType->getTypeQuals())
      return false;

    bool HasObjCConversion = false;
    if (Context.getCanonicalType(FromFunctionType->getResultType())
          == Context.getCanonicalType(ToFunctionType->getResultType())) {
      // Okay, the types match exactly. Nothing to do.
    } else if (isObjCPointerConversion(FromFunctionType->getResultType(),
                                       ToFunctionType->getResultType(),
                                       ConvertedType, IncompatibleObjC)) {
      // Okay, we have an Objective-C pointer conversion.
      HasObjCConversion = true;
    } else {
      // Function types are too different. Abort.
      return false;
    }

    // Check argument types.
    for (unsigned ArgIdx = 0, NumArgs = FromFunctionType->getNumArgs();
         ArgIdx != NumArgs; ++ArgIdx) {
      QualType FromArgType = FromFunctionType->getArgType(ArgIdx);
      QualType ToArgType = ToFunctionType->getArgType(ArgIdx);
      if (Context.getCanonicalType(FromArgType)
            == Context.getCanonicalType(ToArgType)) {
        // Okay, the types match exactly. Nothing to do.
      } else if (isObjCPointerConversion(FromArgType, ToArgType,
                                         ConvertedType, IncompatibleObjC)) {
        // Okay, we have an Objective-C pointer conversion.
        HasObjCConversion = true;
      } else {
        // Argument types are too different. Abort.
        return false;
      }
    }

    if (HasObjCConversion) {
      // We had an Objective-C conversion. Allow this pointer
      // conversion, but complain about it.
      ConvertedType = ToType;
      IncompatibleObjC = true;
      return true;
    }
  }

  return false;
}

/// CheckPointerConversion - Check the pointer conversion from the
/// expression From to the type ToType. This routine checks for
/// ambiguous or inaccessible derived-to-base pointer
/// conversions for which IsPointerConversion has already returned
/// true. It returns true and produces a diagnostic if there was an
/// error, or returns false otherwise.
bool Sema::CheckPointerConversion(Expr *From, QualType ToType,
                                  CastExpr::CastKind &Kind,
                                  bool IgnoreBaseAccess) {
  QualType FromType = From->getType();

  if (const PointerType *FromPtrType = FromType->getAs<PointerType>())
    if (const PointerType *ToPtrType = ToType->getAs<PointerType>()) {
      QualType FromPointeeType = FromPtrType->getPointeeType(),
               ToPointeeType   = ToPtrType->getPointeeType();

      if (FromPointeeType->isRecordType() &&
          ToPointeeType->isRecordType()) {
        // We must have a derived-to-base conversion. Check an
        // ambiguous or inaccessible conversion.
        if (CheckDerivedToBaseConversion(FromPointeeType, ToPointeeType,
                                         From->getExprLoc(),
                                         From->getSourceRange(),
                                         IgnoreBaseAccess))
          return true;
        
        // The conversion was successful.
        Kind = CastExpr::CK_DerivedToBase;
      }
    }
  if (const ObjCObjectPointerType *FromPtrType =
        FromType->getAs<ObjCObjectPointerType>())
    if (const ObjCObjectPointerType *ToPtrType =
          ToType->getAs<ObjCObjectPointerType>()) {
      // Objective-C++ conversions are always okay.
      // FIXME: We should have a different class of conversions for the
      // Objective-C++ implicit conversions.
      if (FromPtrType->isObjCBuiltinType() || ToPtrType->isObjCBuiltinType())
        return false;

  }
  return false;
}

/// IsMemberPointerConversion - Determines whether the conversion of the
/// expression From, which has the (possibly adjusted) type FromType, can be
/// converted to the type ToType via a member pointer conversion (C++ 4.11).
/// If so, returns true and places the converted type (that might differ from
/// ToType in its cv-qualifiers at some level) into ConvertedType.
bool Sema::IsMemberPointerConversion(Expr *From, QualType FromType,
                                     QualType ToType, 
                                     bool InOverloadResolution,
                                     QualType &ConvertedType) {
  const MemberPointerType *ToTypePtr = ToType->getAs<MemberPointerType>();
  if (!ToTypePtr)
    return false;

  // A null pointer constant can be converted to a member pointer (C++ 4.11p1)
  if (From->isNullPointerConstant(Context,
                    InOverloadResolution? Expr::NPC_ValueDependentIsNotNull
                                        : Expr::NPC_ValueDependentIsNull)) {
    ConvertedType = ToType;
    return true;
  }

  // Otherwise, both types have to be member pointers.
  const MemberPointerType *FromTypePtr = FromType->getAs<MemberPointerType>();
  if (!FromTypePtr)
    return false;

  // A pointer to member of B can be converted to a pointer to member of D,
  // where D is derived from B (C++ 4.11p2).
  QualType FromClass(FromTypePtr->getClass(), 0);
  QualType ToClass(ToTypePtr->getClass(), 0);
  // FIXME: What happens when these are dependent? Is this function even called?

  if (IsDerivedFrom(ToClass, FromClass)) {
    ConvertedType = Context.getMemberPointerType(FromTypePtr->getPointeeType(),
                                                 ToClass.getTypePtr());
    return true;
  }

  return false;
}
  
/// CheckMemberPointerConversion - Check the member pointer conversion from the
/// expression From to the type ToType. This routine checks for ambiguous or
/// virtual (FIXME: or inaccessible) base-to-derived member pointer conversions
/// for which IsMemberPointerConversion has already returned true. It returns
/// true and produces a diagnostic if there was an error, or returns false
/// otherwise.
bool Sema::CheckMemberPointerConversion(Expr *From, QualType ToType,
                                        CastExpr::CastKind &Kind,
                                        bool IgnoreBaseAccess) {
  (void)IgnoreBaseAccess;
  QualType FromType = From->getType();
  const MemberPointerType *FromPtrType = FromType->getAs<MemberPointerType>();
  if (!FromPtrType) {
    // This must be a null pointer to member pointer conversion
    assert(From->isNullPointerConstant(Context, 
                                       Expr::NPC_ValueDependentIsNull) &&
           "Expr must be null pointer constant!");
    Kind = CastExpr::CK_NullToMemberPointer;
    return false;
  }

  const MemberPointerType *ToPtrType = ToType->getAs<MemberPointerType>();
  assert(ToPtrType && "No member pointer cast has a target type "
                      "that is not a member pointer.");

  QualType FromClass = QualType(FromPtrType->getClass(), 0);
  QualType ToClass   = QualType(ToPtrType->getClass(), 0);

  // FIXME: What about dependent types?
  assert(FromClass->isRecordType() && "Pointer into non-class.");
  assert(ToClass->isRecordType() && "Pointer into non-class.");

  CXXBasePaths Paths(/*FindAmbiguities=*/true, /*RecordPaths=*/false,
                     /*DetectVirtual=*/true);
  bool DerivationOkay = IsDerivedFrom(ToClass, FromClass, Paths);
  assert(DerivationOkay &&
         "Should not have been called if derivation isn't OK.");
  (void)DerivationOkay;

  if (Paths.isAmbiguous(Context.getCanonicalType(FromClass).
                                  getUnqualifiedType())) {
    // Derivation is ambiguous. Redo the check to find the exact paths.
    Paths.clear();
    Paths.setRecordingPaths(true);
    bool StillOkay = IsDerivedFrom(ToClass, FromClass, Paths);
    assert(StillOkay && "Derivation changed due to quantum fluctuation.");
    (void)StillOkay;

    std::string PathDisplayStr = getAmbiguousPathsDisplayString(Paths);
    Diag(From->getExprLoc(), diag::err_ambiguous_memptr_conv)
      << 0 << FromClass << ToClass << PathDisplayStr << From->getSourceRange();
    return true;
  }

  if (const RecordType *VBase = Paths.getDetectedVirtual()) {
    Diag(From->getExprLoc(), diag::err_memptr_conv_via_virtual)
      << FromClass << ToClass << QualType(VBase, 0)
      << From->getSourceRange();
    return true;
  }

  // Must be a base to derived member conversion.
  Kind = CastExpr::CK_BaseToDerivedMemberPointer;
  return false;
}

/// IsQualificationConversion - Determines whether the conversion from
/// an rvalue of type FromType to ToType is a qualification conversion
/// (C++ 4.4).
bool
Sema::IsQualificationConversion(QualType FromType, QualType ToType) {
  FromType = Context.getCanonicalType(FromType);
  ToType = Context.getCanonicalType(ToType);

  // If FromType and ToType are the same type, this is not a
  // qualification conversion.
  if (FromType == ToType)
    return false;

  // (C++ 4.4p4):
  //   A conversion can add cv-qualifiers at levels other than the first
  //   in multi-level pointers, subject to the following rules: [...]
  bool PreviousToQualsIncludeConst = true;
  bool UnwrappedAnyPointer = false;
  while (UnwrapSimilarPointerTypes(FromType, ToType)) {
    // Within each iteration of the loop, we check the qualifiers to
    // determine if this still looks like a qualification
    // conversion. Then, if all is well, we unwrap one more level of
    // pointers or pointers-to-members and do it all again
    // until there are no more pointers or pointers-to-members left to
    // unwrap.
    UnwrappedAnyPointer = true;

    //   -- for every j > 0, if const is in cv 1,j then const is in cv
    //      2,j, and similarly for volatile.
    if (!ToType.isAtLeastAsQualifiedAs(FromType))
      return false;

    //   -- if the cv 1,j and cv 2,j are different, then const is in
    //      every cv for 0 < k < j.
    if (FromType.getCVRQualifiers() != ToType.getCVRQualifiers()
        && !PreviousToQualsIncludeConst)
      return false;

    // Keep track of whether all prior cv-qualifiers in the "to" type
    // include const.
    PreviousToQualsIncludeConst
      = PreviousToQualsIncludeConst && ToType.isConstQualified();
  }

  // We are left with FromType and ToType being the pointee types
  // after unwrapping the original FromType and ToType the same number
  // of types. If we unwrapped any pointers, and if FromType and
  // ToType have the same unqualified type (since we checked
  // qualifiers above), then this is a qualification conversion.
  return UnwrappedAnyPointer && Context.hasSameUnqualifiedType(FromType,ToType);
}

/// Determines whether there is a user-defined conversion sequence
/// (C++ [over.ics.user]) that converts expression From to the type
/// ToType. If such a conversion exists, User will contain the
/// user-defined conversion sequence that performs such a conversion
/// and this routine will return true. Otherwise, this routine returns
/// false and User is unspecified.
///
/// \param AllowConversionFunctions true if the conversion should
/// consider conversion functions at all. If false, only constructors
/// will be considered.
///
/// \param AllowExplicit  true if the conversion should consider C++0x
/// "explicit" conversion functions as well as non-explicit conversion
/// functions (C++0x [class.conv.fct]p2).
///
/// \param ForceRValue  true if the expression should be treated as an rvalue
/// for overload resolution.
/// \param UserCast true if looking for user defined conversion for a static
/// cast.
OverloadingResult Sema::IsUserDefinedConversion(Expr *From, QualType ToType,
                                          UserDefinedConversionSequence& User,
                                            OverloadCandidateSet& CandidateSet,
                                                bool AllowConversionFunctions,
                                                bool AllowExplicit, 
                                                bool ForceRValue,
                                                bool UserCast) {
  if (const RecordType *ToRecordType = ToType->getAs<RecordType>()) {
    if (RequireCompleteType(From->getLocStart(), ToType, PDiag())) {
      // We're not going to find any constructors.
    } else if (CXXRecordDecl *ToRecordDecl
                 = dyn_cast<CXXRecordDecl>(ToRecordType->getDecl())) {
      // C++ [over.match.ctor]p1:
      //   When objects of class type are direct-initialized (8.5), or
      //   copy-initialized from an expression of the same or a
      //   derived class type (8.5), overload resolution selects the
      //   constructor. [...] For copy-initialization, the candidate
      //   functions are all the converting constructors (12.3.1) of
      //   that class. The argument list is the expression-list within
      //   the parentheses of the initializer.
      bool SuppressUserConversions = !UserCast;
      if (Context.hasSameUnqualifiedType(ToType, From->getType()) ||
          IsDerivedFrom(From->getType(), ToType)) {
        SuppressUserConversions = false;
        AllowConversionFunctions = false;
      }
          
      DeclarationName ConstructorName
        = Context.DeclarationNames.getCXXConstructorName(
                          Context.getCanonicalType(ToType).getUnqualifiedType());
      DeclContext::lookup_iterator Con, ConEnd;
      for (llvm::tie(Con, ConEnd)
             = ToRecordDecl->lookup(ConstructorName);
           Con != ConEnd; ++Con) {
        // Find the constructor (which may be a template).
        CXXConstructorDecl *Constructor = 0;
        FunctionTemplateDecl *ConstructorTmpl
          = dyn_cast<FunctionTemplateDecl>(*Con);
        if (ConstructorTmpl)
          Constructor
            = cast<CXXConstructorDecl>(ConstructorTmpl->getTemplatedDecl());
        else
          Constructor = cast<CXXConstructorDecl>(*Con);
        
        if (!Constructor->isInvalidDecl() &&
            Constructor->isConvertingConstructor(AllowExplicit)) {
          if (ConstructorTmpl)
            AddTemplateOverloadCandidate(ConstructorTmpl, /*ExplicitArgs*/ 0,
                                         &From, 1, CandidateSet, 
                                         SuppressUserConversions, ForceRValue);
          else
            // Allow one user-defined conversion when user specifies a
            // From->ToType conversion via an static cast (c-style, etc).
            AddOverloadCandidate(Constructor, &From, 1, CandidateSet,
                                 SuppressUserConversions, ForceRValue);
        }
      }
    }
  }

  if (!AllowConversionFunctions) {
    // Don't allow any conversion functions to enter the overload set.
  } else if (RequireCompleteType(From->getLocStart(), From->getType(),
                                 PDiag(0)
                                   << From->getSourceRange())) {
    // No conversion functions from incomplete types.
  } else if (const RecordType *FromRecordType
               = From->getType()->getAs<RecordType>()) {
    if (CXXRecordDecl *FromRecordDecl
         = dyn_cast<CXXRecordDecl>(FromRecordType->getDecl())) {
      // Add all of the conversion functions as candidates.
      const UnresolvedSet *Conversions
        = FromRecordDecl->getVisibleConversionFunctions();
      for (UnresolvedSet::iterator I = Conversions->begin(),
             E = Conversions->end(); I != E; ++I) {
        NamedDecl *D = *I;
        CXXRecordDecl *ActingContext = cast<CXXRecordDecl>(D->getDeclContext());
        if (isa<UsingShadowDecl>(D))
          D = cast<UsingShadowDecl>(D)->getTargetDecl();

        CXXConversionDecl *Conv;
        FunctionTemplateDecl *ConvTemplate;
        if ((ConvTemplate = dyn_cast<FunctionTemplateDecl>(*I)))
          Conv = dyn_cast<CXXConversionDecl>(ConvTemplate->getTemplatedDecl());
        else
          Conv = dyn_cast<CXXConversionDecl>(*I);

        if (AllowExplicit || !Conv->isExplicit()) {
          if (ConvTemplate)
            AddTemplateConversionCandidate(ConvTemplate, ActingContext,
                                           From, ToType, CandidateSet);
          else
            AddConversionCandidate(Conv, ActingContext, From, ToType,
                                   CandidateSet);
        }
      }
    }
  }

  OverloadCandidateSet::iterator Best;
  switch (BestViableFunction(CandidateSet, From->getLocStart(), Best)) {
    case OR_Success:
      // Record the standard conversion we used and the conversion function.
      if (CXXConstructorDecl *Constructor
            = dyn_cast<CXXConstructorDecl>(Best->Function)) {
        // C++ [over.ics.user]p1:
        //   If the user-defined conversion is specified by a
        //   constructor (12.3.1), the initial standard conversion
        //   sequence converts the source type to the type required by
        //   the argument of the constructor.
        //
        QualType ThisType = Constructor->getThisType(Context);
        if (Best->Conversions[0].ConversionKind == 
            ImplicitConversionSequence::EllipsisConversion)
          User.EllipsisConversion = true;
        else {
          User.Before = Best->Conversions[0].Standard;
          User.EllipsisConversion = false;
        }
        User.ConversionFunction = Constructor;
        User.After.setAsIdentityConversion();
        User.After.FromTypePtr
          = ThisType->getAs<PointerType>()->getPointeeType().getAsOpaquePtr();
        User.After.ToTypePtr = ToType.getAsOpaquePtr();
        return OR_Success;
      } else if (CXXConversionDecl *Conversion
                   = dyn_cast<CXXConversionDecl>(Best->Function)) {
        // C++ [over.ics.user]p1:
        //
        //   [...] If the user-defined conversion is specified by a
        //   conversion function (12.3.2), the initial standard
        //   conversion sequence converts the source type to the
        //   implicit object parameter of the conversion function.
        User.Before = Best->Conversions[0].Standard;
        User.ConversionFunction = Conversion;
        User.EllipsisConversion = false;

        // C++ [over.ics.user]p2:
        //   The second standard conversion sequence converts the
        //   result of the user-defined conversion to the target type
        //   for the sequence. Since an implicit conversion sequence
        //   is an initialization, the special rules for
        //   initialization by user-defined conversion apply when
        //   selecting the best user-defined conversion for a
        //   user-defined conversion sequence (see 13.3.3 and
        //   13.3.3.1).
        User.After = Best->FinalConversion;
        return OR_Success;
      } else {
        assert(false && "Not a constructor or conversion function?");
        return OR_No_Viable_Function;
      }

    case OR_No_Viable_Function:
      return OR_No_Viable_Function;
    case OR_Deleted:
      // No conversion here! We're done.
      return OR_Deleted;

    case OR_Ambiguous:
      return OR_Ambiguous;
    }

  return OR_No_Viable_Function;
}
  
bool
Sema::DiagnoseMultipleUserDefinedConversion(Expr *From, QualType ToType) {
  ImplicitConversionSequence ICS;
  OverloadCandidateSet CandidateSet;
  OverloadingResult OvResult = 
    IsUserDefinedConversion(From, ToType, ICS.UserDefined,
                            CandidateSet, true, false, false);
  if (OvResult == OR_Ambiguous)
    Diag(From->getSourceRange().getBegin(),
         diag::err_typecheck_ambiguous_condition)
          << From->getType() << ToType << From->getSourceRange();
  else if (OvResult == OR_No_Viable_Function && !CandidateSet.empty())
    Diag(From->getSourceRange().getBegin(),
         diag::err_typecheck_nonviable_condition)
    << From->getType() << ToType << From->getSourceRange();
  else
    return false;
  PrintOverloadCandidates(CandidateSet, /*OnlyViable=*/false);
  return true;  
}

/// CompareImplicitConversionSequences - Compare two implicit
/// conversion sequences to determine whether one is better than the
/// other or if they are indistinguishable (C++ 13.3.3.2).
ImplicitConversionSequence::CompareKind
Sema::CompareImplicitConversionSequences(const ImplicitConversionSequence& ICS1,
                                         const ImplicitConversionSequence& ICS2)
{
  // (C++ 13.3.3.2p2): When comparing the basic forms of implicit
  // conversion sequences (as defined in 13.3.3.1)
  //   -- a standard conversion sequence (13.3.3.1.1) is a better
  //      conversion sequence than a user-defined conversion sequence or
  //      an ellipsis conversion sequence, and
  //   -- a user-defined conversion sequence (13.3.3.1.2) is a better
  //      conversion sequence than an ellipsis conversion sequence
  //      (13.3.3.1.3).
  //
  if (ICS1.ConversionKind < ICS2.ConversionKind)
    return ImplicitConversionSequence::Better;
  else if (ICS2.ConversionKind < ICS1.ConversionKind)
    return ImplicitConversionSequence::Worse;

  // Two implicit conversion sequences of the same form are
  // indistinguishable conversion sequences unless one of the
  // following rules apply: (C++ 13.3.3.2p3):
  if (ICS1.ConversionKind == ImplicitConversionSequence::StandardConversion)
    return CompareStandardConversionSequences(ICS1.Standard, ICS2.Standard);
  else if (ICS1.ConversionKind ==
             ImplicitConversionSequence::UserDefinedConversion) {
    // User-defined conversion sequence U1 is a better conversion
    // sequence than another user-defined conversion sequence U2 if
    // they contain the same user-defined conversion function or
    // constructor and if the second standard conversion sequence of
    // U1 is better than the second standard conversion sequence of
    // U2 (C++ 13.3.3.2p3).
    if (ICS1.UserDefined.ConversionFunction ==
          ICS2.UserDefined.ConversionFunction)
      return CompareStandardConversionSequences(ICS1.UserDefined.After,
                                                ICS2.UserDefined.After);
  }

  return ImplicitConversionSequence::Indistinguishable;
}

/// CompareStandardConversionSequences - Compare two standard
/// conversion sequences to determine whether one is better than the
/// other or if they are indistinguishable (C++ 13.3.3.2p3).
ImplicitConversionSequence::CompareKind
Sema::CompareStandardConversionSequences(const StandardConversionSequence& SCS1,
                                         const StandardConversionSequence& SCS2)
{
  // Standard conversion sequence S1 is a better conversion sequence
  // than standard conversion sequence S2 if (C++ 13.3.3.2p3):

  //  -- S1 is a proper subsequence of S2 (comparing the conversion
  //     sequences in the canonical form defined by 13.3.3.1.1,
  //     excluding any Lvalue Transformation; the identity conversion
  //     sequence is considered to be a subsequence of any
  //     non-identity conversion sequence) or, if not that,
  if (SCS1.Second == SCS2.Second && SCS1.Third == SCS2.Third)
    // Neither is a proper subsequence of the other. Do nothing.
    ;
  else if ((SCS1.Second == ICK_Identity && SCS1.Third == SCS2.Third) ||
           (SCS1.Third == ICK_Identity && SCS1.Second == SCS2.Second) ||
           (SCS1.Second == ICK_Identity &&
            SCS1.Third == ICK_Identity))
    // SCS1 is a proper subsequence of SCS2.
    return ImplicitConversionSequence::Better;
  else if ((SCS2.Second == ICK_Identity && SCS2.Third == SCS1.Third) ||
           (SCS2.Third == ICK_Identity && SCS2.Second == SCS1.Second) ||
           (SCS2.Second == ICK_Identity &&
            SCS2.Third == ICK_Identity))
    // SCS2 is a proper subsequence of SCS1.
    return ImplicitConversionSequence::Worse;

  //  -- the rank of S1 is better than the rank of S2 (by the rules
  //     defined below), or, if not that,
  ImplicitConversionRank Rank1 = SCS1.getRank();
  ImplicitConversionRank Rank2 = SCS2.getRank();
  if (Rank1 < Rank2)
    return ImplicitConversionSequence::Better;
  else if (Rank2 < Rank1)
    return ImplicitConversionSequence::Worse;

  // (C++ 13.3.3.2p4): Two conversion sequences with the same rank
  // are indistinguishable unless one of the following rules
  // applies:

  //   A conversion that is not a conversion of a pointer, or
  //   pointer to member, to bool is better than another conversion
  //   that is such a conversion.
  if (SCS1.isPointerConversionToBool() != SCS2.isPointerConversionToBool())
    return SCS2.isPointerConversionToBool()
             ? ImplicitConversionSequence::Better
             : ImplicitConversionSequence::Worse;

  // C++ [over.ics.rank]p4b2:
  //
  //   If class B is derived directly or indirectly from class A,
  //   conversion of B* to A* is better than conversion of B* to
  //   void*, and conversion of A* to void* is better than conversion
  //   of B* to void*.
  bool SCS1ConvertsToVoid
    = SCS1.isPointerConversionToVoidPointer(Context);
  bool SCS2ConvertsToVoid
    = SCS2.isPointerConversionToVoidPointer(Context);
  if (SCS1ConvertsToVoid != SCS2ConvertsToVoid) {
    // Exactly one of the conversion sequences is a conversion to
    // a void pointer; it's the worse conversion.
    return SCS2ConvertsToVoid ? ImplicitConversionSequence::Better
                              : ImplicitConversionSequence::Worse;
  } else if (!SCS1ConvertsToVoid && !SCS2ConvertsToVoid) {
    // Neither conversion sequence converts to a void pointer; compare
    // their derived-to-base conversions.
    if (ImplicitConversionSequence::CompareKind DerivedCK
          = CompareDerivedToBaseConversions(SCS1, SCS2))
      return DerivedCK;
  } else if (SCS1ConvertsToVoid && SCS2ConvertsToVoid) {
    // Both conversion sequences are conversions to void
    // pointers. Compare the source types to determine if there's an
    // inheritance relationship in their sources.
    QualType FromType1 = QualType::getFromOpaquePtr(SCS1.FromTypePtr);
    QualType FromType2 = QualType::getFromOpaquePtr(SCS2.FromTypePtr);

    // Adjust the types we're converting from via the array-to-pointer
    // conversion, if we need to.
    if (SCS1.First == ICK_Array_To_Pointer)
      FromType1 = Context.getArrayDecayedType(FromType1);
    if (SCS2.First == ICK_Array_To_Pointer)
      FromType2 = Context.getArrayDecayedType(FromType2);

    QualType FromPointee1
      = FromType1->getAs<PointerType>()->getPointeeType().getUnqualifiedType();
    QualType FromPointee2
      = FromType2->getAs<PointerType>()->getPointeeType().getUnqualifiedType();

    if (IsDerivedFrom(FromPointee2, FromPointee1))
      return ImplicitConversionSequence::Better;
    else if (IsDerivedFrom(FromPointee1, FromPointee2))
      return ImplicitConversionSequence::Worse;

    // Objective-C++: If one interface is more specific than the
    // other, it is the better one.
    const ObjCInterfaceType* FromIface1 = FromPointee1->getAs<ObjCInterfaceType>();
    const ObjCInterfaceType* FromIface2 = FromPointee2->getAs<ObjCInterfaceType>();
    if (FromIface1 && FromIface1) {
      if (Context.canAssignObjCInterfaces(FromIface2, FromIface1))
        return ImplicitConversionSequence::Better;
      else if (Context.canAssignObjCInterfaces(FromIface1, FromIface2))
        return ImplicitConversionSequence::Worse;
    }
  }

  // Compare based on qualification conversions (C++ 13.3.3.2p3,
  // bullet 3).
  if (ImplicitConversionSequence::CompareKind QualCK
        = CompareQualificationConversions(SCS1, SCS2))
    return QualCK;

  if (SCS1.ReferenceBinding && SCS2.ReferenceBinding) {
    // C++0x [over.ics.rank]p3b4:
    //   -- S1 and S2 are reference bindings (8.5.3) and neither refers to an
    //      implicit object parameter of a non-static member function declared
    //      without a ref-qualifier, and S1 binds an rvalue reference to an
    //      rvalue and S2 binds an lvalue reference.
    // FIXME: We don't know if we're dealing with the implicit object parameter,
    // or if the member function in this case has a ref qualifier.
    // (Of course, we don't have ref qualifiers yet.)
    if (SCS1.RRefBinding != SCS2.RRefBinding)
      return SCS1.RRefBinding ? ImplicitConversionSequence::Better
                              : ImplicitConversionSequence::Worse;

    // C++ [over.ics.rank]p3b4:
    //   -- S1 and S2 are reference bindings (8.5.3), and the types to
    //      which the references refer are the same type except for
    //      top-level cv-qualifiers, and the type to which the reference
    //      initialized by S2 refers is more cv-qualified than the type
    //      to which the reference initialized by S1 refers.
    QualType T1 = QualType::getFromOpaquePtr(SCS1.ToTypePtr);
    QualType T2 = QualType::getFromOpaquePtr(SCS2.ToTypePtr);
    T1 = Context.getCanonicalType(T1);
    T2 = Context.getCanonicalType(T2);
    Qualifiers T1Quals, T2Quals;
    QualType UnqualT1 = Context.getUnqualifiedArrayType(T1, T1Quals);
    QualType UnqualT2 = Context.getUnqualifiedArrayType(T2, T2Quals);
    if (UnqualT1 == UnqualT2) {
      // If the type is an array type, promote the element qualifiers to the type
      // for comparison.
      if (isa<ArrayType>(T1) && T1Quals)
        T1 = Context.getQualifiedType(UnqualT1, T1Quals);
      if (isa<ArrayType>(T2) && T2Quals)
        T2 = Context.getQualifiedType(UnqualT2, T2Quals);
      if (T2.isMoreQualifiedThan(T1))
        return ImplicitConversionSequence::Better;
      else if (T1.isMoreQualifiedThan(T2))
        return ImplicitConversionSequence::Worse;
    }
  }

  return ImplicitConversionSequence::Indistinguishable;
}

/// CompareQualificationConversions - Compares two standard conversion
/// sequences to determine whether they can be ranked based on their
/// qualification conversions (C++ 13.3.3.2p3 bullet 3).
ImplicitConversionSequence::CompareKind
Sema::CompareQualificationConversions(const StandardConversionSequence& SCS1,
                                      const StandardConversionSequence& SCS2) {
  // C++ 13.3.3.2p3:
  //  -- S1 and S2 differ only in their qualification conversion and
  //     yield similar types T1 and T2 (C++ 4.4), respectively, and the
  //     cv-qualification signature of type T1 is a proper subset of
  //     the cv-qualification signature of type T2, and S1 is not the
  //     deprecated string literal array-to-pointer conversion (4.2).
  if (SCS1.First != SCS2.First || SCS1.Second != SCS2.Second ||
      SCS1.Third != SCS2.Third || SCS1.Third != ICK_Qualification)
    return ImplicitConversionSequence::Indistinguishable;

  // FIXME: the example in the standard doesn't use a qualification
  // conversion (!)
  QualType T1 = QualType::getFromOpaquePtr(SCS1.ToTypePtr);
  QualType T2 = QualType::getFromOpaquePtr(SCS2.ToTypePtr);
  T1 = Context.getCanonicalType(T1);
  T2 = Context.getCanonicalType(T2);
  Qualifiers T1Quals, T2Quals;
  QualType UnqualT1 = Context.getUnqualifiedArrayType(T1, T1Quals);
  QualType UnqualT2 = Context.getUnqualifiedArrayType(T2, T2Quals);

  // If the types are the same, we won't learn anything by unwrapped
  // them.
  if (UnqualT1 == UnqualT2)
    return ImplicitConversionSequence::Indistinguishable;

  // If the type is an array type, promote the element qualifiers to the type
  // for comparison.
  if (isa<ArrayType>(T1) && T1Quals)
    T1 = Context.getQualifiedType(UnqualT1, T1Quals);
  if (isa<ArrayType>(T2) && T2Quals)
    T2 = Context.getQualifiedType(UnqualT2, T2Quals);

  ImplicitConversionSequence::CompareKind Result
    = ImplicitConversionSequence::Indistinguishable;
  while (UnwrapSimilarPointerTypes(T1, T2)) {
    // Within each iteration of the loop, we check the qualifiers to
    // determine if this still looks like a qualification
    // conversion. Then, if all is well, we unwrap one more level of
    // pointers or pointers-to-members and do it all again
    // until there are no more pointers or pointers-to-members left
    // to unwrap. This essentially mimics what
    // IsQualificationConversion does, but here we're checking for a
    // strict subset of qualifiers.
    if (T1.getCVRQualifiers() == T2.getCVRQualifiers())
      // The qualifiers are the same, so this doesn't tell us anything
      // about how the sequences rank.
      ;
    else if (T2.isMoreQualifiedThan(T1)) {
      // T1 has fewer qualifiers, so it could be the better sequence.
      if (Result == ImplicitConversionSequence::Worse)
        // Neither has qualifiers that are a subset of the other's
        // qualifiers.
        return ImplicitConversionSequence::Indistinguishable;

      Result = ImplicitConversionSequence::Better;
    } else if (T1.isMoreQualifiedThan(T2)) {
      // T2 has fewer qualifiers, so it could be the better sequence.
      if (Result == ImplicitConversionSequence::Better)
        // Neither has qualifiers that are a subset of the other's
        // qualifiers.
        return ImplicitConversionSequence::Indistinguishable;

      Result = ImplicitConversionSequence::Worse;
    } else {
      // Qualifiers are disjoint.
      return ImplicitConversionSequence::Indistinguishable;
    }

    // If the types after this point are equivalent, we're done.
    if (Context.hasSameUnqualifiedType(T1, T2))
      break;
  }

  // Check that the winning standard conversion sequence isn't using
  // the deprecated string literal array to pointer conversion.
  switch (Result) {
  case ImplicitConversionSequence::Better:
    if (SCS1.Deprecated)
      Result = ImplicitConversionSequence::Indistinguishable;
    break;

  case ImplicitConversionSequence::Indistinguishable:
    break;

  case ImplicitConversionSequence::Worse:
    if (SCS2.Deprecated)
      Result = ImplicitConversionSequence::Indistinguishable;
    break;
  }

  return Result;
}

/// CompareDerivedToBaseConversions - Compares two standard conversion
/// sequences to determine whether they can be ranked based on their
/// various kinds of derived-to-base conversions (C++
/// [over.ics.rank]p4b3).  As part of these checks, we also look at
/// conversions between Objective-C interface types.
ImplicitConversionSequence::CompareKind
Sema::CompareDerivedToBaseConversions(const StandardConversionSequence& SCS1,
                                      const StandardConversionSequence& SCS2) {
  QualType FromType1 = QualType::getFromOpaquePtr(SCS1.FromTypePtr);
  QualType ToType1 = QualType::getFromOpaquePtr(SCS1.ToTypePtr);
  QualType FromType2 = QualType::getFromOpaquePtr(SCS2.FromTypePtr);
  QualType ToType2 = QualType::getFromOpaquePtr(SCS2.ToTypePtr);

  // Adjust the types we're converting from via the array-to-pointer
  // conversion, if we need to.
  if (SCS1.First == ICK_Array_To_Pointer)
    FromType1 = Context.getArrayDecayedType(FromType1);
  if (SCS2.First == ICK_Array_To_Pointer)
    FromType2 = Context.getArrayDecayedType(FromType2);

  // Canonicalize all of the types.
  FromType1 = Context.getCanonicalType(FromType1);
  ToType1 = Context.getCanonicalType(ToType1);
  FromType2 = Context.getCanonicalType(FromType2);
  ToType2 = Context.getCanonicalType(ToType2);

  // C++ [over.ics.rank]p4b3:
  //
  //   If class B is derived directly or indirectly from class A and
  //   class C is derived directly or indirectly from B,
  //
  // For Objective-C, we let A, B, and C also be Objective-C
  // interfaces.

  // Compare based on pointer conversions.
  if (SCS1.Second == ICK_Pointer_Conversion &&
      SCS2.Second == ICK_Pointer_Conversion &&
      /*FIXME: Remove if Objective-C id conversions get their own rank*/
      FromType1->isPointerType() && FromType2->isPointerType() &&
      ToType1->isPointerType() && ToType2->isPointerType()) {
    QualType FromPointee1
      = FromType1->getAs<PointerType>()->getPointeeType().getUnqualifiedType();
    QualType ToPointee1
      = ToType1->getAs<PointerType>()->getPointeeType().getUnqualifiedType();
    QualType FromPointee2
      = FromType2->getAs<PointerType>()->getPointeeType().getUnqualifiedType();
    QualType ToPointee2
      = ToType2->getAs<PointerType>()->getPointeeType().getUnqualifiedType();

    const ObjCInterfaceType* FromIface1 = FromPointee1->getAs<ObjCInterfaceType>();
    const ObjCInterfaceType* FromIface2 = FromPointee2->getAs<ObjCInterfaceType>();
    const ObjCInterfaceType* ToIface1 = ToPointee1->getAs<ObjCInterfaceType>();
    const ObjCInterfaceType* ToIface2 = ToPointee2->getAs<ObjCInterfaceType>();

    //   -- conversion of C* to B* is better than conversion of C* to A*,
    if (FromPointee1 == FromPointee2 && ToPointee1 != ToPointee2) {
      if (IsDerivedFrom(ToPointee1, ToPointee2))
        return ImplicitConversionSequence::Better;
      else if (IsDerivedFrom(ToPointee2, ToPointee1))
        return ImplicitConversionSequence::Worse;

      if (ToIface1 && ToIface2) {
        if (Context.canAssignObjCInterfaces(ToIface2, ToIface1))
          return ImplicitConversionSequence::Better;
        else if (Context.canAssignObjCInterfaces(ToIface1, ToIface2))
          return ImplicitConversionSequence::Worse;
      }
    }

    //   -- conversion of B* to A* is better than conversion of C* to A*,
    if (FromPointee1 != FromPointee2 && ToPointee1 == ToPointee2) {
      if (IsDerivedFrom(FromPointee2, FromPointee1))
        return ImplicitConversionSequence::Better;
      else if (IsDerivedFrom(FromPointee1, FromPointee2))
        return ImplicitConversionSequence::Worse;

      if (FromIface1 && FromIface2) {
        if (Context.canAssignObjCInterfaces(FromIface1, FromIface2))
          return ImplicitConversionSequence::Better;
        else if (Context.canAssignObjCInterfaces(FromIface2, FromIface1))
          return ImplicitConversionSequence::Worse;
      }
    }
  }

  // Compare based on reference bindings.
  if (SCS1.ReferenceBinding && SCS2.ReferenceBinding &&
      SCS1.Second == ICK_Derived_To_Base) {
    //   -- binding of an expression of type C to a reference of type
    //      B& is better than binding an expression of type C to a
    //      reference of type A&,
    if (Context.hasSameUnqualifiedType(FromType1, FromType2) &&
        !Context.hasSameUnqualifiedType(ToType1, ToType2)) {
      if (IsDerivedFrom(ToType1, ToType2))
        return ImplicitConversionSequence::Better;
      else if (IsDerivedFrom(ToType2, ToType1))
        return ImplicitConversionSequence::Worse;
    }

    //   -- binding of an expression of type B to a reference of type
    //      A& is better than binding an expression of type C to a
    //      reference of type A&,
    if (!Context.hasSameUnqualifiedType(FromType1, FromType2) &&
        Context.hasSameUnqualifiedType(ToType1, ToType2)) {
      if (IsDerivedFrom(FromType2, FromType1))
        return ImplicitConversionSequence::Better;
      else if (IsDerivedFrom(FromType1, FromType2))
        return ImplicitConversionSequence::Worse;
    }
  }
  
  // Ranking of member-pointer types.
  if (SCS1.Second == ICK_Pointer_Member && SCS2.Second == ICK_Pointer_Member &&
      FromType1->isMemberPointerType() && FromType2->isMemberPointerType() &&
      ToType1->isMemberPointerType() && ToType2->isMemberPointerType()) {
    const MemberPointerType * FromMemPointer1 = 
                                        FromType1->getAs<MemberPointerType>();
    const MemberPointerType * ToMemPointer1 = 
                                          ToType1->getAs<MemberPointerType>();
    const MemberPointerType * FromMemPointer2 = 
                                          FromType2->getAs<MemberPointerType>();
    const MemberPointerType * ToMemPointer2 = 
                                          ToType2->getAs<MemberPointerType>();
    const Type *FromPointeeType1 = FromMemPointer1->getClass();
    const Type *ToPointeeType1 = ToMemPointer1->getClass();
    const Type *FromPointeeType2 = FromMemPointer2->getClass();
    const Type *ToPointeeType2 = ToMemPointer2->getClass();
    QualType FromPointee1 = QualType(FromPointeeType1, 0).getUnqualifiedType();
    QualType ToPointee1 = QualType(ToPointeeType1, 0).getUnqualifiedType();
    QualType FromPointee2 = QualType(FromPointeeType2, 0).getUnqualifiedType();
    QualType ToPointee2 = QualType(ToPointeeType2, 0).getUnqualifiedType();
    // conversion of A::* to B::* is better than conversion of A::* to C::*,
    if (FromPointee1 == FromPointee2 && ToPointee1 != ToPointee2) {
      if (IsDerivedFrom(ToPointee1, ToPointee2))
        return ImplicitConversionSequence::Worse;
      else if (IsDerivedFrom(ToPointee2, ToPointee1))
        return ImplicitConversionSequence::Better;
    }
    // conversion of B::* to C::* is better than conversion of A::* to C::*
    if (ToPointee1 == ToPointee2 && FromPointee1 != FromPointee2) {
      if (IsDerivedFrom(FromPointee1, FromPointee2))
        return ImplicitConversionSequence::Better;
      else if (IsDerivedFrom(FromPointee2, FromPointee1))
        return ImplicitConversionSequence::Worse;
    }
  }
  
  if (SCS1.CopyConstructor && SCS2.CopyConstructor &&
      SCS1.Second == ICK_Derived_To_Base) {
    //   -- conversion of C to B is better than conversion of C to A,
    if (Context.hasSameUnqualifiedType(FromType1, FromType2) &&
        !Context.hasSameUnqualifiedType(ToType1, ToType2)) {
      if (IsDerivedFrom(ToType1, ToType2))
        return ImplicitConversionSequence::Better;
      else if (IsDerivedFrom(ToType2, ToType1))
        return ImplicitConversionSequence::Worse;
    }

    //   -- conversion of B to A is better than conversion of C to A.
    if (!Context.hasSameUnqualifiedType(FromType1, FromType2) &&
        Context.hasSameUnqualifiedType(ToType1, ToType2)) {
      if (IsDerivedFrom(FromType2, FromType1))
        return ImplicitConversionSequence::Better;
      else if (IsDerivedFrom(FromType1, FromType2))
        return ImplicitConversionSequence::Worse;
    }
  }

  return ImplicitConversionSequence::Indistinguishable;
}

/// TryCopyInitialization - Try to copy-initialize a value of type
/// ToType from the expression From. Return the implicit conversion
/// sequence required to pass this argument, which may be a bad
/// conversion sequence (meaning that the argument cannot be passed to
/// a parameter of this type). If @p SuppressUserConversions, then we
/// do not permit any user-defined conversion sequences. If @p ForceRValue,
/// then we treat @p From as an rvalue, even if it is an lvalue.
ImplicitConversionSequence
Sema::TryCopyInitialization(Expr *From, QualType ToType,
                            bool SuppressUserConversions, bool ForceRValue,
                            bool InOverloadResolution) {
  if (ToType->isReferenceType()) {
    ImplicitConversionSequence ICS;
    CheckReferenceInit(From, ToType,
                       /*FIXME:*/From->getLocStart(),
                       SuppressUserConversions,
                       /*AllowExplicit=*/false,
                       ForceRValue,
                       &ICS);
    return ICS;
  } else {
    return TryImplicitConversion(From, ToType,
                                 SuppressUserConversions,
                                 /*AllowExplicit=*/false,
                                 ForceRValue,
                                 InOverloadResolution);
  }
}

/// PerformCopyInitialization - Copy-initialize an object of type @p ToType with
/// the expression @p From. Returns true (and emits a diagnostic) if there was
/// an error, returns false if the initialization succeeded. Elidable should
/// be true when the copy may be elided (C++ 12.8p15). Overload resolution works
/// differently in C++0x for this case.
bool Sema::PerformCopyInitialization(Expr *&From, QualType ToType,
                                     AssignmentAction Action, bool Elidable) {
  if (!getLangOptions().CPlusPlus) {
    // In C, argument passing is the same as performing an assignment.
    QualType FromType = From->getType();

    AssignConvertType ConvTy =
      CheckSingleAssignmentConstraints(ToType, From);
    if (ConvTy != Compatible &&
        CheckTransparentUnionArgumentConstraints(ToType, From) == Compatible)
      ConvTy = Compatible;

    return DiagnoseAssignmentResult(ConvTy, From->getLocStart(), ToType,
                                    FromType, From, Action);
  }

  if (ToType->isReferenceType())
    return CheckReferenceInit(From, ToType,
                              /*FIXME:*/From->getLocStart(),
                              /*SuppressUserConversions=*/false,
                              /*AllowExplicit=*/false,
                              /*ForceRValue=*/false);

  if (!PerformImplicitConversion(From, ToType, Action,
                                 /*AllowExplicit=*/false, Elidable))
    return false;
  if (!DiagnoseMultipleUserDefinedConversion(From, ToType))
    return Diag(From->getSourceRange().getBegin(),
                diag::err_typecheck_convert_incompatible)
      << ToType << From->getType() << Action << From->getSourceRange();
  return true;
}

/// TryObjectArgumentInitialization - Try to initialize the object
/// parameter of the given member function (@c Method) from the
/// expression @p From.
ImplicitConversionSequence
Sema::TryObjectArgumentInitialization(QualType FromType,
                                      CXXMethodDecl *Method,
                                      CXXRecordDecl *ActingContext) {
  QualType ClassType = Context.getTypeDeclType(ActingContext);
  // [class.dtor]p2: A destructor can be invoked for a const, volatile or
  //                 const volatile object.
  unsigned Quals = isa<CXXDestructorDecl>(Method) ?
    Qualifiers::Const | Qualifiers::Volatile : Method->getTypeQualifiers();
  QualType ImplicitParamType =  Context.getCVRQualifiedType(ClassType, Quals);

  // Set up the conversion sequence as a "bad" conversion, to allow us
  // to exit early.
  ImplicitConversionSequence ICS;
  ICS.Standard.setAsIdentityConversion();
  ICS.ConversionKind = ImplicitConversionSequence::BadConversion;

  // We need to have an object of class type.
  if (const PointerType *PT = FromType->getAs<PointerType>())
    FromType = PT->getPointeeType();

  assert(FromType->isRecordType());

  // The implicit object parameter is has the type "reference to cv X",
  // where X is the class of which the function is a member
  // (C++ [over.match.funcs]p4). However, when finding an implicit
  // conversion sequence for the argument, we are not allowed to
  // create temporaries or perform user-defined conversions
  // (C++ [over.match.funcs]p5). We perform a simplified version of
  // reference binding here, that allows class rvalues to bind to
  // non-constant references.

  // First check the qualifiers. We don't care about lvalue-vs-rvalue
  // with the implicit object parameter (C++ [over.match.funcs]p5).
  QualType FromTypeCanon = Context.getCanonicalType(FromType);
  if (ImplicitParamType.getCVRQualifiers() 
                                    != FromTypeCanon.getLocalCVRQualifiers() &&
      !ImplicitParamType.isAtLeastAsQualifiedAs(FromTypeCanon))
    return ICS;

  // Check that we have either the same type or a derived type. It
  // affects the conversion rank.
  QualType ClassTypeCanon = Context.getCanonicalType(ClassType);
  if (ClassTypeCanon == FromTypeCanon.getLocalUnqualifiedType())
    ICS.Standard.Second = ICK_Identity;
  else if (IsDerivedFrom(FromType, ClassType))
    ICS.Standard.Second = ICK_Derived_To_Base;
  else
    return ICS;

  // Success. Mark this as a reference binding.
  ICS.ConversionKind = ImplicitConversionSequence::StandardConversion;
  ICS.Standard.FromTypePtr = FromType.getAsOpaquePtr();
  ICS.Standard.ToTypePtr = ImplicitParamType.getAsOpaquePtr();
  ICS.Standard.ReferenceBinding = true;
  ICS.Standard.DirectBinding = true;
  ICS.Standard.RRefBinding = false;
  return ICS;
}

/// PerformObjectArgumentInitialization - Perform initialization of
/// the implicit object parameter for the given Method with the given
/// expression.
bool
Sema::PerformObjectArgumentInitialization(Expr *&From, CXXMethodDecl *Method) {
  QualType FromRecordType, DestType;
  QualType ImplicitParamRecordType  =
    Method->getThisType(Context)->getAs<PointerType>()->getPointeeType();

  if (const PointerType *PT = From->getType()->getAs<PointerType>()) {
    FromRecordType = PT->getPointeeType();
    DestType = Method->getThisType(Context);
  } else {
    FromRecordType = From->getType();
    DestType = ImplicitParamRecordType;
  }

  // Note that we always use the true parent context when performing
  // the actual argument initialization.
  ImplicitConversionSequence ICS
    = TryObjectArgumentInitialization(From->getType(), Method,
                                      Method->getParent());
  if (ICS.ConversionKind == ImplicitConversionSequence::BadConversion)
    return Diag(From->getSourceRange().getBegin(),
                diag::err_implicit_object_parameter_init)
       << ImplicitParamRecordType << FromRecordType << From->getSourceRange();

  if (ICS.Standard.Second == ICK_Derived_To_Base &&
      CheckDerivedToBaseConversion(FromRecordType,
                                   ImplicitParamRecordType,
                                   From->getSourceRange().getBegin(),
                                   From->getSourceRange()))
    return true;

  ImpCastExprToType(From, DestType, CastExpr::CK_DerivedToBase,
                    /*isLvalue=*/true);
  return false;
}

/// TryContextuallyConvertToBool - Attempt to contextually convert the
/// expression From to bool (C++0x [conv]p3).
ImplicitConversionSequence Sema::TryContextuallyConvertToBool(Expr *From) {
  return TryImplicitConversion(From, Context.BoolTy,
                               // FIXME: Are these flags correct?
                               /*SuppressUserConversions=*/false,
                               /*AllowExplicit=*/true,
                               /*ForceRValue=*/false,
                               /*InOverloadResolution=*/false);
}

/// PerformContextuallyConvertToBool - Perform a contextual conversion
/// of the expression From to bool (C++0x [conv]p3).
bool Sema::PerformContextuallyConvertToBool(Expr *&From) {
  ImplicitConversionSequence ICS = TryContextuallyConvertToBool(From);
  if (!PerformImplicitConversion(From, Context.BoolTy, ICS, AA_Converting))
    return false;
  
  if (!DiagnoseMultipleUserDefinedConversion(From, Context.BoolTy))
    return  Diag(From->getSourceRange().getBegin(),
                 diag::err_typecheck_bool_condition)
                  << From->getType() << From->getSourceRange();
  return true;
}

/// AddOverloadCandidate - Adds the given function to the set of
/// candidate functions, using the given function call arguments.  If
/// @p SuppressUserConversions, then don't allow user-defined
/// conversions via constructors or conversion operators.
/// If @p ForceRValue, treat all arguments as rvalues. This is a slightly
/// hacky way to implement the overloading rules for elidable copy
/// initialization in C++0x (C++0x 12.8p15).
///
/// \para PartialOverloading true if we are performing "partial" overloading
/// based on an incomplete set of function arguments. This feature is used by
/// code completion.
void
Sema::AddOverloadCandidate(FunctionDecl *Function,
                           Expr **Args, unsigned NumArgs,
                           OverloadCandidateSet& CandidateSet,
                           bool SuppressUserConversions,
                           bool ForceRValue,
                           bool PartialOverloading) {
  const FunctionProtoType* Proto
    = dyn_cast<FunctionProtoType>(Function->getType()->getAs<FunctionType>());
  assert(Proto && "Functions without a prototype cannot be overloaded");
  assert(!isa<CXXConversionDecl>(Function) &&
         "Use AddConversionCandidate for conversion functions");
  assert(!Function->getDescribedFunctionTemplate() &&
         "Use AddTemplateOverloadCandidate for function templates");

  if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(Function)) {
    if (!isa<CXXConstructorDecl>(Method)) {
      // If we get here, it's because we're calling a member function
      // that is named without a member access expression (e.g.,
      // "this->f") that was either written explicitly or created
      // implicitly. This can happen with a qualified call to a member
      // function, e.g., X::f(). We use an empty type for the implied
      // object argument (C++ [over.call.func]p3), and the acting context
      // is irrelevant.
      AddMethodCandidate(Method, Method->getParent(),
                         QualType(), Args, NumArgs, CandidateSet,
                         SuppressUserConversions, ForceRValue);
      return;
    }
    // We treat a constructor like a non-member function, since its object
    // argument doesn't participate in overload resolution.
  }

  if (!CandidateSet.isNewCandidate(Function))
    return;

  // Overload resolution is always an unevaluated context.
  EnterExpressionEvaluationContext Unevaluated(*this, Action::Unevaluated);

  if (CXXConstructorDecl *Constructor = dyn_cast<CXXConstructorDecl>(Function)){
    // C++ [class.copy]p3:
    //   A member function template is never instantiated to perform the copy
    //   of a class object to an object of its class type.
    QualType ClassType = Context.getTypeDeclType(Constructor->getParent());
    if (NumArgs == 1 && 
        Constructor->isCopyConstructorLikeSpecialization() &&
        Context.hasSameUnqualifiedType(ClassType, Args[0]->getType()))
      return;
  }
  
  // Add this candidate
  CandidateSet.push_back(OverloadCandidate());
  OverloadCandidate& Candidate = CandidateSet.back();
  Candidate.Function = Function;
  Candidate.Viable = true;
  Candidate.IsSurrogate = false;
  Candidate.IgnoreObjectArgument = false;

  unsigned NumArgsInProto = Proto->getNumArgs();

  // (C++ 13.3.2p2): A candidate function having fewer than m
  // parameters is viable only if it has an ellipsis in its parameter
  // list (8.3.5).
  if ((NumArgs + (PartialOverloading && NumArgs)) > NumArgsInProto && 
      !Proto->isVariadic()) {
    Candidate.Viable = false;
    return;
  }

  // (C++ 13.3.2p2): A candidate function having more than m parameters
  // is viable only if the (m+1)st parameter has a default argument
  // (8.3.6). For the purposes of overload resolution, the
  // parameter list is truncated on the right, so that there are
  // exactly m parameters.
  unsigned MinRequiredArgs = Function->getMinRequiredArguments();
  if (NumArgs < MinRequiredArgs && !PartialOverloading) {
    // Not enough arguments.
    Candidate.Viable = false;
    return;
  }

  // Determine the implicit conversion sequences for each of the
  // arguments.
  Candidate.Conversions.resize(NumArgs);
  for (unsigned ArgIdx = 0; ArgIdx < NumArgs; ++ArgIdx) {
    if (ArgIdx < NumArgsInProto) {
      // (C++ 13.3.2p3): for F to be a viable function, there shall
      // exist for each argument an implicit conversion sequence
      // (13.3.3.1) that converts that argument to the corresponding
      // parameter of F.
      QualType ParamType = Proto->getArgType(ArgIdx);
      Candidate.Conversions[ArgIdx]
        = TryCopyInitialization(Args[ArgIdx], ParamType,
                                SuppressUserConversions, ForceRValue,
                                /*InOverloadResolution=*/true);
      if (Candidate.Conversions[ArgIdx].ConversionKind
            == ImplicitConversionSequence::BadConversion) {
      // 13.3.3.1-p10 If several different sequences of conversions exist that 
      // each convert the argument to the parameter type, the implicit conversion 
      // sequence associated with the parameter is defined to be the unique conversion 
      // sequence designated the ambiguous conversion sequence. For the purpose of 
      // ranking implicit conversion sequences as described in 13.3.3.2, the ambiguous 
      // conversion sequence is treated as a user-defined sequence that is 
      // indistinguishable from any other user-defined conversion sequence
        if (!Candidate.Conversions[ArgIdx].ConversionFunctionSet.empty()) {
          Candidate.Conversions[ArgIdx].ConversionKind =
            ImplicitConversionSequence::UserDefinedConversion;
          // Set the conversion function to one of them. As due to ambiguity,
          // they carry the same weight and is needed for overload resolution
          // later.
          Candidate.Conversions[ArgIdx].UserDefined.ConversionFunction =
            Candidate.Conversions[ArgIdx].ConversionFunctionSet[0];
        }
        else {
          Candidate.Viable = false;
          break;
        }
      }
    } else {
      // (C++ 13.3.2p2): For the purposes of overload resolution, any
      // argument for which there is no corresponding parameter is
      // considered to ""match the ellipsis" (C+ 13.3.3.1.3).
      Candidate.Conversions[ArgIdx].ConversionKind
        = ImplicitConversionSequence::EllipsisConversion;
    }
  }
}

/// \brief Add all of the function declarations in the given function set to
/// the overload canddiate set.
void Sema::AddFunctionCandidates(const FunctionSet &Functions,
                                 Expr **Args, unsigned NumArgs,
                                 OverloadCandidateSet& CandidateSet,
                                 bool SuppressUserConversions) {
  for (FunctionSet::const_iterator F = Functions.begin(),
                                FEnd = Functions.end();
       F != FEnd; ++F) {
    // FIXME: using declarations
    if (FunctionDecl *FD = dyn_cast<FunctionDecl>(*F)) {
      if (isa<CXXMethodDecl>(FD) && !cast<CXXMethodDecl>(FD)->isStatic())
        AddMethodCandidate(cast<CXXMethodDecl>(FD),
                           cast<CXXMethodDecl>(FD)->getParent(),
                           Args[0]->getType(), Args + 1, NumArgs - 1, 
                           CandidateSet, SuppressUserConversions);
      else
        AddOverloadCandidate(FD, Args, NumArgs, CandidateSet,
                             SuppressUserConversions);
    } else {
      FunctionTemplateDecl *FunTmpl = cast<FunctionTemplateDecl>(*F);
      if (isa<CXXMethodDecl>(FunTmpl->getTemplatedDecl()) &&
          !cast<CXXMethodDecl>(FunTmpl->getTemplatedDecl())->isStatic())
        AddMethodTemplateCandidate(FunTmpl,
                              cast<CXXRecordDecl>(FunTmpl->getDeclContext()),
                                   /*FIXME: explicit args */ 0,
                                   Args[0]->getType(), Args + 1, NumArgs - 1,
                                   CandidateSet,
                                   SuppressUserConversions);
      else
        AddTemplateOverloadCandidate(FunTmpl,
                                     /*FIXME: explicit args */ 0,
                                     Args, NumArgs, CandidateSet,
                                     SuppressUserConversions);
    }
  }
}

/// AddMethodCandidate - Adds a named decl (which is some kind of
/// method) as a method candidate to the given overload set.
void Sema::AddMethodCandidate(NamedDecl *Decl,
                              QualType ObjectType,
                              Expr **Args, unsigned NumArgs,
                              OverloadCandidateSet& CandidateSet,
                              bool SuppressUserConversions, bool ForceRValue) {

  // FIXME: use this
  CXXRecordDecl *ActingContext = cast<CXXRecordDecl>(Decl->getDeclContext());

  if (isa<UsingShadowDecl>(Decl))
    Decl = cast<UsingShadowDecl>(Decl)->getTargetDecl();
  
  if (FunctionTemplateDecl *TD = dyn_cast<FunctionTemplateDecl>(Decl)) {
    assert(isa<CXXMethodDecl>(TD->getTemplatedDecl()) &&
           "Expected a member function template");
    AddMethodTemplateCandidate(TD, ActingContext, /*ExplicitArgs*/ 0,
                               ObjectType, Args, NumArgs,
                               CandidateSet,
                               SuppressUserConversions,
                               ForceRValue);
  } else {
    AddMethodCandidate(cast<CXXMethodDecl>(Decl), ActingContext,
                       ObjectType, Args, NumArgs,
                       CandidateSet, SuppressUserConversions, ForceRValue);
  }
}

/// AddMethodCandidate - Adds the given C++ member function to the set
/// of candidate functions, using the given function call arguments
/// and the object argument (@c Object). For example, in a call
/// @c o.f(a1,a2), @c Object will contain @c o and @c Args will contain
/// both @c a1 and @c a2. If @p SuppressUserConversions, then don't
/// allow user-defined conversions via constructors or conversion
/// operators. If @p ForceRValue, treat all arguments as rvalues. This is
/// a slightly hacky way to implement the overloading rules for elidable copy
/// initialization in C++0x (C++0x 12.8p15).
void
Sema::AddMethodCandidate(CXXMethodDecl *Method, CXXRecordDecl *ActingContext,
                         QualType ObjectType, Expr **Args, unsigned NumArgs,
                         OverloadCandidateSet& CandidateSet,
                         bool SuppressUserConversions, bool ForceRValue) {
  const FunctionProtoType* Proto
    = dyn_cast<FunctionProtoType>(Method->getType()->getAs<FunctionType>());
  assert(Proto && "Methods without a prototype cannot be overloaded");
  assert(!isa<CXXConversionDecl>(Method) &&
         "Use AddConversionCandidate for conversion functions");
  assert(!isa<CXXConstructorDecl>(Method) &&
         "Use AddOverloadCandidate for constructors");

  if (!CandidateSet.isNewCandidate(Method))
    return;

  // Overload resolution is always an unevaluated context.
  EnterExpressionEvaluationContext Unevaluated(*this, Action::Unevaluated);

  // Add this candidate
  CandidateSet.push_back(OverloadCandidate());
  OverloadCandidate& Candidate = CandidateSet.back();
  Candidate.Function = Method;
  Candidate.IsSurrogate = false;
  Candidate.IgnoreObjectArgument = false;

  unsigned NumArgsInProto = Proto->getNumArgs();

  // (C++ 13.3.2p2): A candidate function having fewer than m
  // parameters is viable only if it has an ellipsis in its parameter
  // list (8.3.5).
  if (NumArgs > NumArgsInProto && !Proto->isVariadic()) {
    Candidate.Viable = false;
    return;
  }

  // (C++ 13.3.2p2): A candidate function having more than m parameters
  // is viable only if the (m+1)st parameter has a default argument
  // (8.3.6). For the purposes of overload resolution, the
  // parameter list is truncated on the right, so that there are
  // exactly m parameters.
  unsigned MinRequiredArgs = Method->getMinRequiredArguments();
  if (NumArgs < MinRequiredArgs) {
    // Not enough arguments.
    Candidate.Viable = false;
    return;
  }

  Candidate.Viable = true;
  Candidate.Conversions.resize(NumArgs + 1);

  if (Method->isStatic() || ObjectType.isNull())
    // The implicit object argument is ignored.
    Candidate.IgnoreObjectArgument = true;
  else {
    // Determine the implicit conversion sequence for the object
    // parameter.
    Candidate.Conversions[0]
      = TryObjectArgumentInitialization(ObjectType, Method, ActingContext);
    if (Candidate.Conversions[0].ConversionKind
          == ImplicitConversionSequence::BadConversion) {
      Candidate.Viable = false;
      return;
    }
  }

  // Determine the implicit conversion sequences for each of the
  // arguments.
  for (unsigned ArgIdx = 0; ArgIdx < NumArgs; ++ArgIdx) {
    if (ArgIdx < NumArgsInProto) {
      // (C++ 13.3.2p3): for F to be a viable function, there shall
      // exist for each argument an implicit conversion sequence
      // (13.3.3.1) that converts that argument to the corresponding
      // parameter of F.
      QualType ParamType = Proto->getArgType(ArgIdx);
      Candidate.Conversions[ArgIdx + 1]
        = TryCopyInitialization(Args[ArgIdx], ParamType,
                                SuppressUserConversions, ForceRValue,
                                /*InOverloadResolution=*/true);
      if (Candidate.Conversions[ArgIdx + 1].ConversionKind
            == ImplicitConversionSequence::BadConversion) {
        Candidate.Viable = false;
        break;
      }
    } else {
      // (C++ 13.3.2p2): For the purposes of overload resolution, any
      // argument for which there is no corresponding parameter is
      // considered to ""match the ellipsis" (C+ 13.3.3.1.3).
      Candidate.Conversions[ArgIdx + 1].ConversionKind
        = ImplicitConversionSequence::EllipsisConversion;
    }
  }
}

/// \brief Add a C++ member function template as a candidate to the candidate
/// set, using template argument deduction to produce an appropriate member
/// function template specialization.
void
Sema::AddMethodTemplateCandidate(FunctionTemplateDecl *MethodTmpl,
                                 CXXRecordDecl *ActingContext,
                        const TemplateArgumentListInfo *ExplicitTemplateArgs,
                                 QualType ObjectType,
                                 Expr **Args, unsigned NumArgs,
                                 OverloadCandidateSet& CandidateSet,
                                 bool SuppressUserConversions,
                                 bool ForceRValue) {
  if (!CandidateSet.isNewCandidate(MethodTmpl))
    return;

  // C++ [over.match.funcs]p7:
  //   In each case where a candidate is a function template, candidate
  //   function template specializations are generated using template argument
  //   deduction (14.8.3, 14.8.2). Those candidates are then handled as
  //   candidate functions in the usual way.113) A given name can refer to one
  //   or more function templates and also to a set of overloaded non-template
  //   functions. In such a case, the candidate functions generated from each
  //   function template are combined with the set of non-template candidate
  //   functions.
  TemplateDeductionInfo Info(Context);
  FunctionDecl *Specialization = 0;
  if (TemplateDeductionResult Result
      = DeduceTemplateArguments(MethodTmpl, ExplicitTemplateArgs,
                                Args, NumArgs, Specialization, Info)) {
        // FIXME: Record what happened with template argument deduction, so
        // that we can give the user a beautiful diagnostic.
        (void)Result;
        return;
      }

  // Add the function template specialization produced by template argument
  // deduction as a candidate.
  assert(Specialization && "Missing member function template specialization?");
  assert(isa<CXXMethodDecl>(Specialization) &&
         "Specialization is not a member function?");
  AddMethodCandidate(cast<CXXMethodDecl>(Specialization), ActingContext,
                     ObjectType, Args, NumArgs,
                     CandidateSet, SuppressUserConversions, ForceRValue);
}

/// \brief Add a C++ function template specialization as a candidate
/// in the candidate set, using template argument deduction to produce
/// an appropriate function template specialization.
void
Sema::AddTemplateOverloadCandidate(FunctionTemplateDecl *FunctionTemplate,
                        const TemplateArgumentListInfo *ExplicitTemplateArgs,
                                   Expr **Args, unsigned NumArgs,
                                   OverloadCandidateSet& CandidateSet,
                                   bool SuppressUserConversions,
                                   bool ForceRValue) {
  if (!CandidateSet.isNewCandidate(FunctionTemplate))
    return;

  // C++ [over.match.funcs]p7:
  //   In each case where a candidate is a function template, candidate
  //   function template specializations are generated using template argument
  //   deduction (14.8.3, 14.8.2). Those candidates are then handled as
  //   candidate functions in the usual way.113) A given name can refer to one
  //   or more function templates and also to a set of overloaded non-template
  //   functions. In such a case, the candidate functions generated from each
  //   function template are combined with the set of non-template candidate
  //   functions.
  TemplateDeductionInfo Info(Context);
  FunctionDecl *Specialization = 0;
  if (TemplateDeductionResult Result
        = DeduceTemplateArguments(FunctionTemplate, ExplicitTemplateArgs,
                                  Args, NumArgs, Specialization, Info)) {
    // FIXME: Record what happened with template argument deduction, so
    // that we can give the user a beautiful diagnostic.
    (void) Result;

    CandidateSet.push_back(OverloadCandidate());
    OverloadCandidate &Candidate = CandidateSet.back();
    Candidate.Function = FunctionTemplate->getTemplatedDecl();
    Candidate.Viable = false;
    Candidate.IsSurrogate = false;
    Candidate.IgnoreObjectArgument = false;
    return;
  }

  // Add the function template specialization produced by template argument
  // deduction as a candidate.
  assert(Specialization && "Missing function template specialization?");
  AddOverloadCandidate(Specialization, Args, NumArgs, CandidateSet,
                       SuppressUserConversions, ForceRValue);
}

/// AddConversionCandidate - Add a C++ conversion function as a
/// candidate in the candidate set (C++ [over.match.conv],
/// C++ [over.match.copy]). From is the expression we're converting from,
/// and ToType is the type that we're eventually trying to convert to
/// (which may or may not be the same type as the type that the
/// conversion function produces).
void
Sema::AddConversionCandidate(CXXConversionDecl *Conversion,
                             CXXRecordDecl *ActingContext,
                             Expr *From, QualType ToType,
                             OverloadCandidateSet& CandidateSet) {
  assert(!Conversion->getDescribedFunctionTemplate() &&
         "Conversion function templates use AddTemplateConversionCandidate");

  if (!CandidateSet.isNewCandidate(Conversion))
    return;

  // Overload resolution is always an unevaluated context.
  EnterExpressionEvaluationContext Unevaluated(*this, Action::Unevaluated);

  // Add this candidate
  CandidateSet.push_back(OverloadCandidate());
  OverloadCandidate& Candidate = CandidateSet.back();
  Candidate.Function = Conversion;
  Candidate.IsSurrogate = false;
  Candidate.IgnoreObjectArgument = false;
  Candidate.FinalConversion.setAsIdentityConversion();
  Candidate.FinalConversion.FromTypePtr
    = Conversion->getConversionType().getAsOpaquePtr();
  Candidate.FinalConversion.ToTypePtr = ToType.getAsOpaquePtr();

  // Determine the implicit conversion sequence for the implicit
  // object parameter.
  Candidate.Viable = true;
  Candidate.Conversions.resize(1);
  Candidate.Conversions[0]
    = TryObjectArgumentInitialization(From->getType(), Conversion,
                                      ActingContext);
  // Conversion functions to a different type in the base class is visible in 
  // the derived class.  So, a derived to base conversion should not participate
  // in overload resolution. 
  if (Candidate.Conversions[0].Standard.Second == ICK_Derived_To_Base)
    Candidate.Conversions[0].Standard.Second = ICK_Identity;
  if (Candidate.Conversions[0].ConversionKind
      == ImplicitConversionSequence::BadConversion) {
    Candidate.Viable = false;
    return;
  }
  
  // We won't go through a user-define type conversion function to convert a 
  // derived to base as such conversions are given Conversion Rank. They only
  // go through a copy constructor. 13.3.3.1.2-p4 [over.ics.user]
  QualType FromCanon
    = Context.getCanonicalType(From->getType().getUnqualifiedType());
  QualType ToCanon = Context.getCanonicalType(ToType).getUnqualifiedType();
  if (FromCanon == ToCanon || IsDerivedFrom(FromCanon, ToCanon)) {
    Candidate.Viable = false;
    return;
  }
  

  // To determine what the conversion from the result of calling the
  // conversion function to the type we're eventually trying to
  // convert to (ToType), we need to synthesize a call to the
  // conversion function and attempt copy initialization from it. This
  // makes sure that we get the right semantics with respect to
  // lvalues/rvalues and the type. Fortunately, we can allocate this
  // call on the stack and we don't need its arguments to be
  // well-formed.
  DeclRefExpr ConversionRef(Conversion, Conversion->getType(),
                            From->getLocStart());
  ImplicitCastExpr ConversionFn(Context.getPointerType(Conversion->getType()),
                                CastExpr::CK_FunctionToPointerDecay,
                                &ConversionRef, false);

  // Note that it is safe to allocate CallExpr on the stack here because
  // there are 0 arguments (i.e., nothing is allocated using ASTContext's
  // allocator).
  CallExpr Call(Context, &ConversionFn, 0, 0,
                Conversion->getConversionType().getNonReferenceType(),
                From->getLocStart());
  ImplicitConversionSequence ICS =
    TryCopyInitialization(&Call, ToType,
                          /*SuppressUserConversions=*/true,
                          /*ForceRValue=*/false,
                          /*InOverloadResolution=*/false);

  switch (ICS.ConversionKind) {
  case ImplicitConversionSequence::StandardConversion:
    Candidate.FinalConversion = ICS.Standard;
    break;

  case ImplicitConversionSequence::BadConversion:
    Candidate.Viable = false;
    break;

  default:
    assert(false &&
           "Can only end up with a standard conversion sequence or failure");
  }
}

/// \brief Adds a conversion function template specialization
/// candidate to the overload set, using template argument deduction
/// to deduce the template arguments of the conversion function
/// template from the type that we are converting to (C++
/// [temp.deduct.conv]).
void
Sema::AddTemplateConversionCandidate(FunctionTemplateDecl *FunctionTemplate,
                                     CXXRecordDecl *ActingDC,
                                     Expr *From, QualType ToType,
                                     OverloadCandidateSet &CandidateSet) {
  assert(isa<CXXConversionDecl>(FunctionTemplate->getTemplatedDecl()) &&
         "Only conversion function templates permitted here");

  if (!CandidateSet.isNewCandidate(FunctionTemplate))
    return;

  TemplateDeductionInfo Info(Context);
  CXXConversionDecl *Specialization = 0;
  if (TemplateDeductionResult Result
        = DeduceTemplateArguments(FunctionTemplate, ToType,
                                  Specialization, Info)) {
    // FIXME: Record what happened with template argument deduction, so
    // that we can give the user a beautiful diagnostic.
    (void)Result;
    return;
  }

  // Add the conversion function template specialization produced by
  // template argument deduction as a candidate.
  assert(Specialization && "Missing function template specialization?");
  AddConversionCandidate(Specialization, ActingDC, From, ToType, CandidateSet);
}

/// AddSurrogateCandidate - Adds a "surrogate" candidate function that
/// converts the given @c Object to a function pointer via the
/// conversion function @c Conversion, and then attempts to call it
/// with the given arguments (C++ [over.call.object]p2-4). Proto is
/// the type of function that we'll eventually be calling.
void Sema::AddSurrogateCandidate(CXXConversionDecl *Conversion,
                                 CXXRecordDecl *ActingContext,
                                 const FunctionProtoType *Proto,
                                 QualType ObjectType,
                                 Expr **Args, unsigned NumArgs,
                                 OverloadCandidateSet& CandidateSet) {
  if (!CandidateSet.isNewCandidate(Conversion))
    return;

  // Overload resolution is always an unevaluated context.
  EnterExpressionEvaluationContext Unevaluated(*this, Action::Unevaluated);

  CandidateSet.push_back(OverloadCandidate());
  OverloadCandidate& Candidate = CandidateSet.back();
  Candidate.Function = 0;
  Candidate.Surrogate = Conversion;
  Candidate.Viable = true;
  Candidate.IsSurrogate = true;
  Candidate.IgnoreObjectArgument = false;
  Candidate.Conversions.resize(NumArgs + 1);

  // Determine the implicit conversion sequence for the implicit
  // object parameter.
  ImplicitConversionSequence ObjectInit
    = TryObjectArgumentInitialization(ObjectType, Conversion, ActingContext);
  if (ObjectInit.ConversionKind == ImplicitConversionSequence::BadConversion) {
    Candidate.Viable = false;
    return;
  }

  // The first conversion is actually a user-defined conversion whose
  // first conversion is ObjectInit's standard conversion (which is
  // effectively a reference binding). Record it as such.
  Candidate.Conversions[0].ConversionKind
    = ImplicitConversionSequence::UserDefinedConversion;
  Candidate.Conversions[0].UserDefined.Before = ObjectInit.Standard;
  Candidate.Conversions[0].UserDefined.EllipsisConversion = false;
  Candidate.Conversions[0].UserDefined.ConversionFunction = Conversion;
  Candidate.Conversions[0].UserDefined.After
    = Candidate.Conversions[0].UserDefined.Before;
  Candidate.Conversions[0].UserDefined.After.setAsIdentityConversion();

  // Find the
  unsigned NumArgsInProto = Proto->getNumArgs();

  // (C++ 13.3.2p2): A candidate function having fewer than m
  // parameters is viable only if it has an ellipsis in its parameter
  // list (8.3.5).
  if (NumArgs > NumArgsInProto && !Proto->isVariadic()) {
    Candidate.Viable = false;
    return;
  }

  // Function types don't have any default arguments, so just check if
  // we have enough arguments.
  if (NumArgs < NumArgsInProto) {
    // Not enough arguments.
    Candidate.Viable = false;
    return;
  }

  // Determine the implicit conversion sequences for each of the
  // arguments.
  for (unsigned ArgIdx = 0; ArgIdx < NumArgs; ++ArgIdx) {
    if (ArgIdx < NumArgsInProto) {
      // (C++ 13.3.2p3): for F to be a viable function, there shall
      // exist for each argument an implicit conversion sequence
      // (13.3.3.1) that converts that argument to the corresponding
      // parameter of F.
      QualType ParamType = Proto->getArgType(ArgIdx);
      Candidate.Conversions[ArgIdx + 1]
        = TryCopyInitialization(Args[ArgIdx], ParamType,
                                /*SuppressUserConversions=*/false,
                                /*ForceRValue=*/false,
                                /*InOverloadResolution=*/false);
      if (Candidate.Conversions[ArgIdx + 1].ConversionKind
            == ImplicitConversionSequence::BadConversion) {
        Candidate.Viable = false;
        break;
      }
    } else {
      // (C++ 13.3.2p2): For the purposes of overload resolution, any
      // argument for which there is no corresponding parameter is
      // considered to ""match the ellipsis" (C+ 13.3.3.1.3).
      Candidate.Conversions[ArgIdx + 1].ConversionKind
        = ImplicitConversionSequence::EllipsisConversion;
    }
  }
}

// FIXME: This will eventually be removed, once we've migrated all of the
// operator overloading logic over to the scheme used by binary operators, which
// works for template instantiation.
void Sema::AddOperatorCandidates(OverloadedOperatorKind Op, Scope *S,
                                 SourceLocation OpLoc,
                                 Expr **Args, unsigned NumArgs,
                                 OverloadCandidateSet& CandidateSet,
                                 SourceRange OpRange) {
  FunctionSet Functions;

  QualType T1 = Args[0]->getType();
  QualType T2;
  if (NumArgs > 1)
    T2 = Args[1]->getType();

  DeclarationName OpName = Context.DeclarationNames.getCXXOperatorName(Op);
  if (S)
    LookupOverloadedOperatorName(Op, S, T1, T2, Functions);
  ArgumentDependentLookup(OpName, /*Operator*/true, Args, NumArgs, Functions);
  AddFunctionCandidates(Functions, Args, NumArgs, CandidateSet);
  AddMemberOperatorCandidates(Op, OpLoc, Args, NumArgs, CandidateSet, OpRange);
  AddBuiltinOperatorCandidates(Op, OpLoc, Args, NumArgs, CandidateSet);
}

/// \brief Add overload candidates for overloaded operators that are
/// member functions.
///
/// Add the overloaded operator candidates that are member functions
/// for the operator Op that was used in an operator expression such
/// as "x Op y". , Args/NumArgs provides the operator arguments, and
/// CandidateSet will store the added overload candidates. (C++
/// [over.match.oper]).
void Sema::AddMemberOperatorCandidates(OverloadedOperatorKind Op,
                                       SourceLocation OpLoc,
                                       Expr **Args, unsigned NumArgs,
                                       OverloadCandidateSet& CandidateSet,
                                       SourceRange OpRange) {
  DeclarationName OpName = Context.DeclarationNames.getCXXOperatorName(Op);

  // C++ [over.match.oper]p3:
  //   For a unary operator @ with an operand of a type whose
  //   cv-unqualified version is T1, and for a binary operator @ with
  //   a left operand of a type whose cv-unqualified version is T1 and
  //   a right operand of a type whose cv-unqualified version is T2,
  //   three sets of candidate functions, designated member
  //   candidates, non-member candidates and built-in candidates, are
  //   constructed as follows:
  QualType T1 = Args[0]->getType();
  QualType T2;
  if (NumArgs > 1)
    T2 = Args[1]->getType();

  //     -- If T1 is a class type, the set of member candidates is the
  //        result of the qualified lookup of T1::operator@
  //        (13.3.1.1.1); otherwise, the set of member candidates is
  //        empty.
  if (const RecordType *T1Rec = T1->getAs<RecordType>()) {
    // Complete the type if it can be completed. Otherwise, we're done.
    if (RequireCompleteType(OpLoc, T1, PDiag()))
      return;

    LookupResult Operators(*this, OpName, OpLoc, LookupOrdinaryName);
    LookupQualifiedName(Operators, T1Rec->getDecl());
    Operators.suppressDiagnostics();

    for (LookupResult::iterator Oper = Operators.begin(),
                             OperEnd = Operators.end();
         Oper != OperEnd;
         ++Oper)
      AddMethodCandidate(*Oper, Args[0]->getType(),
                         Args + 1, NumArgs - 1, CandidateSet,
                         /* SuppressUserConversions = */ false);
  }
}

/// AddBuiltinCandidate - Add a candidate for a built-in
/// operator. ResultTy and ParamTys are the result and parameter types
/// of the built-in candidate, respectively. Args and NumArgs are the
/// arguments being passed to the candidate. IsAssignmentOperator
/// should be true when this built-in candidate is an assignment
/// operator. NumContextualBoolArguments is the number of arguments
/// (at the beginning of the argument list) that will be contextually
/// converted to bool.
void Sema::AddBuiltinCandidate(QualType ResultTy, QualType *ParamTys,
                               Expr **Args, unsigned NumArgs,
                               OverloadCandidateSet& CandidateSet,
                               bool IsAssignmentOperator,
                               unsigned NumContextualBoolArguments) {
  // Overload resolution is always an unevaluated context.
  EnterExpressionEvaluationContext Unevaluated(*this, Action::Unevaluated);

  // Add this candidate
  CandidateSet.push_back(OverloadCandidate());
  OverloadCandidate& Candidate = CandidateSet.back();
  Candidate.Function = 0;
  Candidate.IsSurrogate = false;
  Candidate.IgnoreObjectArgument = false;
  Candidate.BuiltinTypes.ResultTy = ResultTy;
  for (unsigned ArgIdx = 0; ArgIdx < NumArgs; ++ArgIdx)
    Candidate.BuiltinTypes.ParamTypes[ArgIdx] = ParamTys[ArgIdx];

  // Determine the implicit conversion sequences for each of the
  // arguments.
  Candidate.Viable = true;
  Candidate.Conversions.resize(NumArgs);
  for (unsigned ArgIdx = 0; ArgIdx < NumArgs; ++ArgIdx) {
    // C++ [over.match.oper]p4:
    //   For the built-in assignment operators, conversions of the
    //   left operand are restricted as follows:
    //     -- no temporaries are introduced to hold the left operand, and
    //     -- no user-defined conversions are applied to the left
    //        operand to achieve a type match with the left-most
    //        parameter of a built-in candidate.
    //
    // We block these conversions by turning off user-defined
    // conversions, since that is the only way that initialization of
    // a reference to a non-class type can occur from something that
    // is not of the same type.
    if (ArgIdx < NumContextualBoolArguments) {
      assert(ParamTys[ArgIdx] == Context.BoolTy &&
             "Contextual conversion to bool requires bool type");
      Candidate.Conversions[ArgIdx] = TryContextuallyConvertToBool(Args[ArgIdx]);
    } else {
      Candidate.Conversions[ArgIdx]
        = TryCopyInitialization(Args[ArgIdx], ParamTys[ArgIdx],
                                ArgIdx == 0 && IsAssignmentOperator,
                                /*ForceRValue=*/false,
                                /*InOverloadResolution=*/false);
    }
    if (Candidate.Conversions[ArgIdx].ConversionKind
        == ImplicitConversionSequence::BadConversion) {
      Candidate.Viable = false;
      break;
    }
  }
}

/// BuiltinCandidateTypeSet - A set of types that will be used for the
/// candidate operator functions for built-in operators (C++
/// [over.built]). The types are separated into pointer types and
/// enumeration types.
class BuiltinCandidateTypeSet  {
  /// TypeSet - A set of types.
  typedef llvm::SmallPtrSet<QualType, 8> TypeSet;

  /// PointerTypes - The set of pointer types that will be used in the
  /// built-in candidates.
  TypeSet PointerTypes;

  /// MemberPointerTypes - The set of member pointer types that will be
  /// used in the built-in candidates.
  TypeSet MemberPointerTypes;

  /// EnumerationTypes - The set of enumeration types that will be
  /// used in the built-in candidates.
  TypeSet EnumerationTypes;

  /// Sema - The semantic analysis instance where we are building the
  /// candidate type set.
  Sema &SemaRef;

  /// Context - The AST context in which we will build the type sets.
  ASTContext &Context;

  bool AddPointerWithMoreQualifiedTypeVariants(QualType Ty,
                                               const Qualifiers &VisibleQuals);
  bool AddMemberPointerWithMoreQualifiedTypeVariants(QualType Ty);

public:
  /// iterator - Iterates through the types that are part of the set.
  typedef TypeSet::iterator iterator;

  BuiltinCandidateTypeSet(Sema &SemaRef)
    : SemaRef(SemaRef), Context(SemaRef.Context) { }

  void AddTypesConvertedFrom(QualType Ty, 
                             SourceLocation Loc,
                             bool AllowUserConversions,
                             bool AllowExplicitConversions,
                             const Qualifiers &VisibleTypeConversionsQuals);

  /// pointer_begin - First pointer type found;
  iterator pointer_begin() { return PointerTypes.begin(); }

  /// pointer_end - Past the last pointer type found;
  iterator pointer_end() { return PointerTypes.end(); }

  /// member_pointer_begin - First member pointer type found;
  iterator member_pointer_begin() { return MemberPointerTypes.begin(); }

  /// member_pointer_end - Past the last member pointer type found;
  iterator member_pointer_end() { return MemberPointerTypes.end(); }

  /// enumeration_begin - First enumeration type found;
  iterator enumeration_begin() { return EnumerationTypes.begin(); }

  /// enumeration_end - Past the last enumeration type found;
  iterator enumeration_end() { return EnumerationTypes.end(); }
};

/// AddPointerWithMoreQualifiedTypeVariants - Add the pointer type @p Ty to
/// the set of pointer types along with any more-qualified variants of
/// that type. For example, if @p Ty is "int const *", this routine
/// will add "int const *", "int const volatile *", "int const
/// restrict *", and "int const volatile restrict *" to the set of
/// pointer types. Returns true if the add of @p Ty itself succeeded,
/// false otherwise.
///
/// FIXME: what to do about extended qualifiers?
bool
BuiltinCandidateTypeSet::AddPointerWithMoreQualifiedTypeVariants(QualType Ty,
                                             const Qualifiers &VisibleQuals) {

  // Insert this type.
  if (!PointerTypes.insert(Ty))
    return false;

  const PointerType *PointerTy = Ty->getAs<PointerType>();
  assert(PointerTy && "type was not a pointer type!");

  QualType PointeeTy = PointerTy->getPointeeType();
  // Don't add qualified variants of arrays. For one, they're not allowed
  // (the qualifier would sink to the element type), and for another, the
  // only overload situation where it matters is subscript or pointer +- int,
  // and those shouldn't have qualifier variants anyway.
  if (PointeeTy->isArrayType())
    return true;
  unsigned BaseCVR = PointeeTy.getCVRQualifiers();
  if (const ConstantArrayType *Array =Context.getAsConstantArrayType(PointeeTy))
    BaseCVR = Array->getElementType().getCVRQualifiers();
  bool hasVolatile = VisibleQuals.hasVolatile();
  bool hasRestrict = VisibleQuals.hasRestrict();
  
  // Iterate through all strict supersets of BaseCVR.
  for (unsigned CVR = BaseCVR+1; CVR <= Qualifiers::CVRMask; ++CVR) {
    if ((CVR | BaseCVR) != CVR) continue;
    // Skip over Volatile/Restrict if no Volatile/Restrict found anywhere
    // in the types.
    if ((CVR & Qualifiers::Volatile) && !hasVolatile) continue;
    if ((CVR & Qualifiers::Restrict) && !hasRestrict) continue;
    QualType QPointeeTy = Context.getCVRQualifiedType(PointeeTy, CVR);
    PointerTypes.insert(Context.getPointerType(QPointeeTy));
  }

  return true;
}

/// AddMemberPointerWithMoreQualifiedTypeVariants - Add the pointer type @p Ty
/// to the set of pointer types along with any more-qualified variants of
/// that type. For example, if @p Ty is "int const *", this routine
/// will add "int const *", "int const volatile *", "int const
/// restrict *", and "int const volatile restrict *" to the set of
/// pointer types. Returns true if the add of @p Ty itself succeeded,
/// false otherwise.
///
/// FIXME: what to do about extended qualifiers?
bool
BuiltinCandidateTypeSet::AddMemberPointerWithMoreQualifiedTypeVariants(
    QualType Ty) {
  // Insert this type.
  if (!MemberPointerTypes.insert(Ty))
    return false;

  const MemberPointerType *PointerTy = Ty->getAs<MemberPointerType>();
  assert(PointerTy && "type was not a member pointer type!");

  QualType PointeeTy = PointerTy->getPointeeType();
  // Don't add qualified variants of arrays. For one, they're not allowed
  // (the qualifier would sink to the element type), and for another, the
  // only overload situation where it matters is subscript or pointer +- int,
  // and those shouldn't have qualifier variants anyway.
  if (PointeeTy->isArrayType())
    return true;
  const Type *ClassTy = PointerTy->getClass();

  // Iterate through all strict supersets of the pointee type's CVR
  // qualifiers.
  unsigned BaseCVR = PointeeTy.getCVRQualifiers();
  for (unsigned CVR = BaseCVR+1; CVR <= Qualifiers::CVRMask; ++CVR) {
    if ((CVR | BaseCVR) != CVR) continue;
    
    QualType QPointeeTy = Context.getCVRQualifiedType(PointeeTy, CVR);
    MemberPointerTypes.insert(Context.getMemberPointerType(QPointeeTy, ClassTy));
  }

  return true;
}

/// AddTypesConvertedFrom - Add each of the types to which the type @p
/// Ty can be implicit converted to the given set of @p Types. We're
/// primarily interested in pointer types and enumeration types. We also
/// take member pointer types, for the conditional operator.
/// AllowUserConversions is true if we should look at the conversion
/// functions of a class type, and AllowExplicitConversions if we
/// should also include the explicit conversion functions of a class
/// type.
void
BuiltinCandidateTypeSet::AddTypesConvertedFrom(QualType Ty,
                                               SourceLocation Loc,
                                               bool AllowUserConversions,
                                               bool AllowExplicitConversions,
                                               const Qualifiers &VisibleQuals) {
  // Only deal with canonical types.
  Ty = Context.getCanonicalType(Ty);

  // Look through reference types; they aren't part of the type of an
  // expression for the purposes of conversions.
  if (const ReferenceType *RefTy = Ty->getAs<ReferenceType>())
    Ty = RefTy->getPointeeType();

  // We don't care about qualifiers on the type.
  Ty = Ty.getLocalUnqualifiedType();

  // If we're dealing with an array type, decay to the pointer.
  if (Ty->isArrayType())
    Ty = SemaRef.Context.getArrayDecayedType(Ty);

  if (const PointerType *PointerTy = Ty->getAs<PointerType>()) {
    QualType PointeeTy = PointerTy->getPointeeType();

    // Insert our type, and its more-qualified variants, into the set
    // of types.
    if (!AddPointerWithMoreQualifiedTypeVariants(Ty, VisibleQuals))
      return;
  } else if (Ty->isMemberPointerType()) {
    // Member pointers are far easier, since the pointee can't be converted.
    if (!AddMemberPointerWithMoreQualifiedTypeVariants(Ty))
      return;
  } else if (Ty->isEnumeralType()) {
    EnumerationTypes.insert(Ty);
  } else if (AllowUserConversions) {
    if (const RecordType *TyRec = Ty->getAs<RecordType>()) {
      if (SemaRef.RequireCompleteType(Loc, Ty, 0)) {
        // No conversion functions in incomplete types.
        return;
      }

      CXXRecordDecl *ClassDecl = cast<CXXRecordDecl>(TyRec->getDecl());
      const UnresolvedSet *Conversions
        = ClassDecl->getVisibleConversionFunctions();
      for (UnresolvedSet::iterator I = Conversions->begin(),
             E = Conversions->end(); I != E; ++I) {

        // Skip conversion function templates; they don't tell us anything
        // about which builtin types we can convert to.
        if (isa<FunctionTemplateDecl>(*I))
          continue;

        CXXConversionDecl *Conv = cast<CXXConversionDecl>(*I);
        if (AllowExplicitConversions || !Conv->isExplicit()) {
          AddTypesConvertedFrom(Conv->getConversionType(), Loc, false, false, 
                                VisibleQuals);
        }
      }
    }
  }
}

/// \brief Helper function for AddBuiltinOperatorCandidates() that adds
/// the volatile- and non-volatile-qualified assignment operators for the
/// given type to the candidate set.
static void AddBuiltinAssignmentOperatorCandidates(Sema &S,
                                                   QualType T,
                                                   Expr **Args,
                                                   unsigned NumArgs,
                                    OverloadCandidateSet &CandidateSet) {
  QualType ParamTypes[2];

  // T& operator=(T&, T)
  ParamTypes[0] = S.Context.getLValueReferenceType(T);
  ParamTypes[1] = T;
  S.AddBuiltinCandidate(ParamTypes[0], ParamTypes, Args, 2, CandidateSet,
                        /*IsAssignmentOperator=*/true);

  if (!S.Context.getCanonicalType(T).isVolatileQualified()) {
    // volatile T& operator=(volatile T&, T)
    ParamTypes[0]
      = S.Context.getLValueReferenceType(S.Context.getVolatileType(T));
    ParamTypes[1] = T;
    S.AddBuiltinCandidate(ParamTypes[0], ParamTypes, Args, 2, CandidateSet,
                          /*IsAssignmentOperator=*/true);
  }
}

/// CollectVRQualifiers - This routine returns Volatile/Restrict qualifiers,
/// if any, found in visible type conversion functions found in ArgExpr's type.
static  Qualifiers CollectVRQualifiers(ASTContext &Context, Expr* ArgExpr) {
    Qualifiers VRQuals;
    const RecordType *TyRec;
    if (const MemberPointerType *RHSMPType =
        ArgExpr->getType()->getAs<MemberPointerType>())
      TyRec = cast<RecordType>(RHSMPType->getClass());
    else
      TyRec = ArgExpr->getType()->getAs<RecordType>();
    if (!TyRec) {
      // Just to be safe, assume the worst case.
      VRQuals.addVolatile();
      VRQuals.addRestrict();
      return VRQuals;
    }
    
    CXXRecordDecl *ClassDecl = cast<CXXRecordDecl>(TyRec->getDecl());
    const UnresolvedSet *Conversions =
      ClassDecl->getVisibleConversionFunctions();
    
    for (UnresolvedSet::iterator I = Conversions->begin(),
           E = Conversions->end(); I != E; ++I) {
      if (CXXConversionDecl *Conv = dyn_cast<CXXConversionDecl>(*I)) {
        QualType CanTy = Context.getCanonicalType(Conv->getConversionType());
        if (const ReferenceType *ResTypeRef = CanTy->getAs<ReferenceType>())
          CanTy = ResTypeRef->getPointeeType();
        // Need to go down the pointer/mempointer chain and add qualifiers
        // as see them.
        bool done = false;
        while (!done) {
          if (const PointerType *ResTypePtr = CanTy->getAs<PointerType>())
            CanTy = ResTypePtr->getPointeeType();
          else if (const MemberPointerType *ResTypeMPtr = 
                CanTy->getAs<MemberPointerType>())
            CanTy = ResTypeMPtr->getPointeeType();
          else
            done = true;
          if (CanTy.isVolatileQualified())
            VRQuals.addVolatile();
          if (CanTy.isRestrictQualified())
            VRQuals.addRestrict();
          if (VRQuals.hasRestrict() && VRQuals.hasVolatile())
            return VRQuals;
        }
      }
    }
    return VRQuals;
}
  
/// AddBuiltinOperatorCandidates - Add the appropriate built-in
/// operator overloads to the candidate set (C++ [over.built]), based
/// on the operator @p Op and the arguments given. For example, if the
/// operator is a binary '+', this routine might add "int
/// operator+(int, int)" to cover integer addition.
void
Sema::AddBuiltinOperatorCandidates(OverloadedOperatorKind Op,
                                   SourceLocation OpLoc,
                                   Expr **Args, unsigned NumArgs,
                                   OverloadCandidateSet& CandidateSet) {
  // The set of "promoted arithmetic types", which are the arithmetic
  // types are that preserved by promotion (C++ [over.built]p2). Note
  // that the first few of these types are the promoted integral
  // types; these types need to be first.
  // FIXME: What about complex?
  const unsigned FirstIntegralType = 0;
  const unsigned LastIntegralType = 13;
  const unsigned FirstPromotedIntegralType = 7,
                 LastPromotedIntegralType = 13;
  const unsigned FirstPromotedArithmeticType = 7,
                 LastPromotedArithmeticType = 16;
  const unsigned NumArithmeticTypes = 16;
  QualType ArithmeticTypes[NumArithmeticTypes] = {
    Context.BoolTy, Context.CharTy, Context.WCharTy,
// FIXME:   Context.Char16Ty, Context.Char32Ty,
    Context.SignedCharTy, Context.ShortTy,
    Context.UnsignedCharTy, Context.UnsignedShortTy,
    Context.IntTy, Context.LongTy, Context.LongLongTy,
    Context.UnsignedIntTy, Context.UnsignedLongTy, Context.UnsignedLongLongTy,
    Context.FloatTy, Context.DoubleTy, Context.LongDoubleTy
  };
  assert(ArithmeticTypes[FirstPromotedIntegralType] == Context.IntTy &&
         "Invalid first promoted integral type");
  assert(ArithmeticTypes[LastPromotedIntegralType - 1] 
           == Context.UnsignedLongLongTy &&
         "Invalid last promoted integral type");
  assert(ArithmeticTypes[FirstPromotedArithmeticType] == Context.IntTy &&
         "Invalid first promoted arithmetic type");
  assert(ArithmeticTypes[LastPromotedArithmeticType - 1] 
            == Context.LongDoubleTy &&
         "Invalid last promoted arithmetic type");
         
  // Find all of the types that the arguments can convert to, but only
  // if the operator we're looking at has built-in operator candidates
  // that make use of these types.
  Qualifiers VisibleTypeConversionsQuals;
  VisibleTypeConversionsQuals.addConst();
  for (unsigned ArgIdx = 0; ArgIdx < NumArgs; ++ArgIdx)
    VisibleTypeConversionsQuals += CollectVRQualifiers(Context, Args[ArgIdx]);
  
  BuiltinCandidateTypeSet CandidateTypes(*this);
  if (Op == OO_Less || Op == OO_Greater || Op == OO_LessEqual ||
      Op == OO_GreaterEqual || Op == OO_EqualEqual || Op == OO_ExclaimEqual ||
      Op == OO_Plus || (Op == OO_Minus && NumArgs == 2) || Op == OO_Equal ||
      Op == OO_PlusEqual || Op == OO_MinusEqual || Op == OO_Subscript ||
      Op == OO_ArrowStar || Op == OO_PlusPlus || Op == OO_MinusMinus ||
      (Op == OO_Star && NumArgs == 1) || Op == OO_Conditional) {
    for (unsigned ArgIdx = 0; ArgIdx < NumArgs; ++ArgIdx)
      CandidateTypes.AddTypesConvertedFrom(Args[ArgIdx]->getType(),
                                           OpLoc,
                                           true,
                                           (Op == OO_Exclaim ||
                                            Op == OO_AmpAmp ||
                                            Op == OO_PipePipe),
                                           VisibleTypeConversionsQuals);
  }

  bool isComparison = false;
  switch (Op) {
  case OO_None:
  case NUM_OVERLOADED_OPERATORS:
    assert(false && "Expected an overloaded operator");
    break;

  case OO_Star: // '*' is either unary or binary
    if (NumArgs == 1)
      goto UnaryStar;
    else
      goto BinaryStar;
    break;

  case OO_Plus: // '+' is either unary or binary
    if (NumArgs == 1)
      goto UnaryPlus;
    else
      goto BinaryPlus;
    break;

  case OO_Minus: // '-' is either unary or binary
    if (NumArgs == 1)
      goto UnaryMinus;
    else
      goto BinaryMinus;
    break;

  case OO_Amp: // '&' is either unary or binary
    if (NumArgs == 1)
      goto UnaryAmp;
    else
      goto BinaryAmp;

  case OO_PlusPlus:
  case OO_MinusMinus:
    // C++ [over.built]p3:
    //
    //   For every pair (T, VQ), where T is an arithmetic type, and VQ
    //   is either volatile or empty, there exist candidate operator
    //   functions of the form
    //
    //       VQ T&      operator++(VQ T&);
    //       T          operator++(VQ T&, int);
    //
    // C++ [over.built]p4:
    //
    //   For every pair (T, VQ), where T is an arithmetic type other
    //   than bool, and VQ is either volatile or empty, there exist
    //   candidate operator functions of the form
    //
    //       VQ T&      operator--(VQ T&);
    //       T          operator--(VQ T&, int);
    for (unsigned Arith = (Op == OO_PlusPlus? 0 : 1);
         Arith < NumArithmeticTypes; ++Arith) {
      QualType ArithTy = ArithmeticTypes[Arith];
      QualType ParamTypes[2]
        = { Context.getLValueReferenceType(ArithTy), Context.IntTy };

      // Non-volatile version.
      if (NumArgs == 1)
        AddBuiltinCandidate(ParamTypes[0], ParamTypes, Args, 1, CandidateSet);
      else
        AddBuiltinCandidate(ArithTy, ParamTypes, Args, 2, CandidateSet);
      // heuristic to reduce number of builtin candidates in the set.
      // Add volatile version only if there are conversions to a volatile type.
      if (VisibleTypeConversionsQuals.hasVolatile()) {
        // Volatile version
        ParamTypes[0]
          = Context.getLValueReferenceType(Context.getVolatileType(ArithTy));
        if (NumArgs == 1)
          AddBuiltinCandidate(ParamTypes[0], ParamTypes, Args, 1, CandidateSet);
        else
          AddBuiltinCandidate(ArithTy, ParamTypes, Args, 2, CandidateSet);
      }
    }

    // C++ [over.built]p5:
    //
    //   For every pair (T, VQ), where T is a cv-qualified or
    //   cv-unqualified object type, and VQ is either volatile or
    //   empty, there exist candidate operator functions of the form
    //
    //       T*VQ&      operator++(T*VQ&);
    //       T*VQ&      operator--(T*VQ&);
    //       T*         operator++(T*VQ&, int);
    //       T*         operator--(T*VQ&, int);
    for (BuiltinCandidateTypeSet::iterator Ptr = CandidateTypes.pointer_begin();
         Ptr != CandidateTypes.pointer_end(); ++Ptr) {
      // Skip pointer types that aren't pointers to object types.
      if (!(*Ptr)->getAs<PointerType>()->getPointeeType()->isObjectType())
        continue;

      QualType ParamTypes[2] = {
        Context.getLValueReferenceType(*Ptr), Context.IntTy
      };

      // Without volatile
      if (NumArgs == 1)
        AddBuiltinCandidate(ParamTypes[0], ParamTypes, Args, 1, CandidateSet);
      else
        AddBuiltinCandidate(*Ptr, ParamTypes, Args, 2, CandidateSet);

      if (!Context.getCanonicalType(*Ptr).isVolatileQualified() &&
          VisibleTypeConversionsQuals.hasVolatile()) {
        // With volatile
        ParamTypes[0]
          = Context.getLValueReferenceType(Context.getVolatileType(*Ptr));
        if (NumArgs == 1)
          AddBuiltinCandidate(ParamTypes[0], ParamTypes, Args, 1, CandidateSet);
        else
          AddBuiltinCandidate(*Ptr, ParamTypes, Args, 2, CandidateSet);
      }
    }
    break;

  UnaryStar:
    // C++ [over.built]p6:
    //   For every cv-qualified or cv-unqualified object type T, there
    //   exist candidate operator functions of the form
    //
    //       T&         operator*(T*);
    //
    // C++ [over.built]p7:
    //   For every function type T, there exist candidate operator
    //   functions of the form
    //       T&         operator*(T*);
    for (BuiltinCandidateTypeSet::iterator Ptr = CandidateTypes.pointer_begin();
         Ptr != CandidateTypes.pointer_end(); ++Ptr) {
      QualType ParamTy = *Ptr;
      QualType PointeeTy = ParamTy->getAs<PointerType>()->getPointeeType();
      AddBuiltinCandidate(Context.getLValueReferenceType(PointeeTy),
                          &ParamTy, Args, 1, CandidateSet);
    }
    break;

  UnaryPlus:
    // C++ [over.built]p8:
    //   For every type T, there exist candidate operator functions of
    //   the form
    //
    //       T*         operator+(T*);
    for (BuiltinCandidateTypeSet::iterator Ptr = CandidateTypes.pointer_begin();
         Ptr != CandidateTypes.pointer_end(); ++Ptr) {
      QualType ParamTy = *Ptr;
      AddBuiltinCandidate(ParamTy, &ParamTy, Args, 1, CandidateSet);
    }

    // Fall through

  UnaryMinus:
    // C++ [over.built]p9:
    //  For every promoted arithmetic type T, there exist candidate
    //  operator functions of the form
    //
    //       T         operator+(T);
    //       T         operator-(T);
    for (unsigned Arith = FirstPromotedArithmeticType;
         Arith < LastPromotedArithmeticType; ++Arith) {
      QualType ArithTy = ArithmeticTypes[Arith];
      AddBuiltinCandidate(ArithTy, &ArithTy, Args, 1, CandidateSet);
    }
    break;

  case OO_Tilde:
    // C++ [over.built]p10:
    //   For every promoted integral type T, there exist candidate
    //   operator functions of the form
    //
    //        T         operator~(T);
    for (unsigned Int = FirstPromotedIntegralType;
         Int < LastPromotedIntegralType; ++Int) {
      QualType IntTy = ArithmeticTypes[Int];
      AddBuiltinCandidate(IntTy, &IntTy, Args, 1, CandidateSet);
    }
    break;

  case OO_New:
  case OO_Delete:
  case OO_Array_New:
  case OO_Array_Delete:
  case OO_Call:
    assert(false && "Special operators don't use AddBuiltinOperatorCandidates");
    break;

  case OO_Comma:
  UnaryAmp:
  case OO_Arrow:
    // C++ [over.match.oper]p3:
    //   -- For the operator ',', the unary operator '&', or the
    //      operator '->', the built-in candidates set is empty.
    break;

  case OO_EqualEqual:
  case OO_ExclaimEqual:
    // C++ [over.match.oper]p16:
    //   For every pointer to member type T, there exist candidate operator
    //   functions of the form
    //
    //        bool operator==(T,T);
    //        bool operator!=(T,T);
    for (BuiltinCandidateTypeSet::iterator
           MemPtr = CandidateTypes.member_pointer_begin(),
           MemPtrEnd = CandidateTypes.member_pointer_end();
         MemPtr != MemPtrEnd;
         ++MemPtr) {
      QualType ParamTypes[2] = { *MemPtr, *MemPtr };
      AddBuiltinCandidate(Context.BoolTy, ParamTypes, Args, 2, CandidateSet);
    }

    // Fall through

  case OO_Less:
  case OO_Greater:
  case OO_LessEqual:
  case OO_GreaterEqual:
    // C++ [over.built]p15:
    //
    //   For every pointer or enumeration type T, there exist
    //   candidate operator functions of the form
    //
    //        bool       operator<(T, T);
    //        bool       operator>(T, T);
    //        bool       operator<=(T, T);
    //        bool       operator>=(T, T);
    //        bool       operator==(T, T);
    //        bool       operator!=(T, T);
    for (BuiltinCandidateTypeSet::iterator Ptr = CandidateTypes.pointer_begin();
         Ptr != CandidateTypes.pointer_end(); ++Ptr) {
      QualType ParamTypes[2] = { *Ptr, *Ptr };
      AddBuiltinCandidate(Context.BoolTy, ParamTypes, Args, 2, CandidateSet);
    }
    for (BuiltinCandidateTypeSet::iterator Enum
           = CandidateTypes.enumeration_begin();
         Enum != CandidateTypes.enumeration_end(); ++Enum) {
      QualType ParamTypes[2] = { *Enum, *Enum };
      AddBuiltinCandidate(Context.BoolTy, ParamTypes, Args, 2, CandidateSet);
    }

    // Fall through.
    isComparison = true;

  BinaryPlus:
  BinaryMinus:
    if (!isComparison) {
      // We didn't fall through, so we must have OO_Plus or OO_Minus.

      // C++ [over.built]p13:
      //
      //   For every cv-qualified or cv-unqualified object type T
      //   there exist candidate operator functions of the form
      //
      //      T*         operator+(T*, ptrdiff_t);
      //      T&         operator[](T*, ptrdiff_t);    [BELOW]
      //      T*         operator-(T*, ptrdiff_t);
      //      T*         operator+(ptrdiff_t, T*);
      //      T&         operator[](ptrdiff_t, T*);    [BELOW]
      //
      // C++ [over.built]p14:
      //
      //   For every T, where T is a pointer to object type, there
      //   exist candidate operator functions of the form
      //
      //      ptrdiff_t  operator-(T, T);
      for (BuiltinCandidateTypeSet::iterator Ptr
             = CandidateTypes.pointer_begin();
           Ptr != CandidateTypes.pointer_end(); ++Ptr) {
        QualType ParamTypes[2] = { *Ptr, Context.getPointerDiffType() };

        // operator+(T*, ptrdiff_t) or operator-(T*, ptrdiff_t)
        AddBuiltinCandidate(*Ptr, ParamTypes, Args, 2, CandidateSet);

        if (Op == OO_Plus) {
          // T* operator+(ptrdiff_t, T*);
          ParamTypes[0] = ParamTypes[1];
          ParamTypes[1] = *Ptr;
          AddBuiltinCandidate(*Ptr, ParamTypes, Args, 2, CandidateSet);
        } else {
          // ptrdiff_t operator-(T, T);
          ParamTypes[1] = *Ptr;
          AddBuiltinCandidate(Context.getPointerDiffType(), ParamTypes,
                              Args, 2, CandidateSet);
        }
      }
    }
    // Fall through

  case OO_Slash:
  BinaryStar:
  Conditional:
    // C++ [over.built]p12:
    //
    //   For every pair of promoted arithmetic types L and R, there
    //   exist candidate operator functions of the form
    //
    //        LR         operator*(L, R);
    //        LR         operator/(L, R);
    //        LR         operator+(L, R);
    //        LR         operator-(L, R);
    //        bool       operator<(L, R);
    //        bool       operator>(L, R);
    //        bool       operator<=(L, R);
    //        bool       operator>=(L, R);
    //        bool       operator==(L, R);
    //        bool       operator!=(L, R);
    //
    //   where LR is the result of the usual arithmetic conversions
    //   between types L and R.
    //
    // C++ [over.built]p24:
    //
    //   For every pair of promoted arithmetic types L and R, there exist
    //   candidate operator functions of the form
    //
    //        LR       operator?(bool, L, R);
    //
    //   where LR is the result of the usual arithmetic conversions
    //   between types L and R.
    // Our candidates ignore the first parameter.
    for (unsigned Left = FirstPromotedArithmeticType;
         Left < LastPromotedArithmeticType; ++Left) {
      for (unsigned Right = FirstPromotedArithmeticType;
           Right < LastPromotedArithmeticType; ++Right) {
        QualType LandR[2] = { ArithmeticTypes[Left], ArithmeticTypes[Right] };
        QualType Result
          = isComparison
          ? Context.BoolTy
          : Context.UsualArithmeticConversionsType(LandR[0], LandR[1]);
        AddBuiltinCandidate(Result, LandR, Args, 2, CandidateSet);
      }
    }
    break;

  case OO_Percent:
  BinaryAmp:
  case OO_Caret:
  case OO_Pipe:
  case OO_LessLess:
  case OO_GreaterGreater:
    // C++ [over.built]p17:
    //
    //   For every pair of promoted integral types L and R, there
    //   exist candidate operator functions of the form
    //
    //      LR         operator%(L, R);
    //      LR         operator&(L, R);
    //      LR         operator^(L, R);
    //      LR         operator|(L, R);
    //      L          operator<<(L, R);
    //      L          operator>>(L, R);
    //
    //   where LR is the result of the usual arithmetic conversions
    //   between types L and R.
    for (unsigned Left = FirstPromotedIntegralType;
         Left < LastPromotedIntegralType; ++Left) {
      for (unsigned Right = FirstPromotedIntegralType;
           Right < LastPromotedIntegralType; ++Right) {
        QualType LandR[2] = { ArithmeticTypes[Left], ArithmeticTypes[Right] };
        QualType Result = (Op == OO_LessLess || Op == OO_GreaterGreater)
            ? LandR[0]
            : Context.UsualArithmeticConversionsType(LandR[0], LandR[1]);
        AddBuiltinCandidate(Result, LandR, Args, 2, CandidateSet);
      }
    }
    break;

  case OO_Equal:
    // C++ [over.built]p20:
    //
    //   For every pair (T, VQ), where T is an enumeration or
    //   pointer to member type and VQ is either volatile or
    //   empty, there exist candidate operator functions of the form
    //
    //        VQ T&      operator=(VQ T&, T);
    for (BuiltinCandidateTypeSet::iterator
           Enum = CandidateTypes.enumeration_begin(),
           EnumEnd = CandidateTypes.enumeration_end();
         Enum != EnumEnd; ++Enum)
      AddBuiltinAssignmentOperatorCandidates(*this, *Enum, Args, 2,
                                             CandidateSet);
    for (BuiltinCandidateTypeSet::iterator
           MemPtr = CandidateTypes.member_pointer_begin(),
         MemPtrEnd = CandidateTypes.member_pointer_end();
         MemPtr != MemPtrEnd; ++MemPtr)
      AddBuiltinAssignmentOperatorCandidates(*this, *MemPtr, Args, 2,
                                             CandidateSet);
      // Fall through.

  case OO_PlusEqual:
  case OO_MinusEqual:
    // C++ [over.built]p19:
    //
    //   For every pair (T, VQ), where T is any type and VQ is either
    //   volatile or empty, there exist candidate operator functions
    //   of the form
    //
    //        T*VQ&      operator=(T*VQ&, T*);
    //
    // C++ [over.built]p21:
    //
    //   For every pair (T, VQ), where T is a cv-qualified or
    //   cv-unqualified object type and VQ is either volatile or
    //   empty, there exist candidate operator functions of the form
    //
    //        T*VQ&      operator+=(T*VQ&, ptrdiff_t);
    //        T*VQ&      operator-=(T*VQ&, ptrdiff_t);
    for (BuiltinCandidateTypeSet::iterator Ptr = CandidateTypes.pointer_begin();
         Ptr != CandidateTypes.pointer_end(); ++Ptr) {
      QualType ParamTypes[2];
      ParamTypes[1] = (Op == OO_Equal)? *Ptr : Context.getPointerDiffType();

      // non-volatile version
      ParamTypes[0] = Context.getLValueReferenceType(*Ptr);
      AddBuiltinCandidate(ParamTypes[0], ParamTypes, Args, 2, CandidateSet,
                          /*IsAssigmentOperator=*/Op == OO_Equal);

      if (!Context.getCanonicalType(*Ptr).isVolatileQualified() &&
          VisibleTypeConversionsQuals.hasVolatile()) {
        // volatile version
        ParamTypes[0]
          = Context.getLValueReferenceType(Context.getVolatileType(*Ptr));
        AddBuiltinCandidate(ParamTypes[0], ParamTypes, Args, 2, CandidateSet,
                            /*IsAssigmentOperator=*/Op == OO_Equal);
      }
    }
    // Fall through.

  case OO_StarEqual:
  case OO_SlashEqual:
    // C++ [over.built]p18:
    //
    //   For every triple (L, VQ, R), where L is an arithmetic type,
    //   VQ is either volatile or empty, and R is a promoted
    //   arithmetic type, there exist candidate operator functions of
    //   the form
    //
    //        VQ L&      operator=(VQ L&, R);
    //        VQ L&      operator*=(VQ L&, R);
    //        VQ L&      operator/=(VQ L&, R);
    //        VQ L&      operator+=(VQ L&, R);
    //        VQ L&      operator-=(VQ L&, R);
    for (unsigned Left = 0; Left < NumArithmeticTypes; ++Left) {
      for (unsigned Right = FirstPromotedArithmeticType;
           Right < LastPromotedArithmeticType; ++Right) {
        QualType ParamTypes[2];
        ParamTypes[1] = ArithmeticTypes[Right];

        // Add this built-in operator as a candidate (VQ is empty).
        ParamTypes[0] = Context.getLValueReferenceType(ArithmeticTypes[Left]);
        AddBuiltinCandidate(ParamTypes[0], ParamTypes, Args, 2, CandidateSet,
                            /*IsAssigmentOperator=*/Op == OO_Equal);

        // Add this built-in operator as a candidate (VQ is 'volatile').
        if (VisibleTypeConversionsQuals.hasVolatile()) {
          ParamTypes[0] = Context.getVolatileType(ArithmeticTypes[Left]);
          ParamTypes[0] = Context.getLValueReferenceType(ParamTypes[0]);
          AddBuiltinCandidate(ParamTypes[0], ParamTypes, Args, 2, CandidateSet,
                              /*IsAssigmentOperator=*/Op == OO_Equal);
        }
      }
    }
    break;

  case OO_PercentEqual:
  case OO_LessLessEqual:
  case OO_GreaterGreaterEqual:
  case OO_AmpEqual:
  case OO_CaretEqual:
  case OO_PipeEqual:
    // C++ [over.built]p22:
    //
    //   For every triple (L, VQ, R), where L is an integral type, VQ
    //   is either volatile or empty, and R is a promoted integral
    //   type, there exist candidate operator functions of the form
    //
    //        VQ L&       operator%=(VQ L&, R);
    //        VQ L&       operator<<=(VQ L&, R);
    //        VQ L&       operator>>=(VQ L&, R);
    //        VQ L&       operator&=(VQ L&, R);
    //        VQ L&       operator^=(VQ L&, R);
    //        VQ L&       operator|=(VQ L&, R);
    for (unsigned Left = FirstIntegralType; Left < LastIntegralType; ++Left) {
      for (unsigned Right = FirstPromotedIntegralType;
           Right < LastPromotedIntegralType; ++Right) {
        QualType ParamTypes[2];
        ParamTypes[1] = ArithmeticTypes[Right];

        // Add this built-in operator as a candidate (VQ is empty).
        ParamTypes[0] = Context.getLValueReferenceType(ArithmeticTypes[Left]);
        AddBuiltinCandidate(ParamTypes[0], ParamTypes, Args, 2, CandidateSet);
        if (VisibleTypeConversionsQuals.hasVolatile()) {
          // Add this built-in operator as a candidate (VQ is 'volatile').
          ParamTypes[0] = ArithmeticTypes[Left];
          ParamTypes[0] = Context.getVolatileType(ParamTypes[0]);
          ParamTypes[0] = Context.getLValueReferenceType(ParamTypes[0]);
          AddBuiltinCandidate(ParamTypes[0], ParamTypes, Args, 2, CandidateSet);
        }
      }
    }
    break;

  case OO_Exclaim: {
    // C++ [over.operator]p23:
    //
    //   There also exist candidate operator functions of the form
    //
    //        bool        operator!(bool);
    //        bool        operator&&(bool, bool);     [BELOW]
    //        bool        operator||(bool, bool);     [BELOW]
    QualType ParamTy = Context.BoolTy;
    AddBuiltinCandidate(ParamTy, &ParamTy, Args, 1, CandidateSet,
                        /*IsAssignmentOperator=*/false,
                        /*NumContextualBoolArguments=*/1);
    break;
  }

  case OO_AmpAmp:
  case OO_PipePipe: {
    // C++ [over.operator]p23:
    //
    //   There also exist candidate operator functions of the form
    //
    //        bool        operator!(bool);            [ABOVE]
    //        bool        operator&&(bool, bool);
    //        bool        operator||(bool, bool);
    QualType ParamTypes[2] = { Context.BoolTy, Context.BoolTy };
    AddBuiltinCandidate(Context.BoolTy, ParamTypes, Args, 2, CandidateSet,
                        /*IsAssignmentOperator=*/false,
                        /*NumContextualBoolArguments=*/2);
    break;
  }

  case OO_Subscript:
    // C++ [over.built]p13:
    //
    //   For every cv-qualified or cv-unqualified object type T there
    //   exist candidate operator functions of the form
    //
    //        T*         operator+(T*, ptrdiff_t);     [ABOVE]
    //        T&         operator[](T*, ptrdiff_t);
    //        T*         operator-(T*, ptrdiff_t);     [ABOVE]
    //        T*         operator+(ptrdiff_t, T*);     [ABOVE]
    //        T&         operator[](ptrdiff_t, T*);
    for (BuiltinCandidateTypeSet::iterator Ptr = CandidateTypes.pointer_begin();
         Ptr != CandidateTypes.pointer_end(); ++Ptr) {
      QualType ParamTypes[2] = { *Ptr, Context.getPointerDiffType() };
      QualType PointeeType = (*Ptr)->getAs<PointerType>()->getPointeeType();
      QualType ResultTy = Context.getLValueReferenceType(PointeeType);

      // T& operator[](T*, ptrdiff_t)
      AddBuiltinCandidate(ResultTy, ParamTypes, Args, 2, CandidateSet);

      // T& operator[](ptrdiff_t, T*);
      ParamTypes[0] = ParamTypes[1];
      ParamTypes[1] = *Ptr;
      AddBuiltinCandidate(ResultTy, ParamTypes, Args, 2, CandidateSet);
    }
    break;

  case OO_ArrowStar:
    // C++ [over.built]p11:
    //    For every quintuple (C1, C2, T, CV1, CV2), where C2 is a class type, 
    //    C1 is the same type as C2 or is a derived class of C2, T is an object 
    //    type or a function type, and CV1 and CV2 are cv-qualifier-seqs, 
    //    there exist candidate operator functions of the form 
    //    CV12 T& operator->*(CV1 C1*, CV2 T C2::*); 
    //    where CV12 is the union of CV1 and CV2.
    {
      for (BuiltinCandidateTypeSet::iterator Ptr = 
             CandidateTypes.pointer_begin();
           Ptr != CandidateTypes.pointer_end(); ++Ptr) {
        QualType C1Ty = (*Ptr);
        QualType C1;
        QualifierCollector Q1;
        if (const PointerType *PointerTy = C1Ty->getAs<PointerType>()) {
          C1 = QualType(Q1.strip(PointerTy->getPointeeType()), 0);
          if (!isa<RecordType>(C1))
            continue;
          // heuristic to reduce number of builtin candidates in the set.
          // Add volatile/restrict version only if there are conversions to a
          // volatile/restrict type.
          if (!VisibleTypeConversionsQuals.hasVolatile() && Q1.hasVolatile())
            continue;
          if (!VisibleTypeConversionsQuals.hasRestrict() && Q1.hasRestrict())
            continue;
        }
        for (BuiltinCandidateTypeSet::iterator
             MemPtr = CandidateTypes.member_pointer_begin(),
             MemPtrEnd = CandidateTypes.member_pointer_end();
             MemPtr != MemPtrEnd; ++MemPtr) {
          const MemberPointerType *mptr = cast<MemberPointerType>(*MemPtr);
          QualType C2 = QualType(mptr->getClass(), 0);
          C2 = C2.getUnqualifiedType();
          if (C1 != C2 && !IsDerivedFrom(C1, C2))
            break;
          QualType ParamTypes[2] = { *Ptr, *MemPtr };
          // build CV12 T&
          QualType T = mptr->getPointeeType();
          if (!VisibleTypeConversionsQuals.hasVolatile() && 
              T.isVolatileQualified())
            continue;
          if (!VisibleTypeConversionsQuals.hasRestrict() && 
              T.isRestrictQualified())
            continue;
          T = Q1.apply(T);
          QualType ResultTy = Context.getLValueReferenceType(T);
          AddBuiltinCandidate(ResultTy, ParamTypes, Args, 2, CandidateSet);
        }
      }
    }
    break;

  case OO_Conditional:
    // Note that we don't consider the first argument, since it has been
    // contextually converted to bool long ago. The candidates below are
    // therefore added as binary.
    //
    // C++ [over.built]p24:
    //   For every type T, where T is a pointer or pointer-to-member type,
    //   there exist candidate operator functions of the form
    //
    //        T        operator?(bool, T, T);
    //
    for (BuiltinCandidateTypeSet::iterator Ptr = CandidateTypes.pointer_begin(),
         E = CandidateTypes.pointer_end(); Ptr != E; ++Ptr) {
      QualType ParamTypes[2] = { *Ptr, *Ptr };
      AddBuiltinCandidate(*Ptr, ParamTypes, Args, 2, CandidateSet);
    }
    for (BuiltinCandidateTypeSet::iterator Ptr =
           CandidateTypes.member_pointer_begin(),
         E = CandidateTypes.member_pointer_end(); Ptr != E; ++Ptr) {
      QualType ParamTypes[2] = { *Ptr, *Ptr };
      AddBuiltinCandidate(*Ptr, ParamTypes, Args, 2, CandidateSet);
    }
    goto Conditional;
  }
}

/// \brief Add function candidates found via argument-dependent lookup
/// to the set of overloading candidates.
///
/// This routine performs argument-dependent name lookup based on the
/// given function name (which may also be an operator name) and adds
/// all of the overload candidates found by ADL to the overload
/// candidate set (C++ [basic.lookup.argdep]).
void
Sema::AddArgumentDependentLookupCandidates(DeclarationName Name,
                                           Expr **Args, unsigned NumArgs,
                       const TemplateArgumentListInfo *ExplicitTemplateArgs,
                                           OverloadCandidateSet& CandidateSet,
                                           bool PartialOverloading) {
  FunctionSet Functions;

  // FIXME: Should we be trafficking in canonical function decls throughout?
  
  // Record all of the function candidates that we've already
  // added to the overload set, so that we don't add those same
  // candidates a second time.
  for (OverloadCandidateSet::iterator Cand = CandidateSet.begin(),
                                   CandEnd = CandidateSet.end();
       Cand != CandEnd; ++Cand)
    if (Cand->Function) {
      Functions.insert(Cand->Function);
      if (FunctionTemplateDecl *FunTmpl = Cand->Function->getPrimaryTemplate())
        Functions.insert(FunTmpl);
    }

  // FIXME: Pass in the explicit template arguments?
  ArgumentDependentLookup(Name, /*Operator*/false, Args, NumArgs, Functions);

  // Erase all of the candidates we already knew about.
  // FIXME: This is suboptimal. Is there a better way?
  for (OverloadCandidateSet::iterator Cand = CandidateSet.begin(),
                                   CandEnd = CandidateSet.end();
       Cand != CandEnd; ++Cand)
    if (Cand->Function) {
      Functions.erase(Cand->Function);
      if (FunctionTemplateDecl *FunTmpl = Cand->Function->getPrimaryTemplate())
        Functions.erase(FunTmpl);
    }

  // For each of the ADL candidates we found, add it to the overload
  // set.
  for (FunctionSet::iterator Func = Functions.begin(),
                          FuncEnd = Functions.end();
       Func != FuncEnd; ++Func) {
    if (FunctionDecl *FD = dyn_cast<FunctionDecl>(*Func)) {
      if (ExplicitTemplateArgs)
        continue;
      
      AddOverloadCandidate(FD, Args, NumArgs, CandidateSet,
                           false, false, PartialOverloading);
    } else
      AddTemplateOverloadCandidate(cast<FunctionTemplateDecl>(*Func),
                                   ExplicitTemplateArgs,
                                   Args, NumArgs, CandidateSet);
  }
}

/// isBetterOverloadCandidate - Determines whether the first overload
/// candidate is a better candidate than the second (C++ 13.3.3p1).
bool
Sema::isBetterOverloadCandidate(const OverloadCandidate& Cand1,
                                const OverloadCandidate& Cand2) {
  // Define viable functions to be better candidates than non-viable
  // functions.
  if (!Cand2.Viable)
    return Cand1.Viable;
  else if (!Cand1.Viable)
    return false;

  // C++ [over.match.best]p1:
  //
  //   -- if F is a static member function, ICS1(F) is defined such
  //      that ICS1(F) is neither better nor worse than ICS1(G) for
  //      any function G, and, symmetrically, ICS1(G) is neither
  //      better nor worse than ICS1(F).
  unsigned StartArg = 0;
  if (Cand1.IgnoreObjectArgument || Cand2.IgnoreObjectArgument)
    StartArg = 1;

  // C++ [over.match.best]p1:
  //   A viable function F1 is defined to be a better function than another
  //   viable function F2 if for all arguments i, ICSi(F1) is not a worse
  //   conversion sequence than ICSi(F2), and then...
  unsigned NumArgs = Cand1.Conversions.size();
  assert(Cand2.Conversions.size() == NumArgs && "Overload candidate mismatch");
  bool HasBetterConversion = false;
  for (unsigned ArgIdx = StartArg; ArgIdx < NumArgs; ++ArgIdx) {
    switch (CompareImplicitConversionSequences(Cand1.Conversions[ArgIdx],
                                               Cand2.Conversions[ArgIdx])) {
    case ImplicitConversionSequence::Better:
      // Cand1 has a better conversion sequence.
      HasBetterConversion = true;
      break;

    case ImplicitConversionSequence::Worse:
      // Cand1 can't be better than Cand2.
      return false;

    case ImplicitConversionSequence::Indistinguishable:
      // Do nothing.
      break;
    }
  }

  //    -- for some argument j, ICSj(F1) is a better conversion sequence than
  //       ICSj(F2), or, if not that,
  if (HasBetterConversion)
    return true;

  //     - F1 is a non-template function and F2 is a function template
  //       specialization, or, if not that,
  if (Cand1.Function && !Cand1.Function->getPrimaryTemplate() &&
      Cand2.Function && Cand2.Function->getPrimaryTemplate())
    return true;

  //   -- F1 and F2 are function template specializations, and the function
  //      template for F1 is more specialized than the template for F2
  //      according to the partial ordering rules described in 14.5.5.2, or,
  //      if not that,
  if (Cand1.Function && Cand1.Function->getPrimaryTemplate() &&
      Cand2.Function && Cand2.Function->getPrimaryTemplate())
    if (FunctionTemplateDecl *BetterTemplate
          = getMoreSpecializedTemplate(Cand1.Function->getPrimaryTemplate(),
                                       Cand2.Function->getPrimaryTemplate(),
                       isa<CXXConversionDecl>(Cand1.Function)? TPOC_Conversion 
                                                             : TPOC_Call))
      return BetterTemplate == Cand1.Function->getPrimaryTemplate();

  //   -- the context is an initialization by user-defined conversion
  //      (see 8.5, 13.3.1.5) and the standard conversion sequence
  //      from the return type of F1 to the destination type (i.e.,
  //      the type of the entity being initialized) is a better
  //      conversion sequence than the standard conversion sequence
  //      from the return type of F2 to the destination type.
  if (Cand1.Function && Cand2.Function &&
      isa<CXXConversionDecl>(Cand1.Function) &&
      isa<CXXConversionDecl>(Cand2.Function)) {
    switch (CompareStandardConversionSequences(Cand1.FinalConversion,
                                               Cand2.FinalConversion)) {
    case ImplicitConversionSequence::Better:
      // Cand1 has a better conversion sequence.
      return true;

    case ImplicitConversionSequence::Worse:
      // Cand1 can't be better than Cand2.
      return false;

    case ImplicitConversionSequence::Indistinguishable:
      // Do nothing
      break;
    }
  }

  return false;
}

/// \brief Computes the best viable function (C++ 13.3.3)
/// within an overload candidate set.
///
/// \param CandidateSet the set of candidate functions.
///
/// \param Loc the location of the function name (or operator symbol) for
/// which overload resolution occurs.
///
/// \param Best f overload resolution was successful or found a deleted
/// function, Best points to the candidate function found.
///
/// \returns The result of overload resolution.
OverloadingResult Sema::BestViableFunction(OverloadCandidateSet& CandidateSet,
                                           SourceLocation Loc,
                                        OverloadCandidateSet::iterator& Best) {
  // Find the best viable function.
  Best = CandidateSet.end();
  for (OverloadCandidateSet::iterator Cand = CandidateSet.begin();
       Cand != CandidateSet.end(); ++Cand) {
    if (Cand->Viable) {
      if (Best == CandidateSet.end() || isBetterOverloadCandidate(*Cand, *Best))
        Best = Cand;
    }
  }

  // If we didn't find any viable functions, abort.
  if (Best == CandidateSet.end())
    return OR_No_Viable_Function;

  // Make sure that this function is better than every other viable
  // function. If not, we have an ambiguity.
  for (OverloadCandidateSet::iterator Cand = CandidateSet.begin();
       Cand != CandidateSet.end(); ++Cand) {
    if (Cand->Viable &&
        Cand != Best &&
        !isBetterOverloadCandidate(*Best, *Cand)) {
      Best = CandidateSet.end();
      return OR_Ambiguous;
    }
  }

  // Best is the best viable function.
  if (Best->Function &&
      (Best->Function->isDeleted() ||
       Best->Function->getAttr<UnavailableAttr>()))
    return OR_Deleted;

  // C++ [basic.def.odr]p2:
  //   An overloaded function is used if it is selected by overload resolution
  //   when referred to from a potentially-evaluated expression. [Note: this
  //   covers calls to named functions (5.2.2), operator overloading
  //   (clause 13), user-defined conversions (12.3.2), allocation function for
  //   placement new (5.3.4), as well as non-default initialization (8.5).
  if (Best->Function)
    MarkDeclarationReferenced(Loc, Best->Function);
  return OR_Success;
}

/// PrintOverloadCandidates - When overload resolution fails, prints
/// diagnostic messages containing the candidates in the candidate
/// set. If OnlyViable is true, only viable candidates will be printed.
void
Sema::PrintOverloadCandidates(OverloadCandidateSet& CandidateSet,
                              bool OnlyViable,
                              const char *Opc,
                              SourceLocation OpLoc) {
  OverloadCandidateSet::iterator Cand = CandidateSet.begin(),
                             LastCand = CandidateSet.end();
  bool Reported = false;
  for (; Cand != LastCand; ++Cand) {
    if (Cand->Viable || !OnlyViable) {
      if (Cand->Function) {
        if (Cand->Function->isDeleted() ||
            Cand->Function->getAttr<UnavailableAttr>()) {
          // Deleted or "unavailable" function.
          Diag(Cand->Function->getLocation(), diag::err_ovl_candidate_deleted)
            << Cand->Function->isDeleted();
        } else if (FunctionTemplateDecl *FunTmpl 
                     = Cand->Function->getPrimaryTemplate()) {
          // Function template specialization
          // FIXME: Give a better reason!
          Diag(Cand->Function->getLocation(), diag::err_ovl_template_candidate)
            << getTemplateArgumentBindingsText(FunTmpl->getTemplateParameters(),
                              *Cand->Function->getTemplateSpecializationArgs());
        } else {
          // Normal function
          bool errReported = false;
          if (!Cand->Viable && Cand->Conversions.size() > 0) {
            for (int i = Cand->Conversions.size()-1; i >= 0; i--) {
              const ImplicitConversionSequence &Conversion = 
                                                        Cand->Conversions[i];
              if ((Conversion.ConversionKind != 
                   ImplicitConversionSequence::BadConversion) ||
                  Conversion.ConversionFunctionSet.size() == 0)
                continue;
              Diag(Cand->Function->getLocation(), 
                   diag::err_ovl_candidate_not_viable) << (i+1);
              errReported = true;
              for (int j = Conversion.ConversionFunctionSet.size()-1; 
                   j >= 0; j--) {
                FunctionDecl *Func = Conversion.ConversionFunctionSet[j];
                Diag(Func->getLocation(), diag::err_ovl_candidate);
              }
            }
          }
          if (!errReported)
            Diag(Cand->Function->getLocation(), diag::err_ovl_candidate);
        }
      } else if (Cand->IsSurrogate) {
        // Desugar the type of the surrogate down to a function type,
        // retaining as many typedefs as possible while still showing
        // the function type (and, therefore, its parameter types).
        QualType FnType = Cand->Surrogate->getConversionType();
        bool isLValueReference = false;
        bool isRValueReference = false;
        bool isPointer = false;
        if (const LValueReferenceType *FnTypeRef =
              FnType->getAs<LValueReferenceType>()) {
          FnType = FnTypeRef->getPointeeType();
          isLValueReference = true;
        } else if (const RValueReferenceType *FnTypeRef =
                     FnType->getAs<RValueReferenceType>()) {
          FnType = FnTypeRef->getPointeeType();
          isRValueReference = true;
        }
        if (const PointerType *FnTypePtr = FnType->getAs<PointerType>()) {
          FnType = FnTypePtr->getPointeeType();
          isPointer = true;
        }
        // Desugar down to a function type.
        FnType = QualType(FnType->getAs<FunctionType>(), 0);
        // Reconstruct the pointer/reference as appropriate.
        if (isPointer) FnType = Context.getPointerType(FnType);
        if (isRValueReference) FnType = Context.getRValueReferenceType(FnType);
        if (isLValueReference) FnType = Context.getLValueReferenceType(FnType);

        Diag(Cand->Surrogate->getLocation(), diag::err_ovl_surrogate_cand)
          << FnType;
      } else if (OnlyViable) {
        assert(Cand->Conversions.size() <= 2 && 
               "builtin-binary-operator-not-binary");
        std::string TypeStr("operator");
        TypeStr += Opc;
        TypeStr += "(";
        TypeStr += Cand->BuiltinTypes.ParamTypes[0].getAsString();
        if (Cand->Conversions.size() == 1) {
          TypeStr += ")";
          Diag(OpLoc, diag::err_ovl_builtin_unary_candidate) << TypeStr;
        }
        else {
          TypeStr += ", ";
          TypeStr += Cand->BuiltinTypes.ParamTypes[1].getAsString();
          TypeStr += ")";
          Diag(OpLoc, diag::err_ovl_builtin_binary_candidate) << TypeStr;
        }
      }
      else if (!Cand->Viable && !Reported) {
        // Non-viability might be due to ambiguous user-defined conversions,
        // needed for built-in operators. Report them as well, but only once
        // as we have typically many built-in candidates.
        unsigned NoOperands = Cand->Conversions.size();
        for (unsigned ArgIdx = 0; ArgIdx < NoOperands; ++ArgIdx) {
          const ImplicitConversionSequence &ICS = Cand->Conversions[ArgIdx];
          if (ICS.ConversionKind != ImplicitConversionSequence::BadConversion ||
              ICS.ConversionFunctionSet.empty())
            continue;
          if (CXXConversionDecl *Func = dyn_cast<CXXConversionDecl>(
                         Cand->Conversions[ArgIdx].ConversionFunctionSet[0])) {
            QualType FromTy = 
              QualType(
                     static_cast<Type*>(ICS.UserDefined.Before.FromTypePtr),0);
            Diag(OpLoc,diag::note_ambiguous_type_conversion)
                  << FromTy << Func->getConversionType();
          }
          for (unsigned j = 0; j < ICS.ConversionFunctionSet.size(); j++) {
            FunctionDecl *Func = 
              Cand->Conversions[ArgIdx].ConversionFunctionSet[j];
            Diag(Func->getLocation(),diag::err_ovl_candidate);
          }
        }
        Reported = true;
      }
    }
  }
}

/// ResolveAddressOfOverloadedFunction - Try to resolve the address of
/// an overloaded function (C++ [over.over]), where @p From is an
/// expression with overloaded function type and @p ToType is the type
/// we're trying to resolve to. For example:
///
/// @code
/// int f(double);
/// int f(int);
///
/// int (*pfd)(double) = f; // selects f(double)
/// @endcode
///
/// This routine returns the resulting FunctionDecl if it could be
/// resolved, and NULL otherwise. When @p Complain is true, this
/// routine will emit diagnostics if there is an error.
FunctionDecl *
Sema::ResolveAddressOfOverloadedFunction(Expr *From, QualType ToType,
                                         bool Complain) {
  QualType FunctionType = ToType;
  bool IsMember = false;
  if (const PointerType *ToTypePtr = ToType->getAs<PointerType>())
    FunctionType = ToTypePtr->getPointeeType();
  else if (const ReferenceType *ToTypeRef = ToType->getAs<ReferenceType>())
    FunctionType = ToTypeRef->getPointeeType();
  else if (const MemberPointerType *MemTypePtr =
                    ToType->getAs<MemberPointerType>()) {
    FunctionType = MemTypePtr->getPointeeType();
    IsMember = true;
  }

  // We only look at pointers or references to functions.
  FunctionType = Context.getCanonicalType(FunctionType).getUnqualifiedType();
  if (!FunctionType->isFunctionType())
    return 0;

  // Find the actual overloaded function declaration.

  // C++ [over.over]p1:
  //   [...] [Note: any redundant set of parentheses surrounding the
  //   overloaded function name is ignored (5.1). ]
  Expr *OvlExpr = From->IgnoreParens();

  // C++ [over.over]p1:
  //   [...] The overloaded function name can be preceded by the &
  //   operator.
  if (UnaryOperator *UnOp = dyn_cast<UnaryOperator>(OvlExpr)) {
    if (UnOp->getOpcode() == UnaryOperator::AddrOf)
      OvlExpr = UnOp->getSubExpr()->IgnoreParens();
  }

  bool HasExplicitTemplateArgs = false;
  TemplateArgumentListInfo ExplicitTemplateArgs;

  llvm::SmallVector<NamedDecl*,8> Fns;
  
  // Look into the overloaded expression.
  if (UnresolvedLookupExpr *UL
               = dyn_cast<UnresolvedLookupExpr>(OvlExpr)) {
    Fns.append(UL->decls_begin(), UL->decls_end());
    if (UL->hasExplicitTemplateArgs()) {
      HasExplicitTemplateArgs = true;
      UL->copyTemplateArgumentsInto(ExplicitTemplateArgs);
    }
  } else if (UnresolvedMemberExpr *ME
               = dyn_cast<UnresolvedMemberExpr>(OvlExpr)) {
    Fns.append(ME->decls_begin(), ME->decls_end());
    if (ME->hasExplicitTemplateArgs()) {
      HasExplicitTemplateArgs = true;
      ME->copyTemplateArgumentsInto(ExplicitTemplateArgs);
    }
  }

  // If we didn't actually find anything, we're done.
  if (Fns.empty())
    return 0;

  // Look through all of the overloaded functions, searching for one
  // whose type matches exactly.
  llvm::SmallPtrSet<FunctionDecl *, 4> Matches;
  bool FoundNonTemplateFunction = false;
  for (llvm::SmallVectorImpl<NamedDecl*>::iterator I = Fns.begin(),
         E = Fns.end(); I != E; ++I) {
    // Look through any using declarations to find the underlying function.
    NamedDecl *Fn = (*I)->getUnderlyingDecl();

    // C++ [over.over]p3:
    //   Non-member functions and static member functions match
    //   targets of type "pointer-to-function" or "reference-to-function."
    //   Nonstatic member functions match targets of
    //   type "pointer-to-member-function."
    // Note that according to DR 247, the containing class does not matter.

    if (FunctionTemplateDecl *FunctionTemplate
          = dyn_cast<FunctionTemplateDecl>(Fn)) {
      if (CXXMethodDecl *Method
            = dyn_cast<CXXMethodDecl>(FunctionTemplate->getTemplatedDecl())) {
        // Skip non-static function templates when converting to pointer, and
        // static when converting to member pointer.
        if (Method->isStatic() == IsMember)
          continue;
      } else if (IsMember)
        continue;

      // C++ [over.over]p2:
      //   If the name is a function template, template argument deduction is
      //   done (14.8.2.2), and if the argument deduction succeeds, the
      //   resulting template argument list is used to generate a single
      //   function template specialization, which is added to the set of
      //   overloaded functions considered.
      // FIXME: We don't really want to build the specialization here, do we?
      FunctionDecl *Specialization = 0;
      TemplateDeductionInfo Info(Context);
      if (TemplateDeductionResult Result
            = DeduceTemplateArguments(FunctionTemplate,
                       (HasExplicitTemplateArgs ? &ExplicitTemplateArgs : 0),
                                      FunctionType, Specialization, Info)) {
        // FIXME: make a note of the failed deduction for diagnostics.
        (void)Result;
      } else {
        // FIXME: If the match isn't exact, shouldn't we just drop this as
        // a candidate? Find a testcase before changing the code.
        assert(FunctionType
                 == Context.getCanonicalType(Specialization->getType()));
        Matches.insert(
                cast<FunctionDecl>(Specialization->getCanonicalDecl()));
      }

      continue;
    }

    if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(Fn)) {
      // Skip non-static functions when converting to pointer, and static
      // when converting to member pointer.
      if (Method->isStatic() == IsMember)
        continue;
      
      // If we have explicit template arguments, skip non-templates.
      if (HasExplicitTemplateArgs)
        continue;
    } else if (IsMember)
      continue;

    if (FunctionDecl *FunDecl = dyn_cast<FunctionDecl>(Fn)) {
      QualType ResultTy;
      if (Context.hasSameUnqualifiedType(FunctionType, FunDecl->getType()) ||
          IsNoReturnConversion(Context, FunDecl->getType(), FunctionType, 
                               ResultTy)) {
        Matches.insert(cast<FunctionDecl>(FunDecl->getCanonicalDecl()));
        FoundNonTemplateFunction = true;
      }
    }
  }

  // If there were 0 or 1 matches, we're done.
  if (Matches.empty())
    return 0;
  else if (Matches.size() == 1) {
    FunctionDecl *Result = *Matches.begin();
    MarkDeclarationReferenced(From->getLocStart(), Result);
    return Result;
  }

  // C++ [over.over]p4:
  //   If more than one function is selected, [...]
  typedef llvm::SmallPtrSet<FunctionDecl *, 4>::iterator MatchIter;
  if (!FoundNonTemplateFunction) {
    //   [...] and any given function template specialization F1 is
    //   eliminated if the set contains a second function template
    //   specialization whose function template is more specialized
    //   than the function template of F1 according to the partial
    //   ordering rules of 14.5.5.2.

    // The algorithm specified above is quadratic. We instead use a
    // two-pass algorithm (similar to the one used to identify the
    // best viable function in an overload set) that identifies the
    // best function template (if it exists).
    llvm::SmallVector<FunctionDecl *, 8> TemplateMatches(Matches.begin(),
                                                         Matches.end());
    FunctionDecl *Result =
        getMostSpecialized(TemplateMatches.data(), TemplateMatches.size(),
                           TPOC_Other, From->getLocStart(),
                           PDiag(),
                           PDiag(diag::err_addr_ovl_ambiguous)
                               << TemplateMatches[0]->getDeclName(),
                           PDiag(diag::err_ovl_template_candidate));
    MarkDeclarationReferenced(From->getLocStart(), Result);
    return Result;
  }

  //   [...] any function template specializations in the set are
  //   eliminated if the set also contains a non-template function, [...]
  llvm::SmallVector<FunctionDecl *, 4> RemainingMatches;
  for (MatchIter M = Matches.begin(), MEnd = Matches.end(); M != MEnd; ++M)
    if ((*M)->getPrimaryTemplate() == 0)
      RemainingMatches.push_back(*M);
  
  // [...] After such eliminations, if any, there shall remain exactly one
  // selected function.
  if (RemainingMatches.size() == 1) {
    FunctionDecl *Result = RemainingMatches.front();
    MarkDeclarationReferenced(From->getLocStart(), Result);
    return Result;
  }

  // FIXME: We should probably return the same thing that BestViableFunction
  // returns (even if we issue the diagnostics here).
  Diag(From->getLocStart(), diag::err_addr_ovl_ambiguous)
    << RemainingMatches[0]->getDeclName();
  for (unsigned I = 0, N = RemainingMatches.size(); I != N; ++I)
    Diag(RemainingMatches[I]->getLocation(), diag::err_ovl_candidate);
  return 0;
}

/// \brief Given an expression that refers to an overloaded function, try to 
/// resolve that overloaded function expression down to a single function.
///
/// This routine can only resolve template-ids that refer to a single function
/// template, where that template-id refers to a single template whose template
/// arguments are either provided by the template-id or have defaults, 
/// as described in C++0x [temp.arg.explicit]p3.
FunctionDecl *Sema::ResolveSingleFunctionTemplateSpecialization(Expr *From) {
  // C++ [over.over]p1:
  //   [...] [Note: any redundant set of parentheses surrounding the
  //   overloaded function name is ignored (5.1). ]
  Expr *OvlExpr = From->IgnoreParens();
  
  // C++ [over.over]p1:
  //   [...] The overloaded function name can be preceded by the &
  //   operator.
  if (UnaryOperator *UnOp = dyn_cast<UnaryOperator>(OvlExpr)) {
    if (UnOp->getOpcode() == UnaryOperator::AddrOf)
      OvlExpr = UnOp->getSubExpr()->IgnoreParens();
  }
  
  bool HasExplicitTemplateArgs = false;
  TemplateArgumentListInfo ExplicitTemplateArgs;
  
  llvm::SmallVector<NamedDecl*,8> Fns;
  
  // Look into the overloaded expression.
  if (UnresolvedLookupExpr *UL
      = dyn_cast<UnresolvedLookupExpr>(OvlExpr)) {
    Fns.append(UL->decls_begin(), UL->decls_end());
    if (UL->hasExplicitTemplateArgs()) {
      HasExplicitTemplateArgs = true;
      UL->copyTemplateArgumentsInto(ExplicitTemplateArgs);
    }
  } else if (UnresolvedMemberExpr *ME
             = dyn_cast<UnresolvedMemberExpr>(OvlExpr)) {
    Fns.append(ME->decls_begin(), ME->decls_end());
    if (ME->hasExplicitTemplateArgs()) {
      HasExplicitTemplateArgs = true;
      ME->copyTemplateArgumentsInto(ExplicitTemplateArgs);
    }
  }
  
  // If we didn't actually find any template-ids, we're done.
  if (Fns.empty() || !HasExplicitTemplateArgs)
    return 0;
  
  // Look through all of the overloaded functions, searching for one
  // whose type matches exactly.
  FunctionDecl *Matched = 0;
  for (llvm::SmallVectorImpl<NamedDecl*>::iterator I = Fns.begin(),
       E = Fns.end(); I != E; ++I) {
    // C++0x [temp.arg.explicit]p3:
    //   [...] In contexts where deduction is done and fails, or in contexts
    //   where deduction is not done, if a template argument list is 
    //   specified and it, along with any default template arguments, 
    //   identifies a single function template specialization, then the 
    //   template-id is an lvalue for the function template specialization.
    FunctionTemplateDecl *FunctionTemplate = cast<FunctionTemplateDecl>(*I);
    
    // C++ [over.over]p2:
    //   If the name is a function template, template argument deduction is
    //   done (14.8.2.2), and if the argument deduction succeeds, the
    //   resulting template argument list is used to generate a single
    //   function template specialization, which is added to the set of
    //   overloaded functions considered.
    // FIXME: We don't really want to build the specialization here, do we?
    FunctionDecl *Specialization = 0;
    TemplateDeductionInfo Info(Context);
    if (TemplateDeductionResult Result
          = DeduceTemplateArguments(FunctionTemplate, &ExplicitTemplateArgs,
                                    Specialization, Info)) {
      // FIXME: make a note of the failed deduction for diagnostics.
      (void)Result;
      continue;
    } 
    
    // Multiple matches; we can't resolve to a single declaration.
    if (Matched)
      return 0;

    Matched = Specialization;
  }

  return Matched;
}
    
/// \brief Add a single candidate to the overload set.
static void AddOverloadedCallCandidate(Sema &S,
                                       NamedDecl *Callee,
                       const TemplateArgumentListInfo *ExplicitTemplateArgs,
                                       Expr **Args, unsigned NumArgs,
                                       OverloadCandidateSet &CandidateSet,
                                       bool PartialOverloading) {
  if (isa<UsingShadowDecl>(Callee))
    Callee = cast<UsingShadowDecl>(Callee)->getTargetDecl();

  if (FunctionDecl *Func = dyn_cast<FunctionDecl>(Callee)) {
    assert(!ExplicitTemplateArgs && "Explicit template arguments?");
    S.AddOverloadCandidate(Func, Args, NumArgs, CandidateSet, false, false,
                           PartialOverloading);
    return;
  }

  if (FunctionTemplateDecl *FuncTemplate
      = dyn_cast<FunctionTemplateDecl>(Callee)) {
    S.AddTemplateOverloadCandidate(FuncTemplate, ExplicitTemplateArgs,
                                   Args, NumArgs, CandidateSet);
    return;
  }

  assert(false && "unhandled case in overloaded call candidate");

  // do nothing?
}
  
/// \brief Add the overload candidates named by callee and/or found by argument
/// dependent lookup to the given overload set.
void Sema::AddOverloadedCallCandidates(UnresolvedLookupExpr *ULE,
                                       Expr **Args, unsigned NumArgs,
                                       OverloadCandidateSet &CandidateSet,
                                       bool PartialOverloading) {

#ifndef NDEBUG
  // Verify that ArgumentDependentLookup is consistent with the rules
  // in C++0x [basic.lookup.argdep]p3:
  //
  //   Let X be the lookup set produced by unqualified lookup (3.4.1)
  //   and let Y be the lookup set produced by argument dependent
  //   lookup (defined as follows). If X contains
  //
  //     -- a declaration of a class member, or
  //
  //     -- a block-scope function declaration that is not a
  //        using-declaration, or
  //
  //     -- a declaration that is neither a function or a function
  //        template
  //
  //   then Y is empty.

  if (ULE->requiresADL()) {
    for (UnresolvedLookupExpr::decls_iterator I = ULE->decls_begin(),
           E = ULE->decls_end(); I != E; ++I) {
      assert(!(*I)->getDeclContext()->isRecord());
      assert(isa<UsingShadowDecl>(*I) ||
             !(*I)->getDeclContext()->isFunctionOrMethod());
      assert((*I)->getUnderlyingDecl()->isFunctionOrFunctionTemplate());
    }
  }
#endif

  // It would be nice to avoid this copy.
  TemplateArgumentListInfo TABuffer;
  const TemplateArgumentListInfo *ExplicitTemplateArgs = 0;
  if (ULE->hasExplicitTemplateArgs()) {
    ULE->copyTemplateArgumentsInto(TABuffer);
    ExplicitTemplateArgs = &TABuffer;
  }

  for (UnresolvedLookupExpr::decls_iterator I = ULE->decls_begin(),
         E = ULE->decls_end(); I != E; ++I)
    AddOverloadedCallCandidate(*this, *I, ExplicitTemplateArgs,
                               Args, NumArgs, CandidateSet, 
                               PartialOverloading);

  if (ULE->requiresADL())
    AddArgumentDependentLookupCandidates(ULE->getName(), Args, NumArgs,
                                         ExplicitTemplateArgs,
                                         CandidateSet,
                                         PartialOverloading);  
}

static Sema::OwningExprResult Destroy(Sema &SemaRef, Expr *Fn,
                                      Expr **Args, unsigned NumArgs) {
  Fn->Destroy(SemaRef.Context);
  for (unsigned Arg = 0; Arg < NumArgs; ++Arg)
    Args[Arg]->Destroy(SemaRef.Context);
  return SemaRef.ExprError();
}

/// Attempts to recover from a call where no functions were found.
///
/// Returns true if new candidates were found.
static Sema::OwningExprResult
BuildRecoveryCallExpr(Sema &SemaRef, Expr *Fn,
                      UnresolvedLookupExpr *ULE,
                      SourceLocation LParenLoc,
                      Expr **Args, unsigned NumArgs,
                      SourceLocation *CommaLocs,
                      SourceLocation RParenLoc) {

  CXXScopeSpec SS;
  if (ULE->getQualifier()) {
    SS.setScopeRep(ULE->getQualifier());
    SS.setRange(ULE->getQualifierRange());
  }

  TemplateArgumentListInfo TABuffer;
  const TemplateArgumentListInfo *ExplicitTemplateArgs = 0;
  if (ULE->hasExplicitTemplateArgs()) {
    ULE->copyTemplateArgumentsInto(TABuffer);
    ExplicitTemplateArgs = &TABuffer;
  }

  LookupResult R(SemaRef, ULE->getName(), ULE->getNameLoc(),
                 Sema::LookupOrdinaryName);
  if (SemaRef.DiagnoseEmptyLookup(SS, R))
    return Destroy(SemaRef, Fn, Args, NumArgs);

  assert(!R.empty() && "lookup results empty despite recovery");

  // Build an implicit member call if appropriate.  Just drop the
  // casts and such from the call, we don't really care.
  Sema::OwningExprResult NewFn = SemaRef.ExprError();
  if ((*R.begin())->isCXXClassMember())
    NewFn = SemaRef.BuildPossibleImplicitMemberExpr(SS, R, ExplicitTemplateArgs);
  else if (ExplicitTemplateArgs)
    NewFn = SemaRef.BuildTemplateIdExpr(SS, R, false, *ExplicitTemplateArgs);
  else
    NewFn = SemaRef.BuildDeclarationNameExpr(SS, R, false);

  if (NewFn.isInvalid())
    return Destroy(SemaRef, Fn, Args, NumArgs);

  Fn->Destroy(SemaRef.Context);

  // This shouldn't cause an infinite loop because we're giving it
  // an expression with non-empty lookup results, which should never
  // end up here.
  return SemaRef.ActOnCallExpr(/*Scope*/ 0, move(NewFn), LParenLoc,
                         Sema::MultiExprArg(SemaRef, (void**) Args, NumArgs),
                               CommaLocs, RParenLoc);
}
  
/// ResolveOverloadedCallFn - Given the call expression that calls Fn
/// (which eventually refers to the declaration Func) and the call
/// arguments Args/NumArgs, attempt to resolve the function call down
/// to a specific function. If overload resolution succeeds, returns
/// the function declaration produced by overload
/// resolution. Otherwise, emits diagnostics, deletes all of the
/// arguments and Fn, and returns NULL.
Sema::OwningExprResult
Sema::BuildOverloadedCallExpr(Expr *Fn, UnresolvedLookupExpr *ULE,
                              SourceLocation LParenLoc,
                              Expr **Args, unsigned NumArgs,
                              SourceLocation *CommaLocs,
                              SourceLocation RParenLoc) {
#ifndef NDEBUG
  if (ULE->requiresADL()) {
    // To do ADL, we must have found an unqualified name.
    assert(!ULE->getQualifier() && "qualified name with ADL");

    // We don't perform ADL for implicit declarations of builtins.
    // Verify that this was correctly set up.
    FunctionDecl *F;
    if (ULE->decls_begin() + 1 == ULE->decls_end() &&
        (F = dyn_cast<FunctionDecl>(*ULE->decls_begin())) &&
        F->getBuiltinID() && F->isImplicit())
      assert(0 && "performing ADL for builtin");
      
    // We don't perform ADL in C.
    assert(getLangOptions().CPlusPlus && "ADL enabled in C");
  }
#endif

  OverloadCandidateSet CandidateSet;

  // Add the functions denoted by the callee to the set of candidate
  // functions, including those from argument-dependent lookup.
  AddOverloadedCallCandidates(ULE, Args, NumArgs, CandidateSet);

  // If we found nothing, try to recover.
  // AddRecoveryCallCandidates diagnoses the error itself, so we just
  // bailout out if it fails.
  if (CandidateSet.empty())
    return BuildRecoveryCallExpr(*this, Fn, ULE, LParenLoc, Args, NumArgs,
                                 CommaLocs, RParenLoc);

  OverloadCandidateSet::iterator Best;
  switch (BestViableFunction(CandidateSet, Fn->getLocStart(), Best)) {
  case OR_Success: {
    FunctionDecl *FDecl = Best->Function;
    Fn = FixOverloadedFunctionReference(Fn, FDecl);
    return BuildResolvedCallExpr(Fn, FDecl, LParenLoc, Args, NumArgs, RParenLoc);
  }

  case OR_No_Viable_Function:
    Diag(Fn->getSourceRange().getBegin(),
         diag::err_ovl_no_viable_function_in_call)
      << ULE->getName() << Fn->getSourceRange();
    PrintOverloadCandidates(CandidateSet, /*OnlyViable=*/false);
    break;

  case OR_Ambiguous:
    Diag(Fn->getSourceRange().getBegin(), diag::err_ovl_ambiguous_call)
      << ULE->getName() << Fn->getSourceRange();
    PrintOverloadCandidates(CandidateSet, /*OnlyViable=*/true);
    break;

  case OR_Deleted:
    Diag(Fn->getSourceRange().getBegin(), diag::err_ovl_deleted_call)
      << Best->Function->isDeleted()
      << ULE->getName()
      << Fn->getSourceRange();
    PrintOverloadCandidates(CandidateSet, /*OnlyViable=*/true);
    break;
  }

  // Overload resolution failed. Destroy all of the subexpressions and
  // return NULL.
  Fn->Destroy(Context);
  for (unsigned Arg = 0; Arg < NumArgs; ++Arg)
    Args[Arg]->Destroy(Context);
  return ExprError();
}

static bool IsOverloaded(const Sema::FunctionSet &Functions) {
  return Functions.size() > 1 ||
    (Functions.size() == 1 && isa<FunctionTemplateDecl>(*Functions.begin()));
}

/// \brief Create a unary operation that may resolve to an overloaded
/// operator.
///
/// \param OpLoc The location of the operator itself (e.g., '*').
///
/// \param OpcIn The UnaryOperator::Opcode that describes this
/// operator.
///
/// \param Functions The set of non-member functions that will be
/// considered by overload resolution. The caller needs to build this
/// set based on the context using, e.g.,
/// LookupOverloadedOperatorName() and ArgumentDependentLookup(). This
/// set should not contain any member functions; those will be added
/// by CreateOverloadedUnaryOp().
///
/// \param input The input argument.
Sema::OwningExprResult Sema::CreateOverloadedUnaryOp(SourceLocation OpLoc,
                                                     unsigned OpcIn,
                                                     FunctionSet &Functions,
                                                     ExprArg input) {
  UnaryOperator::Opcode Opc = static_cast<UnaryOperator::Opcode>(OpcIn);
  Expr *Input = (Expr *)input.get();

  OverloadedOperatorKind Op = UnaryOperator::getOverloadedOperator(Opc);
  assert(Op != OO_None && "Invalid opcode for overloaded unary operator");
  DeclarationName OpName = Context.DeclarationNames.getCXXOperatorName(Op);

  Expr *Args[2] = { Input, 0 };
  unsigned NumArgs = 1;

  // For post-increment and post-decrement, add the implicit '0' as
  // the second argument, so that we know this is a post-increment or
  // post-decrement.
  if (Opc == UnaryOperator::PostInc || Opc == UnaryOperator::PostDec) {
    llvm::APSInt Zero(Context.getTypeSize(Context.IntTy), false);
    Args[1] = new (Context) IntegerLiteral(Zero, Context.IntTy,
                                           SourceLocation());
    NumArgs = 2;
  }

  if (Input->isTypeDependent()) {
    UnresolvedLookupExpr *Fn
      = UnresolvedLookupExpr::Create(Context, /*Dependent*/ true,
                                     0, SourceRange(), OpName, OpLoc,
                                     /*ADL*/ true, IsOverloaded(Functions));
    for (FunctionSet::iterator Func = Functions.begin(),
                            FuncEnd = Functions.end();
         Func != FuncEnd; ++Func)
      Fn->addDecl(*Func);

    input.release();
    return Owned(new (Context) CXXOperatorCallExpr(Context, Op, Fn,
                                                   &Args[0], NumArgs,
                                                   Context.DependentTy,
                                                   OpLoc));
  }

  // Build an empty overload set.
  OverloadCandidateSet CandidateSet;

  // Add the candidates from the given function set.
  AddFunctionCandidates(Functions, &Args[0], NumArgs, CandidateSet, false);

  // Add operator candidates that are member functions.
  AddMemberOperatorCandidates(Op, OpLoc, &Args[0], NumArgs, CandidateSet);

  // Add builtin operator candidates.
  AddBuiltinOperatorCandidates(Op, OpLoc, &Args[0], NumArgs, CandidateSet);

  // Perform overload resolution.
  OverloadCandidateSet::iterator Best;
  switch (BestViableFunction(CandidateSet, OpLoc, Best)) {
  case OR_Success: {
    // We found a built-in operator or an overloaded operator.
    FunctionDecl *FnDecl = Best->Function;

    if (FnDecl) {
      // We matched an overloaded operator. Build a call to that
      // operator.

      // Convert the arguments.
      if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(FnDecl)) {
        if (PerformObjectArgumentInitialization(Input, Method))
          return ExprError();
      } else {
        // Convert the arguments.
        OwningExprResult InputInit
          = PerformCopyInitialization(InitializedEntity::InitializeParameter(
                                                      FnDecl->getParamDecl(0)),
                                      SourceLocation(), 
                                      move(input));
        if (InputInit.isInvalid())
          return ExprError();
        
        input = move(InputInit);
        Input = (Expr *)input.get();
      }

      // Determine the result type
      QualType ResultTy = FnDecl->getResultType().getNonReferenceType();

      // Build the actual expression node.
      Expr *FnExpr = new (Context) DeclRefExpr(FnDecl, FnDecl->getType(),
                                               SourceLocation());
      UsualUnaryConversions(FnExpr);

      input.release();
      Args[0] = Input;
      ExprOwningPtr<CallExpr> TheCall(this,
        new (Context) CXXOperatorCallExpr(Context, Op, FnExpr,
                                          Args, NumArgs, ResultTy, OpLoc));
      
      if (CheckCallReturnType(FnDecl->getResultType(), OpLoc, TheCall.get(), 
                              FnDecl))
        return ExprError();

      return MaybeBindToTemporary(TheCall.release());
    } else {
      // We matched a built-in operator. Convert the arguments, then
      // break out so that we will build the appropriate built-in
      // operator node.
        if (PerformImplicitConversion(Input, Best->BuiltinTypes.ParamTypes[0],
                                      Best->Conversions[0], AA_Passing))
          return ExprError();

        break;
      }
    }

    case OR_No_Viable_Function:
      // No viable function; fall through to handling this as a
      // built-in operator, which will produce an error message for us.
      break;

    case OR_Ambiguous:
      Diag(OpLoc,  diag::err_ovl_ambiguous_oper)
          << UnaryOperator::getOpcodeStr(Opc)
          << Input->getSourceRange();
      PrintOverloadCandidates(CandidateSet, /*OnlyViable=*/true, 
                              UnaryOperator::getOpcodeStr(Opc), OpLoc);
      return ExprError();

    case OR_Deleted:
      Diag(OpLoc, diag::err_ovl_deleted_oper)
        << Best->Function->isDeleted()
        << UnaryOperator::getOpcodeStr(Opc)
        << Input->getSourceRange();
      PrintOverloadCandidates(CandidateSet, /*OnlyViable=*/true);
      return ExprError();
    }

  // Either we found no viable overloaded operator or we matched a
  // built-in operator. In either case, fall through to trying to
  // build a built-in operation.
  input.release();
  return CreateBuiltinUnaryOp(OpLoc, Opc, Owned(Input));
}

/// \brief Create a binary operation that may resolve to an overloaded
/// operator.
///
/// \param OpLoc The location of the operator itself (e.g., '+').
///
/// \param OpcIn The BinaryOperator::Opcode that describes this
/// operator.
///
/// \param Functions The set of non-member functions that will be
/// considered by overload resolution. The caller needs to build this
/// set based on the context using, e.g.,
/// LookupOverloadedOperatorName() and ArgumentDependentLookup(). This
/// set should not contain any member functions; those will be added
/// by CreateOverloadedBinOp().
///
/// \param LHS Left-hand argument.
/// \param RHS Right-hand argument.
Sema::OwningExprResult
Sema::CreateOverloadedBinOp(SourceLocation OpLoc,
                            unsigned OpcIn,
                            FunctionSet &Functions,
                            Expr *LHS, Expr *RHS) {
  Expr *Args[2] = { LHS, RHS };
  LHS=RHS=0; //Please use only Args instead of LHS/RHS couple

  BinaryOperator::Opcode Opc = static_cast<BinaryOperator::Opcode>(OpcIn);
  OverloadedOperatorKind Op = BinaryOperator::getOverloadedOperator(Opc);
  DeclarationName OpName = Context.DeclarationNames.getCXXOperatorName(Op);

  // If either side is type-dependent, create an appropriate dependent
  // expression.
  if (Args[0]->isTypeDependent() || Args[1]->isTypeDependent()) {
    if (Functions.empty()) {
      // If there are no functions to store, just build a dependent 
      // BinaryOperator or CompoundAssignment.
      if (Opc <= BinaryOperator::Assign || Opc > BinaryOperator::OrAssign)
        return Owned(new (Context) BinaryOperator(Args[0], Args[1], Opc,
                                                  Context.DependentTy, OpLoc));
      
      return Owned(new (Context) CompoundAssignOperator(Args[0], Args[1], Opc,
                                                        Context.DependentTy,
                                                        Context.DependentTy,
                                                        Context.DependentTy,
                                                        OpLoc));
    }
    
    UnresolvedLookupExpr *Fn
      = UnresolvedLookupExpr::Create(Context, /*Dependent*/ true,
                                     0, SourceRange(), OpName, OpLoc,
                                     /* ADL */ true, IsOverloaded(Functions));
                                     
    for (FunctionSet::iterator Func = Functions.begin(),
                            FuncEnd = Functions.end();
         Func != FuncEnd; ++Func)
      Fn->addDecl(*Func);

    return Owned(new (Context) CXXOperatorCallExpr(Context, Op, Fn,
                                                   Args, 2,
                                                   Context.DependentTy,
                                                   OpLoc));
  }

  // If this is the .* operator, which is not overloadable, just
  // create a built-in binary operator.
  if (Opc == BinaryOperator::PtrMemD)
    return CreateBuiltinBinOp(OpLoc, Opc, Args[0], Args[1]);

  // If this is the assignment operator, we only perform overload resolution
  // if the left-hand side is a class or enumeration type. This is actually
  // a hack. The standard requires that we do overload resolution between the
  // various built-in candidates, but as DR507 points out, this can lead to
  // problems. So we do it this way, which pretty much follows what GCC does.
  // Note that we go the traditional code path for compound assignment forms.
  if (Opc==BinaryOperator::Assign && !Args[0]->getType()->isOverloadableType())
    return CreateBuiltinBinOp(OpLoc, Opc, Args[0], Args[1]);

  // Build an empty overload set.
  OverloadCandidateSet CandidateSet;

  // Add the candidates from the given function set.
  AddFunctionCandidates(Functions, Args, 2, CandidateSet, false);

  // Add operator candidates that are member functions.
  AddMemberOperatorCandidates(Op, OpLoc, Args, 2, CandidateSet);

  // Add builtin operator candidates.
  AddBuiltinOperatorCandidates(Op, OpLoc, Args, 2, CandidateSet);

  // Perform overload resolution.
  OverloadCandidateSet::iterator Best;
  switch (BestViableFunction(CandidateSet, OpLoc, Best)) {
    case OR_Success: {
      // We found a built-in operator or an overloaded operator.
      FunctionDecl *FnDecl = Best->Function;

      if (FnDecl) {
        // We matched an overloaded operator. Build a call to that
        // operator.

        // Convert the arguments.
        if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(FnDecl)) {
          OwningExprResult Arg1
            = PerformCopyInitialization(
                                        InitializedEntity::InitializeParameter(
                                                        FnDecl->getParamDecl(0)),
                                        SourceLocation(),
                                        Owned(Args[1]));
          if (Arg1.isInvalid())
            return ExprError();

          if (PerformObjectArgumentInitialization(Args[0], Method))
            return ExprError();

          Args[1] = RHS = Arg1.takeAs<Expr>();
        } else {
          // Convert the arguments.
          OwningExprResult Arg0
            = PerformCopyInitialization(
                                        InitializedEntity::InitializeParameter(
                                                        FnDecl->getParamDecl(0)),
                                        SourceLocation(),
                                        Owned(Args[0]));
          if (Arg0.isInvalid())
            return ExprError();

          OwningExprResult Arg1
            = PerformCopyInitialization(
                                        InitializedEntity::InitializeParameter(
                                                        FnDecl->getParamDecl(1)),
                                        SourceLocation(),
                                        Owned(Args[1]));
          if (Arg1.isInvalid())
            return ExprError();
          Args[0] = LHS = Arg0.takeAs<Expr>();
          Args[1] = RHS = Arg1.takeAs<Expr>();
        }

        // Determine the result type
        QualType ResultTy
          = FnDecl->getType()->getAs<FunctionType>()->getResultType();
        ResultTy = ResultTy.getNonReferenceType();

        // Build the actual expression node.
        Expr *FnExpr = new (Context) DeclRefExpr(FnDecl, FnDecl->getType(),
                                                 OpLoc);
        UsualUnaryConversions(FnExpr);

        ExprOwningPtr<CXXOperatorCallExpr> 
          TheCall(this, new (Context) CXXOperatorCallExpr(Context, Op, FnExpr,
                                                          Args, 2, ResultTy, 
                                                          OpLoc));
        
        if (CheckCallReturnType(FnDecl->getResultType(), OpLoc, TheCall.get(), 
                                FnDecl))
          return ExprError();

        return MaybeBindToTemporary(TheCall.release());
      } else {
        // We matched a built-in operator. Convert the arguments, then
        // break out so that we will build the appropriate built-in
        // operator node.
        if (PerformImplicitConversion(Args[0], Best->BuiltinTypes.ParamTypes[0],
                                      Best->Conversions[0], AA_Passing) ||
            PerformImplicitConversion(Args[1], Best->BuiltinTypes.ParamTypes[1],
                                      Best->Conversions[1], AA_Passing))
          return ExprError();

        break;
      }
    }

    case OR_No_Viable_Function: {
      // C++ [over.match.oper]p9:
      //   If the operator is the operator , [...] and there are no
      //   viable functions, then the operator is assumed to be the
      //   built-in operator and interpreted according to clause 5.
      if (Opc == BinaryOperator::Comma)
        break;

      // For class as left operand for assignment or compound assigment operator
      // do not fall through to handling in built-in, but report that no overloaded
      // assignment operator found
      OwningExprResult Result = ExprError();
      if (Args[0]->getType()->isRecordType() && 
          Opc >= BinaryOperator::Assign && Opc <= BinaryOperator::OrAssign) {
        Diag(OpLoc,  diag::err_ovl_no_viable_oper)
             << BinaryOperator::getOpcodeStr(Opc)
             << Args[0]->getSourceRange() << Args[1]->getSourceRange();
      } else {
        // No viable function; try to create a built-in operation, which will
        // produce an error. Then, show the non-viable candidates.
        Result = CreateBuiltinBinOp(OpLoc, Opc, Args[0], Args[1]);
      }
      assert(Result.isInvalid() && 
             "C++ binary operator overloading is missing candidates!");
      if (Result.isInvalid())
        PrintOverloadCandidates(CandidateSet, /*OnlyViable=*/false, 
                                BinaryOperator::getOpcodeStr(Opc), OpLoc);
      return move(Result);
    }

    case OR_Ambiguous:
      Diag(OpLoc,  diag::err_ovl_ambiguous_oper)
          << BinaryOperator::getOpcodeStr(Opc)
          << Args[0]->getSourceRange() << Args[1]->getSourceRange();
      PrintOverloadCandidates(CandidateSet, /*OnlyViable=*/true, 
                              BinaryOperator::getOpcodeStr(Opc), OpLoc);
      return ExprError();

    case OR_Deleted:
      Diag(OpLoc, diag::err_ovl_deleted_oper)
        << Best->Function->isDeleted()
        << BinaryOperator::getOpcodeStr(Opc)
        << Args[0]->getSourceRange() << Args[1]->getSourceRange();
      PrintOverloadCandidates(CandidateSet, /*OnlyViable=*/true);
      return ExprError();
    }

  // We matched a built-in operator; build it.
  return CreateBuiltinBinOp(OpLoc, Opc, Args[0], Args[1]);
}

Action::OwningExprResult
Sema::CreateOverloadedArraySubscriptExpr(SourceLocation LLoc,
                                         SourceLocation RLoc,
                                         ExprArg Base, ExprArg Idx) {
  Expr *Args[2] = { static_cast<Expr*>(Base.get()),
                    static_cast<Expr*>(Idx.get()) };
  DeclarationName OpName =
      Context.DeclarationNames.getCXXOperatorName(OO_Subscript);

  // If either side is type-dependent, create an appropriate dependent
  // expression.
  if (Args[0]->isTypeDependent() || Args[1]->isTypeDependent()) {

    UnresolvedLookupExpr *Fn
      = UnresolvedLookupExpr::Create(Context, /*Dependent*/ true,
                                     0, SourceRange(), OpName, LLoc,
                                     /*ADL*/ true, /*Overloaded*/ false);
    // Can't add any actual overloads yet

    Base.release();
    Idx.release();
    return Owned(new (Context) CXXOperatorCallExpr(Context, OO_Subscript, Fn,
                                                   Args, 2,
                                                   Context.DependentTy,
                                                   RLoc));
  }

  // Build an empty overload set.
  OverloadCandidateSet CandidateSet;

  // Subscript can only be overloaded as a member function.

  // Add operator candidates that are member functions.
  AddMemberOperatorCandidates(OO_Subscript, LLoc, Args, 2, CandidateSet);

  // Add builtin operator candidates.
  AddBuiltinOperatorCandidates(OO_Subscript, LLoc, Args, 2, CandidateSet);

  // Perform overload resolution.
  OverloadCandidateSet::iterator Best;
  switch (BestViableFunction(CandidateSet, LLoc, Best)) {
    case OR_Success: {
      // We found a built-in operator or an overloaded operator.
      FunctionDecl *FnDecl = Best->Function;

      if (FnDecl) {
        // We matched an overloaded operator. Build a call to that
        // operator.

        // Convert the arguments.
        CXXMethodDecl *Method = cast<CXXMethodDecl>(FnDecl);
        if (PerformObjectArgumentInitialization(Args[0], Method) ||
            PerformCopyInitialization(Args[1],
                                      FnDecl->getParamDecl(0)->getType(),
                                      AA_Passing))
          return ExprError();

        // Determine the result type
        QualType ResultTy
          = FnDecl->getType()->getAs<FunctionType>()->getResultType();
        ResultTy = ResultTy.getNonReferenceType();

        // Build the actual expression node.
        Expr *FnExpr = new (Context) DeclRefExpr(FnDecl, FnDecl->getType(),
                                                 LLoc);
        UsualUnaryConversions(FnExpr);

        Base.release();
        Idx.release();
        ExprOwningPtr<CXXOperatorCallExpr>
          TheCall(this, new (Context) CXXOperatorCallExpr(Context, OO_Subscript,
                                                          FnExpr, Args, 2,
                                                          ResultTy, RLoc));

        if (CheckCallReturnType(FnDecl->getResultType(), LLoc, TheCall.get(),
                                FnDecl))
          return ExprError();

        return MaybeBindToTemporary(TheCall.release());
      } else {
        // We matched a built-in operator. Convert the arguments, then
        // break out so that we will build the appropriate built-in
        // operator node.
        if (PerformImplicitConversion(Args[0], Best->BuiltinTypes.ParamTypes[0],
                                      Best->Conversions[0], AA_Passing) ||
            PerformImplicitConversion(Args[1], Best->BuiltinTypes.ParamTypes[1],
                                      Best->Conversions[1], AA_Passing))
          return ExprError();

        break;
      }
    }

    case OR_No_Viable_Function: {
      // No viable function; try to create a built-in operation, which will
      // produce an error. Then, show the non-viable candidates.
      OwningExprResult Result =
          CreateBuiltinArraySubscriptExpr(move(Base), LLoc, move(Idx), RLoc);
      assert(Result.isInvalid() && 
             "C++ subscript operator overloading is missing candidates!");
      if (Result.isInvalid())
        PrintOverloadCandidates(CandidateSet, /*OnlyViable=*/false,
                                "[]", LLoc);
      return move(Result);
    }

    case OR_Ambiguous:
      Diag(LLoc,  diag::err_ovl_ambiguous_oper)
          << "[]" << Args[0]->getSourceRange() << Args[1]->getSourceRange();
      PrintOverloadCandidates(CandidateSet, /*OnlyViable=*/true,
                              "[]", LLoc);
      return ExprError();

    case OR_Deleted:
      Diag(LLoc, diag::err_ovl_deleted_oper)
        << Best->Function->isDeleted() << "[]"
        << Args[0]->getSourceRange() << Args[1]->getSourceRange();
      PrintOverloadCandidates(CandidateSet, /*OnlyViable=*/true);
      return ExprError();
    }

  // We matched a built-in operator; build it.
  Base.release();
  Idx.release();
  return CreateBuiltinArraySubscriptExpr(Owned(Args[0]), LLoc,
                                         Owned(Args[1]), RLoc);
}

/// BuildCallToMemberFunction - Build a call to a member
/// function. MemExpr is the expression that refers to the member
/// function (and includes the object parameter), Args/NumArgs are the
/// arguments to the function call (not including the object
/// parameter). The caller needs to validate that the member
/// expression refers to a member function or an overloaded member
/// function.
Sema::OwningExprResult
Sema::BuildCallToMemberFunction(Scope *S, Expr *MemExprE,
                                SourceLocation LParenLoc, Expr **Args,
                                unsigned NumArgs, SourceLocation *CommaLocs,
                                SourceLocation RParenLoc) {
  // Dig out the member expression. This holds both the object
  // argument and the member function we're referring to.
  Expr *NakedMemExpr = MemExprE->IgnoreParens();
  
  MemberExpr *MemExpr;
  CXXMethodDecl *Method = 0;
  if (isa<MemberExpr>(NakedMemExpr)) {
    MemExpr = cast<MemberExpr>(NakedMemExpr);
    Method = cast<CXXMethodDecl>(MemExpr->getMemberDecl());
  } else {
    UnresolvedMemberExpr *UnresExpr = cast<UnresolvedMemberExpr>(NakedMemExpr);

    QualType ObjectType = UnresExpr->getBaseType();

    // Add overload candidates
    OverloadCandidateSet CandidateSet;

    // FIXME: avoid copy.
    TemplateArgumentListInfo TemplateArgsBuffer, *TemplateArgs = 0;
    if (UnresExpr->hasExplicitTemplateArgs()) {
      UnresExpr->copyTemplateArgumentsInto(TemplateArgsBuffer);
      TemplateArgs = &TemplateArgsBuffer;
    }

    for (UnresolvedMemberExpr::decls_iterator I = UnresExpr->decls_begin(),
           E = UnresExpr->decls_end(); I != E; ++I) {

      NamedDecl *Func = *I;
      CXXRecordDecl *ActingDC = cast<CXXRecordDecl>(Func->getDeclContext());
      if (isa<UsingShadowDecl>(Func))
        Func = cast<UsingShadowDecl>(Func)->getTargetDecl();

      if ((Method = dyn_cast<CXXMethodDecl>(Func))) {
        // If explicit template arguments were provided, we can't call a
        // non-template member function.
        if (TemplateArgs)
          continue;
        
        AddMethodCandidate(Method, ActingDC, ObjectType, Args, NumArgs,
                           CandidateSet, /*SuppressUserConversions=*/false);
      } else {
        AddMethodTemplateCandidate(cast<FunctionTemplateDecl>(Func),
                                   ActingDC, TemplateArgs,
                                   ObjectType, Args, NumArgs,
                                   CandidateSet,
                                   /*SuppressUsedConversions=*/false);
      }
    }

    DeclarationName DeclName = UnresExpr->getMemberName();

    OverloadCandidateSet::iterator Best;
    switch (BestViableFunction(CandidateSet, UnresExpr->getLocStart(), Best)) {
    case OR_Success:
      Method = cast<CXXMethodDecl>(Best->Function);
      break;

    case OR_No_Viable_Function:
      Diag(UnresExpr->getMemberLoc(),
           diag::err_ovl_no_viable_member_function_in_call)
        << DeclName << MemExprE->getSourceRange();
      PrintOverloadCandidates(CandidateSet, /*OnlyViable=*/false);
      // FIXME: Leaking incoming expressions!
      return ExprError();

    case OR_Ambiguous:
      Diag(UnresExpr->getMemberLoc(), diag::err_ovl_ambiguous_member_call)
        << DeclName << MemExprE->getSourceRange();
      PrintOverloadCandidates(CandidateSet, /*OnlyViable=*/false);
      // FIXME: Leaking incoming expressions!
      return ExprError();

    case OR_Deleted:
      Diag(UnresExpr->getMemberLoc(), diag::err_ovl_deleted_member_call)
        << Best->Function->isDeleted()
        << DeclName << MemExprE->getSourceRange();
      PrintOverloadCandidates(CandidateSet, /*OnlyViable=*/false);
      // FIXME: Leaking incoming expressions!
      return ExprError();
    }

    MemExprE = FixOverloadedFunctionReference(MemExprE, Method);

    // If overload resolution picked a static member, build a
    // non-member call based on that function.
    if (Method->isStatic()) {
      return BuildResolvedCallExpr(MemExprE, Method, LParenLoc,
                                   Args, NumArgs, RParenLoc);
    }

    MemExpr = cast<MemberExpr>(MemExprE->IgnoreParens());
  }

  assert(Method && "Member call to something that isn't a method?");
  ExprOwningPtr<CXXMemberCallExpr>
    TheCall(this, new (Context) CXXMemberCallExpr(Context, MemExprE, Args,
                                                  NumArgs,
                                  Method->getResultType().getNonReferenceType(),
                                  RParenLoc));

  // Check for a valid return type.
  if (CheckCallReturnType(Method->getResultType(), MemExpr->getMemberLoc(), 
                          TheCall.get(), Method))
    return ExprError();
  
  // Convert the object argument (for a non-static member function call).
  Expr *ObjectArg = MemExpr->getBase();
  if (!Method->isStatic() &&
      PerformObjectArgumentInitialization(ObjectArg, Method))
    return ExprError();
  MemExpr->setBase(ObjectArg);

  // Convert the rest of the arguments
  const FunctionProtoType *Proto = cast<FunctionProtoType>(Method->getType());
  if (ConvertArgumentsForCall(&*TheCall, MemExpr, Method, Proto, Args, NumArgs,
                              RParenLoc))
    return ExprError();

  if (CheckFunctionCall(Method, TheCall.get()))
    return ExprError();

  return MaybeBindToTemporary(TheCall.release());
}

/// BuildCallToObjectOfClassType - Build a call to an object of class
/// type (C++ [over.call.object]), which can end up invoking an
/// overloaded function call operator (@c operator()) or performing a
/// user-defined conversion on the object argument.
Sema::ExprResult
Sema::BuildCallToObjectOfClassType(Scope *S, Expr *Object,
                                   SourceLocation LParenLoc,
                                   Expr **Args, unsigned NumArgs,
                                   SourceLocation *CommaLocs,
                                   SourceLocation RParenLoc) {
  assert(Object->getType()->isRecordType() && "Requires object type argument");
  const RecordType *Record = Object->getType()->getAs<RecordType>();

  // C++ [over.call.object]p1:
  //  If the primary-expression E in the function call syntax
  //  evaluates to a class object of type "cv T", then the set of
  //  candidate functions includes at least the function call
  //  operators of T. The function call operators of T are obtained by
  //  ordinary lookup of the name operator() in the context of
  //  (E).operator().
  OverloadCandidateSet CandidateSet;
  DeclarationName OpName = Context.DeclarationNames.getCXXOperatorName(OO_Call);

  if (RequireCompleteType(LParenLoc, Object->getType(), 
                          PartialDiagnostic(diag::err_incomplete_object_call)
                          << Object->getSourceRange()))
    return true;
  
  LookupResult R(*this, OpName, LParenLoc, LookupOrdinaryName);
  LookupQualifiedName(R, Record->getDecl());
  R.suppressDiagnostics();

  for (LookupResult::iterator Oper = R.begin(), OperEnd = R.end();
       Oper != OperEnd; ++Oper) {
    AddMethodCandidate(*Oper, Object->getType(), Args, NumArgs, CandidateSet,
                       /*SuppressUserConversions=*/ false);
  }
  
  // C++ [over.call.object]p2:
  //   In addition, for each conversion function declared in T of the
  //   form
  //
  //        operator conversion-type-id () cv-qualifier;
  //
  //   where cv-qualifier is the same cv-qualification as, or a
  //   greater cv-qualification than, cv, and where conversion-type-id
  //   denotes the type "pointer to function of (P1,...,Pn) returning
  //   R", or the type "reference to pointer to function of
  //   (P1,...,Pn) returning R", or the type "reference to function
  //   of (P1,...,Pn) returning R", a surrogate call function [...]
  //   is also considered as a candidate function. Similarly,
  //   surrogate call functions are added to the set of candidate
  //   functions for each conversion function declared in an
  //   accessible base class provided the function is not hidden
  //   within T by another intervening declaration.
  // FIXME: Look in base classes for more conversion operators!
  const UnresolvedSet *Conversions
    = cast<CXXRecordDecl>(Record->getDecl())->getConversionFunctions();
  for (UnresolvedSet::iterator I = Conversions->begin(),
         E = Conversions->end(); I != E; ++I) {
    NamedDecl *D = *I;
    CXXRecordDecl *ActingContext = cast<CXXRecordDecl>(D->getDeclContext());
    if (isa<UsingShadowDecl>(D))
      D = cast<UsingShadowDecl>(D)->getTargetDecl();
    
    // Skip over templated conversion functions; they aren't
    // surrogates.
    if (isa<FunctionTemplateDecl>(D))
      continue;

    CXXConversionDecl *Conv = cast<CXXConversionDecl>(D);

    // Strip the reference type (if any) and then the pointer type (if
    // any) to get down to what might be a function type.
    QualType ConvType = Conv->getConversionType().getNonReferenceType();
    if (const PointerType *ConvPtrType = ConvType->getAs<PointerType>())
      ConvType = ConvPtrType->getPointeeType();

    if (const FunctionProtoType *Proto = ConvType->getAs<FunctionProtoType>())
      AddSurrogateCandidate(Conv, ActingContext, Proto,
                            Object->getType(), Args, NumArgs,
                            CandidateSet);
  }

  // Perform overload resolution.
  OverloadCandidateSet::iterator Best;
  switch (BestViableFunction(CandidateSet, Object->getLocStart(), Best)) {
  case OR_Success:
    // Overload resolution succeeded; we'll build the appropriate call
    // below.
    break;

  case OR_No_Viable_Function:
    Diag(Object->getSourceRange().getBegin(),
         diag::err_ovl_no_viable_object_call)
      << Object->getType() << Object->getSourceRange();
    PrintOverloadCandidates(CandidateSet, /*OnlyViable=*/false);
    break;

  case OR_Ambiguous:
    Diag(Object->getSourceRange().getBegin(),
         diag::err_ovl_ambiguous_object_call)
      << Object->getType() << Object->getSourceRange();
    PrintOverloadCandidates(CandidateSet, /*OnlyViable=*/true);
    break;

  case OR_Deleted:
    Diag(Object->getSourceRange().getBegin(),
         diag::err_ovl_deleted_object_call)
      << Best->Function->isDeleted()
      << Object->getType() << Object->getSourceRange();
    PrintOverloadCandidates(CandidateSet, /*OnlyViable=*/true);
    break;
  }

  if (Best == CandidateSet.end()) {
    // We had an error; delete all of the subexpressions and return
    // the error.
    Object->Destroy(Context);
    for (unsigned ArgIdx = 0; ArgIdx < NumArgs; ++ArgIdx)
      Args[ArgIdx]->Destroy(Context);
    return true;
  }

  if (Best->Function == 0) {
    // Since there is no function declaration, this is one of the
    // surrogate candidates. Dig out the conversion function.
    CXXConversionDecl *Conv
      = cast<CXXConversionDecl>(
                         Best->Conversions[0].UserDefined.ConversionFunction);

    // We selected one of the surrogate functions that converts the
    // object parameter to a function pointer. Perform the conversion
    // on the object argument, then let ActOnCallExpr finish the job.
    
    // Create an implicit member expr to refer to the conversion operator.
    // and then call it.
    CXXMemberCallExpr *CE = BuildCXXMemberCallExpr(Object, Conv);
      
    return ActOnCallExpr(S, ExprArg(*this, CE), LParenLoc,
                         MultiExprArg(*this, (ExprTy**)Args, NumArgs),
                         CommaLocs, RParenLoc).release();
  }

  // We found an overloaded operator(). Build a CXXOperatorCallExpr
  // that calls this method, using Object for the implicit object
  // parameter and passing along the remaining arguments.
  CXXMethodDecl *Method = cast<CXXMethodDecl>(Best->Function);
  const FunctionProtoType *Proto = Method->getType()->getAs<FunctionProtoType>();

  unsigned NumArgsInProto = Proto->getNumArgs();
  unsigned NumArgsToCheck = NumArgs;

  // Build the full argument list for the method call (the
  // implicit object parameter is placed at the beginning of the
  // list).
  Expr **MethodArgs;
  if (NumArgs < NumArgsInProto) {
    NumArgsToCheck = NumArgsInProto;
    MethodArgs = new Expr*[NumArgsInProto + 1];
  } else {
    MethodArgs = new Expr*[NumArgs + 1];
  }
  MethodArgs[0] = Object;
  for (unsigned ArgIdx = 0; ArgIdx < NumArgs; ++ArgIdx)
    MethodArgs[ArgIdx + 1] = Args[ArgIdx];

  Expr *NewFn = new (Context) DeclRefExpr(Method, Method->getType(),
                                          SourceLocation());
  UsualUnaryConversions(NewFn);

  // Once we've built TheCall, all of the expressions are properly
  // owned.
  QualType ResultTy = Method->getResultType().getNonReferenceType();
  ExprOwningPtr<CXXOperatorCallExpr>
    TheCall(this, new (Context) CXXOperatorCallExpr(Context, OO_Call, NewFn,
                                                    MethodArgs, NumArgs + 1,
                                                    ResultTy, RParenLoc));
  delete [] MethodArgs;

  if (CheckCallReturnType(Method->getResultType(), LParenLoc, TheCall.get(), 
                          Method))
    return true;
  
  // We may have default arguments. If so, we need to allocate more
  // slots in the call for them.
  if (NumArgs < NumArgsInProto)
    TheCall->setNumArgs(Context, NumArgsInProto + 1);
  else if (NumArgs > NumArgsInProto)
    NumArgsToCheck = NumArgsInProto;

  bool IsError = false;

  // Initialize the implicit object parameter.
  IsError |= PerformObjectArgumentInitialization(Object, Method);
  TheCall->setArg(0, Object);


  // Check the argument types.
  for (unsigned i = 0; i != NumArgsToCheck; i++) {
    Expr *Arg;
    if (i < NumArgs) {
      Arg = Args[i];

      // Pass the argument.
      QualType ProtoArgType = Proto->getArgType(i);
      IsError |= PerformCopyInitialization(Arg, ProtoArgType, AA_Passing);
    } else {
      OwningExprResult DefArg
        = BuildCXXDefaultArgExpr(LParenLoc, Method, Method->getParamDecl(i));
      if (DefArg.isInvalid()) {
        IsError = true;
        break;
      }
      
      Arg = DefArg.takeAs<Expr>();
    }

    TheCall->setArg(i + 1, Arg);
  }

  // If this is a variadic call, handle args passed through "...".
  if (Proto->isVariadic()) {
    // Promote the arguments (C99 6.5.2.2p7).
    for (unsigned i = NumArgsInProto; i != NumArgs; i++) {
      Expr *Arg = Args[i];
      IsError |= DefaultVariadicArgumentPromotion(Arg, VariadicMethod);
      TheCall->setArg(i + 1, Arg);
    }
  }

  if (IsError) return true;

  if (CheckFunctionCall(Method, TheCall.get()))
    return true;

  return MaybeBindToTemporary(TheCall.release()).release();
}

/// BuildOverloadedArrowExpr - Build a call to an overloaded @c operator->
///  (if one exists), where @c Base is an expression of class type and
/// @c Member is the name of the member we're trying to find.
Sema::OwningExprResult
Sema::BuildOverloadedArrowExpr(Scope *S, ExprArg BaseIn, SourceLocation OpLoc) {
  Expr *Base = static_cast<Expr *>(BaseIn.get());
  assert(Base->getType()->isRecordType() && "left-hand side must have class type");

  // C++ [over.ref]p1:
  //
  //   [...] An expression x->m is interpreted as (x.operator->())->m
  //   for a class object x of type T if T::operator->() exists and if
  //   the operator is selected as the best match function by the
  //   overload resolution mechanism (13.3).
  DeclarationName OpName = Context.DeclarationNames.getCXXOperatorName(OO_Arrow);
  OverloadCandidateSet CandidateSet;
  const RecordType *BaseRecord = Base->getType()->getAs<RecordType>();

  if (RequireCompleteType(Base->getLocStart(), Base->getType(),
                          PDiag(diag::err_typecheck_incomplete_tag)
                            << Base->getSourceRange()))
    return ExprError();

  LookupResult R(*this, OpName, OpLoc, LookupOrdinaryName);
  LookupQualifiedName(R, BaseRecord->getDecl());
  R.suppressDiagnostics();

  for (LookupResult::iterator Oper = R.begin(), OperEnd = R.end();
       Oper != OperEnd; ++Oper) {
    NamedDecl *D = *Oper;
    CXXRecordDecl *ActingContext = cast<CXXRecordDecl>(D->getDeclContext());
    if (isa<UsingShadowDecl>(D))
      D = cast<UsingShadowDecl>(D)->getTargetDecl();

    AddMethodCandidate(cast<CXXMethodDecl>(D), ActingContext,
                       Base->getType(), 0, 0, CandidateSet,
                       /*SuppressUserConversions=*/false);
  }

  // Perform overload resolution.
  OverloadCandidateSet::iterator Best;
  switch (BestViableFunction(CandidateSet, OpLoc, Best)) {
  case OR_Success:
    // Overload resolution succeeded; we'll build the call below.
    break;

  case OR_No_Viable_Function:
    if (CandidateSet.empty())
      Diag(OpLoc, diag::err_typecheck_member_reference_arrow)
        << Base->getType() << Base->getSourceRange();
    else
      Diag(OpLoc, diag::err_ovl_no_viable_oper)
        << "operator->" << Base->getSourceRange();
    PrintOverloadCandidates(CandidateSet, /*OnlyViable=*/false);
    return ExprError();

  case OR_Ambiguous:
    Diag(OpLoc,  diag::err_ovl_ambiguous_oper)
      << "->" << Base->getSourceRange();
    PrintOverloadCandidates(CandidateSet, /*OnlyViable=*/true);
    return ExprError();

  case OR_Deleted:
    Diag(OpLoc,  diag::err_ovl_deleted_oper)
      << Best->Function->isDeleted()
      << "->" << Base->getSourceRange();
    PrintOverloadCandidates(CandidateSet, /*OnlyViable=*/true);
    return ExprError();
  }

  // Convert the object parameter.
  CXXMethodDecl *Method = cast<CXXMethodDecl>(Best->Function);
  if (PerformObjectArgumentInitialization(Base, Method))
    return ExprError();

  // No concerns about early exits now.
  BaseIn.release();

  // Build the operator call.
  Expr *FnExpr = new (Context) DeclRefExpr(Method, Method->getType(),
                                           SourceLocation());
  UsualUnaryConversions(FnExpr);
  
  QualType ResultTy = Method->getResultType().getNonReferenceType();
  ExprOwningPtr<CXXOperatorCallExpr> 
    TheCall(this, new (Context) CXXOperatorCallExpr(Context, OO_Arrow, FnExpr, 
                                                    &Base, 1, ResultTy, OpLoc));

  if (CheckCallReturnType(Method->getResultType(), OpLoc, TheCall.get(), 
                          Method))
          return ExprError();
  return move(TheCall);
}

/// FixOverloadedFunctionReference - E is an expression that refers to
/// a C++ overloaded function (possibly with some parentheses and
/// perhaps a '&' around it). We have resolved the overloaded function
/// to the function declaration Fn, so patch up the expression E to
/// refer (possibly indirectly) to Fn. Returns the new expr.
Expr *Sema::FixOverloadedFunctionReference(Expr *E, FunctionDecl *Fn) {
  if (ParenExpr *PE = dyn_cast<ParenExpr>(E)) {
    Expr *SubExpr = FixOverloadedFunctionReference(PE->getSubExpr(), Fn);
    if (SubExpr == PE->getSubExpr())
      return PE->Retain();
    
    return new (Context) ParenExpr(PE->getLParen(), PE->getRParen(), SubExpr);
  } 
  
  if (ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(E)) {
    Expr *SubExpr = FixOverloadedFunctionReference(ICE->getSubExpr(), Fn);
    assert(Context.hasSameType(ICE->getSubExpr()->getType(), 
                               SubExpr->getType()) &&
           "Implicit cast type cannot be determined from overload");
    if (SubExpr == ICE->getSubExpr())
      return ICE->Retain();
    
    return new (Context) ImplicitCastExpr(ICE->getType(), 
                                          ICE->getCastKind(),
                                          SubExpr,
                                          ICE->isLvalueCast());
  } 
  
  if (UnaryOperator *UnOp = dyn_cast<UnaryOperator>(E)) {
    assert(UnOp->getOpcode() == UnaryOperator::AddrOf &&
           "Can only take the address of an overloaded function");
    if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(Fn)) {
      if (Method->isStatic()) {
        // Do nothing: static member functions aren't any different
        // from non-member functions.
      } else {
        // Fix the sub expression, which really has to be an
        // UnresolvedLookupExpr holding an overloaded member function
        // or template.
        Expr *SubExpr = FixOverloadedFunctionReference(UnOp->getSubExpr(), Fn);
        if (SubExpr == UnOp->getSubExpr())
          return UnOp->Retain();

        assert(isa<DeclRefExpr>(SubExpr)
               && "fixed to something other than a decl ref");
        assert(cast<DeclRefExpr>(SubExpr)->getQualifier()
               && "fixed to a member ref with no nested name qualifier");

        // We have taken the address of a pointer to member
        // function. Perform the computation here so that we get the
        // appropriate pointer to member type.
        QualType ClassType
          = Context.getTypeDeclType(cast<RecordDecl>(Method->getDeclContext()));
        QualType MemPtrType
          = Context.getMemberPointerType(Fn->getType(), ClassType.getTypePtr());

        return new (Context) UnaryOperator(SubExpr, UnaryOperator::AddrOf,
                                           MemPtrType, UnOp->getOperatorLoc());
      }
    }
    Expr *SubExpr = FixOverloadedFunctionReference(UnOp->getSubExpr(), Fn);
    if (SubExpr == UnOp->getSubExpr())
      return UnOp->Retain();
    
    return new (Context) UnaryOperator(SubExpr, UnaryOperator::AddrOf,
                                     Context.getPointerType(SubExpr->getType()),
                                       UnOp->getOperatorLoc());
  } 

  if (UnresolvedLookupExpr *ULE = dyn_cast<UnresolvedLookupExpr>(E)) {
    // FIXME: avoid copy.
    TemplateArgumentListInfo TemplateArgsBuffer, *TemplateArgs = 0;
    if (ULE->hasExplicitTemplateArgs()) {
      ULE->copyTemplateArgumentsInto(TemplateArgsBuffer);
      TemplateArgs = &TemplateArgsBuffer;
    }

    return DeclRefExpr::Create(Context,
                               ULE->getQualifier(),
                               ULE->getQualifierRange(),
                               Fn,
                               ULE->getNameLoc(),
                               Fn->getType(),
                               TemplateArgs);
  }

  if (UnresolvedMemberExpr *MemExpr = dyn_cast<UnresolvedMemberExpr>(E)) {
    // FIXME: avoid copy.
    TemplateArgumentListInfo TemplateArgsBuffer, *TemplateArgs = 0;
    if (MemExpr->hasExplicitTemplateArgs()) {
      MemExpr->copyTemplateArgumentsInto(TemplateArgsBuffer);
      TemplateArgs = &TemplateArgsBuffer;
    }

    Expr *Base;

    // If we're filling in 
    if (MemExpr->isImplicitAccess()) {
      if (cast<CXXMethodDecl>(Fn)->isStatic()) {
        return DeclRefExpr::Create(Context,
                                   MemExpr->getQualifier(),
                                   MemExpr->getQualifierRange(),
                                   Fn,
                                   MemExpr->getMemberLoc(),
                                   Fn->getType(),
                                   TemplateArgs);
      } else
        Base = new (Context) CXXThisExpr(SourceLocation(),
                                         MemExpr->getBaseType());
    } else
      Base = MemExpr->getBase()->Retain();

    return MemberExpr::Create(Context, Base,
                              MemExpr->isArrow(), 
                              MemExpr->getQualifier(), 
                              MemExpr->getQualifierRange(),
                              Fn, 
                              MemExpr->getMemberLoc(),
                              TemplateArgs,
                              Fn->getType());
  }
  
  assert(false && "Invalid reference to overloaded function");
  return E->Retain();
}

Sema::OwningExprResult Sema::FixOverloadedFunctionReference(OwningExprResult E, 
                                                            FunctionDecl *Fn) {
  return Owned(FixOverloadedFunctionReference((Expr *)E.get(), Fn));
}

} // end namespace clang
