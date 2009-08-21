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
#include "SemaInherit.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/TypeOrdering.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Compiler.h"
#include <algorithm>

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
  static const char* Name[(int)ICK_Num_Conversion_Kinds] = {
    "No conversion",
    "Lvalue-to-rvalue",
    "Array-to-pointer",
    "Function-to-pointer",
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
bool StandardConversionSequence::isPointerConversionToBool() const
{
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
isPointerConversionToVoidPointer(ASTContext& Context) const
{
  QualType FromType = QualType::getFromOpaquePtr(FromTypePtr);
  QualType ToType = QualType::getFromOpaquePtr(ToTypePtr);

  // Note that FromType has not necessarily been transformed by the
  // array-to-pointer implicit conversion, so check for its presence
  // and redo the conversion to get a pointer.
  if (First == ICK_Array_To_Pointer)
    FromType = Context.getArrayDecayedType(FromType);

  if (Second == ICK_Pointer_Conversion)
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
// overload of the Old declaration. This routine returns false if New
// and Old cannot be overloaded, e.g., if they are functions with the
// same signature (C++ 1.3.10) or if the Old declaration isn't a
// function (or overload set). When it does return false and Old is an
// OverloadedFunctionDecl, MatchedDecl will be set to point to the
// FunctionDecl that New cannot be overloaded with. 
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
// When we process #2, Old is a FunctionDecl for #1.  By comparing the
// parameter types, we see that #1 and #2 are overloaded (since they
// have different signatures), so this routine returns false;
// MatchedDecl is unchanged.
//
// When we process #3, Old is an OverloadedFunctionDecl containing #1
// and #2. We compare the signatures of #3 to #1 (they're overloaded,
// so we do nothing) and then #3 to #2. Since the signatures of #3 and
// #2 are identical (return types of functions are not part of the
// signature), IsOverload returns false and MatchedDecl will be set to
// point to the FunctionDecl for #2.
bool
Sema::IsOverload(FunctionDecl *New, Decl* OldD, 
                 OverloadedFunctionDecl::function_iterator& MatchedDecl)
{
  if (OverloadedFunctionDecl* Ovl = dyn_cast<OverloadedFunctionDecl>(OldD)) {
    // Is this new function an overload of every function in the
    // overload set?
    OverloadedFunctionDecl::function_iterator Func = Ovl->function_begin(),
                                           FuncEnd = Ovl->function_end();
    for (; Func != FuncEnd; ++Func) {
      if (!IsOverload(New, *Func, MatchedDecl)) {
        MatchedDecl = Func;
        return false;
      }
    }

    // This function overloads every function in the overload set.
    return true;
  } else if (FunctionTemplateDecl *Old = dyn_cast<FunctionTemplateDecl>(OldD))
    return IsOverload(New, Old->getTemplatedDecl(), MatchedDecl);
  else if (FunctionDecl* Old = dyn_cast<FunctionDecl>(OldD)) {
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
                                         false, false, SourceLocation()) ||
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
  } else {
    // (C++ 13p1):
    //   Only function declarations can be overloaded; object and type
    //   declarations cannot be overloaded.
    return false;
  }
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
ImplicitConversionSequence
Sema::TryImplicitConversion(Expr* From, QualType ToType,
                            bool SuppressUserConversions,
                            bool AllowExplicit, bool ForceRValue)
{
  ImplicitConversionSequence ICS;
  if (IsStandardConversion(From, ToType, ICS.Standard))
    ICS.ConversionKind = ImplicitConversionSequence::StandardConversion;
  else if (getLangOptions().CPlusPlus &&
           IsUserDefinedConversion(From, ToType, ICS.UserDefined, 
                                   !SuppressUserConversions, AllowExplicit,
                                   ForceRValue)) {
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
      if (FromCanon == ToCanon || IsDerivedFrom(FromCanon, ToCanon)) {
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
  } else
    ICS.ConversionKind = ImplicitConversionSequence::BadConversion;

  return ICS;
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
                           StandardConversionSequence &SCS)
{
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

    // FIXME: Doesn't see through to qualifiers behind a typedef!
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
  } else if (IsPointerConversion(From, FromType, ToType, FromType, 
                                 IncompatibleObjC)) {
    // Pointer conversions (C++ 4.10).
    SCS.Second = ICK_Pointer_Conversion;
    SCS.IncompatibleObjC = IncompatibleObjC;
  } else if (IsMemberPointerConversion(From, FromType, ToType, FromType)) {
    // Pointer to member conversions (4.11).
    SCS.Second = ICK_Pointer_Member;
  } else if (ToType->isBooleanType() &&
             (FromType->isArithmeticType() ||
              FromType->isEnumeralType() ||
              FromType->isPointerType() ||
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
    if (CanonFrom.getUnqualifiedType() == CanonTo.getUnqualifiedType() &&
        CanonFrom.getCVRQualifiers() != CanonTo.getCVRQualifiers()) {
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
bool Sema::IsIntegralPromotion(Expr *From, QualType FromType, QualType ToType)
{
  const BuiltinType *To = ToType->getAsBuiltinType();
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
  if ((FromType->isEnumeralType() || FromType->isWideCharType())
      && ToType->isIntegerType()) {
    // Determine whether the type we're converting from is signed or
    // unsigned.
    bool FromIsSigned;
    uint64_t FromSize = Context.getTypeSize(FromType);
    if (const EnumType *FromEnumType = FromType->getAsEnumType()) {
      QualType UnderlyingType = FromEnumType->getDecl()->getIntegerType();
      FromIsSigned = UnderlyingType->isSignedIntegerType();
    } else {
      // FIXME: Is wchar_t signed or unsigned? We assume it's signed for now.
      FromIsSigned = true;
    }

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
        return Context.getCanonicalType(ToType).getUnqualifiedType()
          == Context.getCanonicalType(PromoteTypes[Idx]).getUnqualifiedType();
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
bool Sema::IsFloatingPointPromotion(QualType FromType, QualType ToType)
{
  /// An rvalue of type float can be converted to an rvalue of type
  /// double. (C++ 4.6p1).
  if (const BuiltinType *FromBuiltin = FromType->getAsBuiltinType())
    if (const BuiltinType *ToBuiltin = ToType->getAsBuiltinType()) {
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
  const ComplexType *FromComplex = FromType->getAsComplexType();
  if (!FromComplex)
    return false;

  const ComplexType *ToComplex = ToType->getAsComplexType();
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
  unsigned Quals = CanonFromPointee.getCVRQualifiers();
  
  // Exact qualifier match -> return the pointer type we're converting to.  
  if (CanonToPointee.getCVRQualifiers() == Quals) {
    // ToType is exactly what we need. Return it.
    if (ToType.getTypePtr())
      return ToType;

    // Build a pointer to ToPointee. It has the right qualifiers
    // already.
    return Context.getPointerType(ToPointee);
  }

  // Just build a canonical type that has the right qualifiers.
  return Context.getPointerType(CanonToPointee.getQualifiedType(Quals));
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
                               QualType& ConvertedType,
                               bool &IncompatibleObjC)
{
  IncompatibleObjC = false;
  if (isObjCPointerConversion(FromType, ToType, ConvertedType, IncompatibleObjC))
    return true;

  // Conversion from a null pointer constant to any Objective-C pointer type. 
  if (ToType->isObjCObjectPointerType() && 
      From->isNullPointerConstant(Context)) {
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
  if (ToType->isBlockPointerType() && From->isNullPointerConstant(Context)) {
    ConvertedType = ToType;
    return true;
  }

  // If the left-hand-side is nullptr_t, the right side can be a null
  // pointer constant.
  if (ToType->isNullPtrType() && From->isNullPointerConstant(Context)) {
    ConvertedType = ToType;
    return true;
  }

  const PointerType* ToTypePtr = ToType->getAs<PointerType>();
  if (!ToTypePtr)
    return false;

  // A null pointer constant can be converted to a pointer type (C++ 4.10p1).
  if (From->isNullPointerConstant(Context)) {
    ConvertedType = ToType;
    return true;
  }

  // Beyond this point, both types need to be pointers.
  const PointerType *FromTypePtr = FromType->getAs<PointerType>();
  if (!FromTypePtr)
    return false;

  QualType FromPointeeType = FromTypePtr->getPointeeType();
  QualType ToPointeeType = ToTypePtr->getPointeeType();

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
  const ObjCObjectPointerType* ToObjCPtr = ToType->getAsObjCObjectPointerType();
  const ObjCObjectPointerType *FromObjCPtr = 
    FromType->getAsObjCObjectPointerType();

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
    = FromPointeeType->getAsFunctionProtoType();
  const FunctionProtoType *ToFunctionType
    = ToPointeeType->getAsFunctionProtoType();
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
bool Sema::CheckPointerConversion(Expr *From, QualType ToType) {
  QualType FromType = From->getType();

  if (const PointerType *FromPtrType = FromType->getAs<PointerType>())
    if (const PointerType *ToPtrType = ToType->getAs<PointerType>()) {
      QualType FromPointeeType = FromPtrType->getPointeeType(),
               ToPointeeType   = ToPtrType->getPointeeType();

      if (FromPointeeType->isRecordType() &&
          ToPointeeType->isRecordType()) {
        // We must have a derived-to-base conversion. Check an
        // ambiguous or inaccessible conversion.
        return CheckDerivedToBaseConversion(FromPointeeType, ToPointeeType,
                                            From->getExprLoc(),
                                            From->getSourceRange());
      }
    }
  if (const ObjCObjectPointerType *FromPtrType = 
        FromType->getAsObjCObjectPointerType())
    if (const ObjCObjectPointerType *ToPtrType = 
          ToType->getAsObjCObjectPointerType()) {
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
                                     QualType ToType, QualType &ConvertedType)
{
  const MemberPointerType *ToTypePtr = ToType->getAs<MemberPointerType>();
  if (!ToTypePtr)
    return false;

  // A null pointer constant can be converted to a member pointer (C++ 4.11p1)
  if (From->isNullPointerConstant(Context)) {
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
bool Sema::CheckMemberPointerConversion(Expr *From, QualType ToType) {
  QualType FromType = From->getType();
  const MemberPointerType *FromPtrType = FromType->getAs<MemberPointerType>();
  if (!FromPtrType)
    return false;

  const MemberPointerType *ToPtrType = ToType->getAs<MemberPointerType>();
  assert(ToPtrType && "No member pointer cast has a target type "
                      "that is not a member pointer.");

  QualType FromClass = QualType(FromPtrType->getClass(), 0);
  QualType ToClass   = QualType(ToPtrType->getClass(), 0);

  // FIXME: What about dependent types?
  assert(FromClass->isRecordType() && "Pointer into non-class.");
  assert(ToClass->isRecordType() && "Pointer into non-class.");

  BasePaths Paths(/*FindAmbiguities=*/true, /*RecordPaths=*/false,
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

  return false;
}

/// IsQualificationConversion - Determines whether the conversion from
/// an rvalue of type FromType to ToType is a qualification conversion
/// (C++ 4.4).
bool 
Sema::IsQualificationConversion(QualType FromType, QualType ToType)
{
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
  return UnwrappedAnyPointer &&
    FromType.getUnqualifiedType() == ToType.getUnqualifiedType();
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
bool Sema::IsUserDefinedConversion(Expr *From, QualType ToType, 
                                   UserDefinedConversionSequence& User,
                                   bool AllowConversionFunctions,
                                   bool AllowExplicit, bool ForceRValue)
{
  OverloadCandidateSet CandidateSet;
  if (const RecordType *ToRecordType = ToType->getAs<RecordType>()) {
    if (CXXRecordDecl *ToRecordDecl 
          = dyn_cast<CXXRecordDecl>(ToRecordType->getDecl())) {
      // C++ [over.match.ctor]p1:
      //   When objects of class type are direct-initialized (8.5), or
      //   copy-initialized from an expression of the same or a
      //   derived class type (8.5), overload resolution selects the
      //   constructor. [...] For copy-initialization, the candidate
      //   functions are all the converting constructors (12.3.1) of
      //   that class. The argument list is the expression-list within
      //   the parentheses of the initializer.
      DeclarationName ConstructorName 
        = Context.DeclarationNames.getCXXConstructorName(
                          Context.getCanonicalType(ToType).getUnqualifiedType());
      DeclContext::lookup_iterator Con, ConEnd;
      for (llvm::tie(Con, ConEnd) 
             = ToRecordDecl->lookup(ConstructorName);
           Con != ConEnd; ++Con) {
        CXXConstructorDecl *Constructor = cast<CXXConstructorDecl>(*Con);
        if (!Constructor->isInvalidDecl() &&
            Constructor->isConvertingConstructor())
          AddOverloadCandidate(Constructor, &From, 1, CandidateSet,
                               /*SuppressUserConversions=*/true, ForceRValue);
      }
    }
  }

  if (!AllowConversionFunctions) {
    // Don't allow any conversion functions to enter the overload set.
  } else if (const RecordType *FromRecordType 
               = From->getType()->getAs<RecordType>()) {
    if (CXXRecordDecl *FromRecordDecl 
          = dyn_cast<CXXRecordDecl>(FromRecordType->getDecl())) {
      // Add all of the conversion functions as candidates.
      // FIXME: Look for conversions in base classes!
      OverloadedFunctionDecl *Conversions 
        = FromRecordDecl->getConversionFunctions();
      for (OverloadedFunctionDecl::function_iterator Func 
             = Conversions->function_begin();
           Func != Conversions->function_end(); ++Func) {
        CXXConversionDecl *Conv = cast<CXXConversionDecl>(*Func);
        if (AllowExplicit || !Conv->isExplicit())
          AddConversionCandidate(Conv, From, ToType, CandidateSet);
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
        // FIXME: What about ellipsis conversions?
        QualType ThisType = Constructor->getThisType(Context);
        User.Before = Best->Conversions[0].Standard;
        User.ConversionFunction = Constructor;
        User.After.setAsIdentityConversion();
        User.After.FromTypePtr 
          = ThisType->getAs<PointerType>()->getPointeeType().getAsOpaquePtr();
        User.After.ToTypePtr = ToType.getAsOpaquePtr();
        return true;
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
        return true;
      } else {
        assert(false && "Not a constructor or conversion function?");
        return false;
      }
      
    case OR_No_Viable_Function:
    case OR_Deleted:
      // No conversion here! We're done.
      return false;

    case OR_Ambiguous:
      // FIXME: See C++ [over.best.ics]p10 for the handling of
      // ambiguous conversion sequences.
      return false;
    }

  return false;
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
    const ObjCInterfaceType* FromIface1 = FromPointee1->getAsObjCInterfaceType();
    const ObjCInterfaceType* FromIface2 = FromPointee2->getAsObjCInterfaceType();
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
    if (T1.getUnqualifiedType() == T2.getUnqualifiedType()) {
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
                                      const StandardConversionSequence& SCS2)
{
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

  // If the types are the same, we won't learn anything by unwrapped
  // them.
  if (T1.getUnqualifiedType() == T2.getUnqualifiedType())
    return ImplicitConversionSequence::Indistinguishable;

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
    if (T1.getUnqualifiedType() == T2.getUnqualifiedType())
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

    const ObjCInterfaceType* FromIface1 = FromPointee1->getAsObjCInterfaceType();
    const ObjCInterfaceType* FromIface2 = FromPointee2->getAsObjCInterfaceType();
    const ObjCInterfaceType* ToIface1 = ToPointee1->getAsObjCInterfaceType();
    const ObjCInterfaceType* ToIface2 = ToPointee2->getAsObjCInterfaceType();

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
    if (FromType1.getUnqualifiedType() == FromType2.getUnqualifiedType() &&
        ToType1.getUnqualifiedType() != ToType2.getUnqualifiedType()) {
      if (IsDerivedFrom(ToType1, ToType2))
        return ImplicitConversionSequence::Better;
      else if (IsDerivedFrom(ToType2, ToType1))
        return ImplicitConversionSequence::Worse;
    }

    //   -- binding of an expression of type B to a reference of type
    //      A& is better than binding an expression of type C to a
    //      reference of type A&,
    if (FromType1.getUnqualifiedType() != FromType2.getUnqualifiedType() &&
        ToType1.getUnqualifiedType() == ToType2.getUnqualifiedType()) {
      if (IsDerivedFrom(FromType2, FromType1))
        return ImplicitConversionSequence::Better;
      else if (IsDerivedFrom(FromType1, FromType2))
        return ImplicitConversionSequence::Worse;
    }
  }


  // FIXME: conversion of A::* to B::* is better than conversion of
  // A::* to C::*,

  // FIXME: conversion of B::* to C::* is better than conversion of
  // A::* to C::*, and

  if (SCS1.CopyConstructor && SCS2.CopyConstructor &&
      SCS1.Second == ICK_Derived_To_Base) {
    //   -- conversion of C to B is better than conversion of C to A,
    if (FromType1.getUnqualifiedType() == FromType2.getUnqualifiedType() &&
        ToType1.getUnqualifiedType() != ToType2.getUnqualifiedType()) {
      if (IsDerivedFrom(ToType1, ToType2))
        return ImplicitConversionSequence::Better;
      else if (IsDerivedFrom(ToType2, ToType1))
        return ImplicitConversionSequence::Worse;
    }

    //   -- conversion of B to A is better than conversion of C to A.
    if (FromType1.getUnqualifiedType() != FromType2.getUnqualifiedType() &&
        ToType1.getUnqualifiedType() == ToType2.getUnqualifiedType()) {
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
                            bool SuppressUserConversions, bool ForceRValue) {
  if (ToType->isReferenceType()) {
    ImplicitConversionSequence ICS;
    CheckReferenceInit(From, ToType, &ICS, SuppressUserConversions,
                       /*AllowExplicit=*/false, ForceRValue);
    return ICS;
  } else {
    return TryImplicitConversion(From, ToType, SuppressUserConversions,
                                 ForceRValue);
  }
}

/// PerformCopyInitialization - Copy-initialize an object of type @p ToType with
/// the expression @p From. Returns true (and emits a diagnostic) if there was
/// an error, returns false if the initialization succeeded. Elidable should
/// be true when the copy may be elided (C++ 12.8p15). Overload resolution works
/// differently in C++0x for this case.
bool Sema::PerformCopyInitialization(Expr *&From, QualType ToType, 
                                     const char* Flavor, bool Elidable) {
  if (!getLangOptions().CPlusPlus) {
    // In C, argument passing is the same as performing an assignment.
    QualType FromType = From->getType();
    
    AssignConvertType ConvTy =
      CheckSingleAssignmentConstraints(ToType, From);
    if (ConvTy != Compatible &&
        CheckTransparentUnionArgumentConstraints(ToType, From) == Compatible)
      ConvTy = Compatible;
    
    return DiagnoseAssignmentResult(ConvTy, From->getLocStart(), ToType,
                                    FromType, From, Flavor);
  }

  if (ToType->isReferenceType())
    return CheckReferenceInit(From, ToType);

  if (!PerformImplicitConversion(From, ToType, Flavor,
                                 /*AllowExplicit=*/false, Elidable))
    return false;

  return Diag(From->getSourceRange().getBegin(),
              diag::err_typecheck_convert_incompatible)
    << ToType << From->getType() << Flavor << From->getSourceRange();
}

/// TryObjectArgumentInitialization - Try to initialize the object
/// parameter of the given member function (@c Method) from the
/// expression @p From.
ImplicitConversionSequence
Sema::TryObjectArgumentInitialization(Expr *From, CXXMethodDecl *Method) {
  QualType ClassType = Context.getTypeDeclType(Method->getParent());
  unsigned MethodQuals = Method->getTypeQualifiers();
  QualType ImplicitParamType = ClassType.getQualifiedType(MethodQuals);

  // Set up the conversion sequence as a "bad" conversion, to allow us
  // to exit early.
  ImplicitConversionSequence ICS;
  ICS.Standard.setAsIdentityConversion();
  ICS.ConversionKind = ImplicitConversionSequence::BadConversion;

  // We need to have an object of class type.
  QualType FromType = From->getType();
  if (const PointerType *PT = FromType->getAs<PointerType>())
    FromType = PT->getPointeeType();

  assert(FromType->isRecordType());

  // The implicit object parmeter is has the type "reference to cv X",
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
  if (ImplicitParamType.getCVRQualifiers() != FromType.getCVRQualifiers() &&
      !ImplicitParamType.isAtLeastAsQualifiedAs(FromType))
    return ICS;

  // Check that we have either the same type or a derived type. It
  // affects the conversion rank.
  QualType ClassTypeCanon = Context.getCanonicalType(ClassType);
  if (ClassTypeCanon == FromTypeCanon.getUnqualifiedType())
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

  ImplicitConversionSequence ICS 
    = TryObjectArgumentInitialization(From, Method);
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
  return TryImplicitConversion(From, Context.BoolTy, false, true);
}

/// PerformContextuallyConvertToBool - Perform a contextual conversion
/// of the expression From to bool (C++0x [conv]p3).
bool Sema::PerformContextuallyConvertToBool(Expr *&From) {
  ImplicitConversionSequence ICS = TryContextuallyConvertToBool(From);
  if (!PerformImplicitConversion(From, Context.BoolTy, ICS, "converting"))
    return false;

  return Diag(From->getSourceRange().getBegin(), 
              diag::err_typecheck_bool_condition)
    << From->getType() << From->getSourceRange();
}

/// AddOverloadCandidate - Adds the given function to the set of
/// candidate functions, using the given function call arguments.  If
/// @p SuppressUserConversions, then don't allow user-defined
/// conversions via constructors or conversion operators.
/// If @p ForceRValue, treat all arguments as rvalues. This is a slightly
/// hacky way to implement the overloading rules for elidable copy
/// initialization in C++0x (C++0x 12.8p15).
void 
Sema::AddOverloadCandidate(FunctionDecl *Function, 
                           Expr **Args, unsigned NumArgs,
                           OverloadCandidateSet& CandidateSet,
                           bool SuppressUserConversions,
                           bool ForceRValue)
{
  const FunctionProtoType* Proto 
    = dyn_cast<FunctionProtoType>(Function->getType()->getAsFunctionType());
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
      // function, e.g., X::f(). We use a NULL object as the implied
      // object argument (C++ [over.call.func]p3).
      AddMethodCandidate(Method, 0, Args, NumArgs, CandidateSet, 
                         SuppressUserConversions, ForceRValue);
      return;
    }
    // We treat a constructor like a non-member function, since its object
    // argument doesn't participate in overload resolution.
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
  if (NumArgs > NumArgsInProto && !Proto->isVariadic()) {
    Candidate.Viable = false;
    return;
  }

  // (C++ 13.3.2p2): A candidate function having more than m parameters
  // is viable only if the (m+1)st parameter has a default argument
  // (8.3.6). For the purposes of overload resolution, the
  // parameter list is truncated on the right, so that there are
  // exactly m parameters.
  unsigned MinRequiredArgs = Function->getMinRequiredArguments();
  if (NumArgs < MinRequiredArgs) {
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
                                SuppressUserConversions, ForceRValue);
      if (Candidate.Conversions[ArgIdx].ConversionKind 
            == ImplicitConversionSequence::BadConversion) {
        Candidate.Viable = false;
        break;
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
    if (FunctionDecl *FD = dyn_cast<FunctionDecl>(*F))
      AddOverloadCandidate(FD, Args, NumArgs, CandidateSet, 
                           SuppressUserConversions);
    else
      AddTemplateOverloadCandidate(cast<FunctionTemplateDecl>(*F),
                                   /*FIXME: explicit args */false, 0, 0,
                                   Args, NumArgs, CandidateSet, 
                                   SuppressUserConversions);
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
Sema::AddMethodCandidate(CXXMethodDecl *Method, Expr *Object,
                         Expr **Args, unsigned NumArgs,
                         OverloadCandidateSet& CandidateSet,
                         bool SuppressUserConversions, bool ForceRValue)
{
  const FunctionProtoType* Proto 
    = dyn_cast<FunctionProtoType>(Method->getType()->getAsFunctionType());
  assert(Proto && "Methods without a prototype cannot be overloaded");
  assert(!isa<CXXConversionDecl>(Method) &&
         "Use AddConversionCandidate for conversion functions");
  assert(!isa<CXXConstructorDecl>(Method) &&
         "Use AddOverloadCandidate for constructors");

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

  if (Method->isStatic() || !Object)
    // The implicit object argument is ignored.
    Candidate.IgnoreObjectArgument = true;
  else {
    // Determine the implicit conversion sequence for the object
    // parameter.
    Candidate.Conversions[0] = TryObjectArgumentInitialization(Object, Method);
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
                                SuppressUserConversions, ForceRValue);
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
                                 bool HasExplicitTemplateArgs,
                                 const TemplateArgument *ExplicitTemplateArgs,
                                 unsigned NumExplicitTemplateArgs,
                                 Expr *Object, Expr **Args, unsigned NumArgs,
                                 OverloadCandidateSet& CandidateSet,
                                 bool SuppressUserConversions,
                                 bool ForceRValue) {
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
      = DeduceTemplateArguments(MethodTmpl, HasExplicitTemplateArgs,
                                ExplicitTemplateArgs, NumExplicitTemplateArgs,
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
  AddMethodCandidate(cast<CXXMethodDecl>(Specialization), Object, Args, NumArgs, 
                     CandidateSet, SuppressUserConversions, ForceRValue);
}

/// \brief Add a C++ function template as a candidate in the candidate set,
/// using template argument deduction to produce an appropriate function
/// template specialization.
void 
Sema::AddTemplateOverloadCandidate(FunctionTemplateDecl *FunctionTemplate,
                                   bool HasExplicitTemplateArgs,
                                 const TemplateArgument *ExplicitTemplateArgs,
                                   unsigned NumExplicitTemplateArgs,
                                   Expr **Args, unsigned NumArgs,
                                   OverloadCandidateSet& CandidateSet,
                                   bool SuppressUserConversions,
                                   bool ForceRValue) {
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
        = DeduceTemplateArguments(FunctionTemplate, HasExplicitTemplateArgs,
                                  ExplicitTemplateArgs, NumExplicitTemplateArgs,
                                  Args, NumArgs, Specialization, Info)) {
    // FIXME: Record what happened with template argument deduction, so
    // that we can give the user a beautiful diagnostic.
    (void)Result;
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
                             Expr *From, QualType ToType,
                             OverloadCandidateSet& CandidateSet) {
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
  Candidate.Conversions[0] = TryObjectArgumentInitialization(From, Conversion);

  if (Candidate.Conversions[0].ConversionKind 
      == ImplicitConversionSequence::BadConversion) {
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
                            SourceLocation());
  ImplicitCastExpr ConversionFn(Context.getPointerType(Conversion->getType()),
                                CastExpr::CK_Unknown,
                                &ConversionRef, false);
  
  // Note that it is safe to allocate CallExpr on the stack here because 
  // there are 0 arguments (i.e., nothing is allocated using ASTContext's
  // allocator).
  CallExpr Call(Context, &ConversionFn, 0, 0, 
                Conversion->getConversionType().getNonReferenceType(),
                SourceLocation());
  ImplicitConversionSequence ICS = TryCopyInitialization(&Call, ToType, true);
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

/// AddSurrogateCandidate - Adds a "surrogate" candidate function that
/// converts the given @c Object to a function pointer via the
/// conversion function @c Conversion, and then attempts to call it
/// with the given arguments (C++ [over.call.object]p2-4). Proto is
/// the type of function that we'll eventually be calling.
void Sema::AddSurrogateCandidate(CXXConversionDecl *Conversion,
                                 const FunctionProtoType *Proto,
                                 Expr *Object, Expr **Args, unsigned NumArgs,
                                 OverloadCandidateSet& CandidateSet) {
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
    = TryObjectArgumentInitialization(Object, Conversion);
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
                                /*SuppressUserConversions=*/false);
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
  ArgumentDependentLookup(OpName, Args, NumArgs, Functions);
  AddFunctionCandidates(Functions, Args, NumArgs, CandidateSet);
  AddMemberOperatorCandidates(Op, OpLoc, Args, NumArgs, CandidateSet, OpRange);
  AddBuiltinOperatorCandidates(Op, Args, NumArgs, CandidateSet);
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
  // FIXME: Lookup in base classes, too!
  if (const RecordType *T1Rec = T1->getAs<RecordType>()) {
    DeclContext::lookup_const_iterator Oper, OperEnd;
    for (llvm::tie(Oper, OperEnd) = T1Rec->getDecl()->lookup(OpName);
         Oper != OperEnd; ++Oper)
      AddMethodCandidate(cast<CXXMethodDecl>(*Oper), Args[0], 
                         Args+1, NumArgs - 1, CandidateSet,
                         /*SuppressUserConversions=*/false);
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
                                ArgIdx == 0 && IsAssignmentOperator);
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

  /// Context - The AST context in which we will build the type sets.
  ASTContext &Context;

  bool AddPointerWithMoreQualifiedTypeVariants(QualType Ty);
  bool AddMemberPointerWithMoreQualifiedTypeVariants(QualType Ty);

public:
  /// iterator - Iterates through the types that are part of the set.
  typedef TypeSet::iterator iterator;

  BuiltinCandidateTypeSet(ASTContext &Context) : Context(Context) { }

  void AddTypesConvertedFrom(QualType Ty, bool AllowUserConversions,
                             bool AllowExplicitConversions);

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
bool
BuiltinCandidateTypeSet::AddPointerWithMoreQualifiedTypeVariants(QualType Ty) {
  // Insert this type.
  if (!PointerTypes.insert(Ty))
    return false;

  if (const PointerType *PointerTy = Ty->getAs<PointerType>()) {
    QualType PointeeTy = PointerTy->getPointeeType();
    // FIXME: Optimize this so that we don't keep trying to add the same types.

    // FIXME: Do we have to add CVR qualifiers at *all* levels to deal with all
    // pointer conversions that don't cast away constness?
    if (!PointeeTy.isConstQualified())
      AddPointerWithMoreQualifiedTypeVariants
        (Context.getPointerType(PointeeTy.withConst()));
    if (!PointeeTy.isVolatileQualified())
      AddPointerWithMoreQualifiedTypeVariants
        (Context.getPointerType(PointeeTy.withVolatile()));
    if (!PointeeTy.isRestrictQualified())
      AddPointerWithMoreQualifiedTypeVariants
        (Context.getPointerType(PointeeTy.withRestrict()));
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
bool
BuiltinCandidateTypeSet::AddMemberPointerWithMoreQualifiedTypeVariants(
    QualType Ty) {
  // Insert this type.
  if (!MemberPointerTypes.insert(Ty))
    return false;

  if (const MemberPointerType *PointerTy = Ty->getAs<MemberPointerType>()) {
    QualType PointeeTy = PointerTy->getPointeeType();
    const Type *ClassTy = PointerTy->getClass();
    // FIXME: Optimize this so that we don't keep trying to add the same types.

    if (!PointeeTy.isConstQualified())
      AddMemberPointerWithMoreQualifiedTypeVariants
        (Context.getMemberPointerType(PointeeTy.withConst(), ClassTy));
    if (!PointeeTy.isVolatileQualified())
      AddMemberPointerWithMoreQualifiedTypeVariants
        (Context.getMemberPointerType(PointeeTy.withVolatile(), ClassTy));
    if (!PointeeTy.isRestrictQualified())
      AddMemberPointerWithMoreQualifiedTypeVariants
        (Context.getMemberPointerType(PointeeTy.withRestrict(), ClassTy));
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
                                               bool AllowUserConversions,
                                               bool AllowExplicitConversions) {
  // Only deal with canonical types.
  Ty = Context.getCanonicalType(Ty);

  // Look through reference types; they aren't part of the type of an
  // expression for the purposes of conversions.
  if (const ReferenceType *RefTy = Ty->getAs<ReferenceType>())
    Ty = RefTy->getPointeeType();

  // We don't care about qualifiers on the type.
  Ty = Ty.getUnqualifiedType();

  if (const PointerType *PointerTy = Ty->getAs<PointerType>()) {
    QualType PointeeTy = PointerTy->getPointeeType();

    // Insert our type, and its more-qualified variants, into the set
    // of types.
    if (!AddPointerWithMoreQualifiedTypeVariants(Ty))
      return;

    // Add 'cv void*' to our set of types.
    if (!Ty->isVoidType()) {
      QualType QualVoid 
        = Context.VoidTy.getQualifiedType(PointeeTy.getCVRQualifiers());
      AddPointerWithMoreQualifiedTypeVariants(Context.getPointerType(QualVoid));
    }

    // If this is a pointer to a class type, add pointers to its bases
    // (with the same level of cv-qualification as the original
    // derived class, of course).
    if (const RecordType *PointeeRec = PointeeTy->getAs<RecordType>()) {
      CXXRecordDecl *ClassDecl = cast<CXXRecordDecl>(PointeeRec->getDecl());
      for (CXXRecordDecl::base_class_iterator Base = ClassDecl->bases_begin();
           Base != ClassDecl->bases_end(); ++Base) {
        QualType BaseTy = Context.getCanonicalType(Base->getType());
        BaseTy = BaseTy.getQualifiedType(PointeeTy.getCVRQualifiers());

        // Add the pointer type, recursively, so that we get all of
        // the indirect base classes, too.
        AddTypesConvertedFrom(Context.getPointerType(BaseTy), false, false);
      }
    }
  } else if (Ty->isMemberPointerType()) {
    // Member pointers are far easier, since the pointee can't be converted.
    if (!AddMemberPointerWithMoreQualifiedTypeVariants(Ty))
      return;
  } else if (Ty->isEnumeralType()) {
    EnumerationTypes.insert(Ty);
  } else if (AllowUserConversions) {
    if (const RecordType *TyRec = Ty->getAs<RecordType>()) {
      CXXRecordDecl *ClassDecl = cast<CXXRecordDecl>(TyRec->getDecl());
      // FIXME: Visit conversion functions in the base classes, too.
      OverloadedFunctionDecl *Conversions 
        = ClassDecl->getConversionFunctions();
      for (OverloadedFunctionDecl::function_iterator Func 
             = Conversions->function_begin();
           Func != Conversions->function_end(); ++Func) {
        CXXConversionDecl *Conv = cast<CXXConversionDecl>(*Func);
        if (AllowExplicitConversions || !Conv->isExplicit())
          AddTypesConvertedFrom(Conv->getConversionType(), false, false);
      }
    }
  }
}

/// AddBuiltinOperatorCandidates - Add the appropriate built-in
/// operator overloads to the candidate set (C++ [over.built]), based
/// on the operator @p Op and the arguments given. For example, if the
/// operator is a binary '+', this routine might add "int
/// operator+(int, int)" to cover integer addition.
void
Sema::AddBuiltinOperatorCandidates(OverloadedOperatorKind Op, 
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
//    Context.Char16Ty, Context.Char32Ty, 
    Context.SignedCharTy, Context.ShortTy,
    Context.UnsignedCharTy, Context.UnsignedShortTy,
    Context.IntTy, Context.LongTy, Context.LongLongTy,
    Context.UnsignedIntTy, Context.UnsignedLongTy, Context.UnsignedLongLongTy,
    Context.FloatTy, Context.DoubleTy, Context.LongDoubleTy
  };

  // Find all of the types that the arguments can convert to, but only
  // if the operator we're looking at has built-in operator candidates
  // that make use of these types.
  BuiltinCandidateTypeSet CandidateTypes(Context);
  if (Op == OO_Less || Op == OO_Greater || Op == OO_LessEqual ||
      Op == OO_GreaterEqual || Op == OO_EqualEqual || Op == OO_ExclaimEqual ||
      Op == OO_Plus || (Op == OO_Minus && NumArgs == 2) || Op == OO_Equal ||
      Op == OO_PlusEqual || Op == OO_MinusEqual || Op == OO_Subscript ||
      Op == OO_ArrowStar || Op == OO_PlusPlus || Op == OO_MinusMinus ||
      (Op == OO_Star && NumArgs == 1) || Op == OO_Conditional) {
    for (unsigned ArgIdx = 0; ArgIdx < NumArgs; ++ArgIdx)
      CandidateTypes.AddTypesConvertedFrom(Args[ArgIdx]->getType(),
                                           true,
                                           (Op == OO_Exclaim ||
                                            Op == OO_AmpAmp ||
                                            Op == OO_PipePipe));
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

      // Volatile version
      ParamTypes[0] = Context.getLValueReferenceType(ArithTy.withVolatile());
      if (NumArgs == 1)
        AddBuiltinCandidate(ParamTypes[0], ParamTypes, Args, 1, CandidateSet);
      else
        AddBuiltinCandidate(ArithTy, ParamTypes, Args, 2, CandidateSet);
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

      if (!Context.getCanonicalType(*Ptr).isVolatileQualified()) {
        // With volatile
        ParamTypes[0] = Context.getLValueReferenceType((*Ptr).withVolatile());
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

  case OO_Less:
  case OO_Greater:
  case OO_LessEqual:
  case OO_GreaterEqual:
  case OO_EqualEqual:
  case OO_ExclaimEqual:
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
    //   (FIXME:) pointer to member type and VQ is either volatile or
    //   empty, there exist candidate operator functions of the form
    //
    //        VQ T&      operator=(VQ T&, T);
    for (BuiltinCandidateTypeSet::iterator Enum 
           = CandidateTypes.enumeration_begin();
         Enum != CandidateTypes.enumeration_end(); ++Enum) {
      QualType ParamTypes[2];

      // T& operator=(T&, T)
      ParamTypes[0] = Context.getLValueReferenceType(*Enum);
      ParamTypes[1] = *Enum;
      AddBuiltinCandidate(ParamTypes[0], ParamTypes, Args, 2, CandidateSet,
                          /*IsAssignmentOperator=*/false);

      if (!Context.getCanonicalType(*Enum).isVolatileQualified()) {
        // volatile T& operator=(volatile T&, T)
        ParamTypes[0] = Context.getLValueReferenceType((*Enum).withVolatile());
        ParamTypes[1] = *Enum;
        AddBuiltinCandidate(ParamTypes[0], ParamTypes, Args, 2, CandidateSet,
                            /*IsAssignmentOperator=*/false);
      }
    }
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

      if (!Context.getCanonicalType(*Ptr).isVolatileQualified()) {
        // volatile version
        ParamTypes[0] = Context.getLValueReferenceType((*Ptr).withVolatile());
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
        ParamTypes[0] = ArithmeticTypes[Left].withVolatile();
        ParamTypes[0] = Context.getLValueReferenceType(ParamTypes[0]);
        AddBuiltinCandidate(ParamTypes[0], ParamTypes, Args, 2, CandidateSet,
                            /*IsAssigmentOperator=*/Op == OO_Equal);
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

        // Add this built-in operator as a candidate (VQ is 'volatile').
        ParamTypes[0] = ArithmeticTypes[Left];
        ParamTypes[0].addVolatile();
        ParamTypes[0] = Context.getLValueReferenceType(ParamTypes[0]);
        AddBuiltinCandidate(ParamTypes[0], ParamTypes, Args, 2, CandidateSet);
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
    // FIXME: No support for pointer-to-members yet.
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
                                           OverloadCandidateSet& CandidateSet) {
  FunctionSet Functions;

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

  ArgumentDependentLookup(Name, Args, NumArgs, Functions);

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
    if (FunctionDecl *FD = dyn_cast<FunctionDecl>(*Func))
      AddOverloadCandidate(FD, Args, NumArgs, CandidateSet);
    else
      AddTemplateOverloadCandidate(cast<FunctionTemplateDecl>(*Func),  
                                   /*FIXME: explicit args */false, 0, 0,
                                   Args, NumArgs, CandidateSet);
  }
}

/// isBetterOverloadCandidate - Determines whether the first overload
/// candidate is a better candidate than the second (C++ 13.3.3p1).
bool 
Sema::isBetterOverloadCandidate(const OverloadCandidate& Cand1,
                                const OverloadCandidate& Cand2)
{
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
    // FIXME: Implement partial ordering of function templates.
    Diag(SourceLocation(), diag::unsup_function_template_partial_ordering);

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
Sema::OverloadingResult 
Sema::BestViableFunction(OverloadCandidateSet& CandidateSet,
                         SourceLocation Loc,
                         OverloadCandidateSet::iterator& Best)
{
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
                              bool OnlyViable)
{
  OverloadCandidateSet::iterator Cand = CandidateSet.begin(),
                             LastCand = CandidateSet.end();
  for (; Cand != LastCand; ++Cand) {
    if (Cand->Viable || !OnlyViable) {
      if (Cand->Function) {
        if (Cand->Function->isDeleted() ||
            Cand->Function->getAttr<UnavailableAttr>()) {
          // Deleted or "unavailable" function.
          Diag(Cand->Function->getLocation(), diag::err_ovl_candidate_deleted)
            << Cand->Function->isDeleted();
        } else {
          // Normal function
          // FIXME: Give a better reason!
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
        FnType = QualType(FnType->getAsFunctionType(), 0);
        // Reconstruct the pointer/reference as appropriate.
        if (isPointer) FnType = Context.getPointerType(FnType);
        if (isRValueReference) FnType = Context.getRValueReferenceType(FnType);
        if (isLValueReference) FnType = Context.getLValueReferenceType(FnType);

        Diag(Cand->Surrogate->getLocation(), diag::err_ovl_surrogate_cand)
          << FnType;
      } else {
        // FIXME: We need to get the identifier in here
        // FIXME: Do we want the error message to point at the operator?
        // (built-ins won't have a location)
        QualType FnType 
          = Context.getFunctionType(Cand->BuiltinTypes.ResultTy,
                                    Cand->BuiltinTypes.ParamTypes,
                                    Cand->Conversions.size(),
                                    false, 0);

        Diag(SourceLocation(), diag::err_ovl_builtin_candidate) << FnType;
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
  OverloadedFunctionDecl *Ovl = 0;
  
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

  // Try to dig out the overloaded function.
  FunctionTemplateDecl *FunctionTemplate = 0;
  if (DeclRefExpr *DR = dyn_cast<DeclRefExpr>(OvlExpr)) {
    Ovl = dyn_cast<OverloadedFunctionDecl>(DR->getDecl());
    FunctionTemplate = dyn_cast<FunctionTemplateDecl>(DR->getDecl());
  }

  // If there's no overloaded function declaration or function template, 
  // we're done.
  if (!Ovl && !FunctionTemplate)
    return 0;
 
  OverloadIterator Fun;
  if (Ovl)
    Fun = Ovl;
  else
    Fun = FunctionTemplate;
  
  // Look through all of the overloaded functions, searching for one
  // whose type matches exactly.
  llvm::SmallPtrSet<FunctionDecl *, 4> Matches;
  
  bool FoundNonTemplateFunction = false;
  for (OverloadIterator FunEnd; Fun != FunEnd; ++Fun) {
    // C++ [over.over]p3:
    //   Non-member functions and static member functions match
    //   targets of type "pointer-to-function" or "reference-to-function."
    //   Nonstatic member functions match targets of
    //   type "pointer-to-member-function."
    // Note that according to DR 247, the containing class does not matter.

    if (FunctionTemplateDecl *FunctionTemplate 
          = dyn_cast<FunctionTemplateDecl>(*Fun)) {
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
      FunctionDecl *Specialization = 0;
      TemplateDeductionInfo Info(Context);
      if (TemplateDeductionResult Result
            = DeduceTemplateArguments(FunctionTemplate, /*FIXME*/false,
                                      /*FIXME:*/0, /*FIXME:*/0,
                                      FunctionType, Specialization, Info)) {
        // FIXME: make a note of the failed deduction for diagnostics.
        (void)Result;
      } else {
        assert(FunctionType 
                 == Context.getCanonicalType(Specialization->getType()));
        Matches.insert(
                cast<FunctionDecl>(Specialization->getCanonicalDecl()));
      }
    }
    
    if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(*Fun)) {
      // Skip non-static functions when converting to pointer, and static
      // when converting to member pointer.
      if (Method->isStatic() == IsMember)
        continue;
    } else if (IsMember)
      continue;

    if (FunctionDecl *FunDecl = dyn_cast<FunctionDecl>(*Fun)) {
      if (FunctionType == Context.getCanonicalType(FunDecl->getType())) {
        Matches.insert(cast<FunctionDecl>(Fun->getCanonicalDecl()));
        FoundNonTemplateFunction = true;
      }
    } 
  }

  // If there were 0 or 1 matches, we're done.
  if (Matches.empty())
    return 0;
  else if (Matches.size() == 1)
    return *Matches.begin();

  // C++ [over.over]p4:
  //   If more than one function is selected, [...]
  llvm::SmallVector<FunctionDecl *, 4> RemainingMatches;
  if (FoundNonTemplateFunction) {
    // [...] any function template specializations in the set are eliminated 
    // if the set also contains a non-template function, [...]
    for (llvm::SmallPtrSet<FunctionDecl *, 4>::iterator M = Matches.begin(),
                                                     MEnd = Matches.end();
         M != MEnd; ++M)
      if ((*M)->getPrimaryTemplate() == 0)
        RemainingMatches.push_back(*M);
  } else {
    // [...] and any given function template specialization F1 is eliminated 
    // if the set contains a second function template specialization whose 
    // function template is more specialized than the function template of F1 
    // according to the partial ordering rules of 14.5.5.2.
    // FIXME: Implement this!
    RemainingMatches.append(Matches.begin(), Matches.end());
  }
  
  // [...] After such eliminations, if any, there shall remain exactly one 
  // selected function.
  if (RemainingMatches.size() == 1)
    return RemainingMatches.front();
  
  // FIXME: We should probably return the same thing that BestViableFunction
  // returns (even if we issue the diagnostics here).
  Diag(From->getLocStart(), diag::err_addr_ovl_ambiguous)
    << RemainingMatches[0]->getDeclName();
  for (unsigned I = 0, N = RemainingMatches.size(); I != N; ++I)
    Diag(RemainingMatches[I]->getLocation(), diag::err_ovl_candidate);
  return 0;
}

/// ResolveOverloadedCallFn - Given the call expression that calls Fn
/// (which eventually refers to the declaration Func) and the call
/// arguments Args/NumArgs, attempt to resolve the function call down
/// to a specific function. If overload resolution succeeds, returns
/// the function declaration produced by overload
/// resolution. Otherwise, emits diagnostics, deletes all of the
/// arguments and Fn, and returns NULL.
FunctionDecl *Sema::ResolveOverloadedCallFn(Expr *Fn, NamedDecl *Callee,
                                            DeclarationName UnqualifiedName,
                                            bool HasExplicitTemplateArgs,
                                 const TemplateArgument *ExplicitTemplateArgs,
                                            unsigned NumExplicitTemplateArgs,
                                            SourceLocation LParenLoc,
                                            Expr **Args, unsigned NumArgs,
                                            SourceLocation *CommaLocs, 
                                            SourceLocation RParenLoc,
                                            bool &ArgumentDependentLookup) {
  OverloadCandidateSet CandidateSet;

  // Add the functions denoted by Callee to the set of candidate
  // functions. While we're doing so, track whether argument-dependent
  // lookup still applies, per:
  //
  // C++0x [basic.lookup.argdep]p3:
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
  if (OverloadedFunctionDecl *Ovl 
        = dyn_cast_or_null<OverloadedFunctionDecl>(Callee)) {
    for (OverloadedFunctionDecl::function_iterator Func = Ovl->function_begin(),
                                                FuncEnd = Ovl->function_end();
         Func != FuncEnd; ++Func) {
      DeclContext *Ctx = 0;
      if (FunctionDecl *FunDecl = dyn_cast<FunctionDecl>(*Func)) {
        if (HasExplicitTemplateArgs)
          continue;
        
        AddOverloadCandidate(FunDecl, Args, NumArgs, CandidateSet);
        Ctx = FunDecl->getDeclContext();
      } else {
        FunctionTemplateDecl *FunTmpl = cast<FunctionTemplateDecl>(*Func);
        AddTemplateOverloadCandidate(FunTmpl, HasExplicitTemplateArgs,
                                     ExplicitTemplateArgs,
                                     NumExplicitTemplateArgs,
                                     Args, NumArgs, CandidateSet);
        Ctx = FunTmpl->getDeclContext();
      }


      if (Ctx->isRecord() || Ctx->isFunctionOrMethod())
        ArgumentDependentLookup = false;
    }
  } else if (FunctionDecl *Func = dyn_cast_or_null<FunctionDecl>(Callee)) {
    assert(!HasExplicitTemplateArgs && "Explicit template arguments?");
    AddOverloadCandidate(Func, Args, NumArgs, CandidateSet);

    if (Func->getDeclContext()->isRecord() ||
        Func->getDeclContext()->isFunctionOrMethod())
      ArgumentDependentLookup = false;
  } else if (FunctionTemplateDecl *FuncTemplate 
               = dyn_cast_or_null<FunctionTemplateDecl>(Callee)) {
    AddTemplateOverloadCandidate(FuncTemplate, HasExplicitTemplateArgs,
                                 ExplicitTemplateArgs,
                                 NumExplicitTemplateArgs, 
                                 Args, NumArgs, CandidateSet);

    if (FuncTemplate->getDeclContext()->isRecord())
      ArgumentDependentLookup = false;
  }

  if (Callee)
    UnqualifiedName = Callee->getDeclName();

  // FIXME: Pass explicit template arguments through for ADL
  if (ArgumentDependentLookup)
    AddArgumentDependentLookupCandidates(UnqualifiedName, Args, NumArgs,
                                         CandidateSet);

  OverloadCandidateSet::iterator Best;
  switch (BestViableFunction(CandidateSet, Fn->getLocStart(), Best)) {
  case OR_Success:
    return Best->Function;

  case OR_No_Viable_Function:
    Diag(Fn->getSourceRange().getBegin(),
         diag::err_ovl_no_viable_function_in_call)
      << UnqualifiedName << Fn->getSourceRange();
    PrintOverloadCandidates(CandidateSet, /*OnlyViable=*/false);
    break;

  case OR_Ambiguous:
    Diag(Fn->getSourceRange().getBegin(), diag::err_ovl_ambiguous_call)
      << UnqualifiedName << Fn->getSourceRange();
    PrintOverloadCandidates(CandidateSet, /*OnlyViable=*/true);
    break;

  case OR_Deleted:
    Diag(Fn->getSourceRange().getBegin(), diag::err_ovl_deleted_call)
      << Best->Function->isDeleted()
      << UnqualifiedName
      << Fn->getSourceRange();
    PrintOverloadCandidates(CandidateSet, /*OnlyViable=*/true);
    break;
  }

  // Overload resolution failed. Destroy all of the subexpressions and
  // return NULL.
  Fn->Destroy(Context);
  for (unsigned Arg = 0; Arg < NumArgs; ++Arg)
    Args[Arg]->Destroy(Context);
  return 0;
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
    OverloadedFunctionDecl *Overloads 
      = OverloadedFunctionDecl::Create(Context, CurContext, OpName);
    for (FunctionSet::iterator Func = Functions.begin(), 
                            FuncEnd = Functions.end();
         Func != FuncEnd; ++Func)
      Overloads->addOverload(*Func);

    DeclRefExpr *Fn = new (Context) DeclRefExpr(Overloads, Context.OverloadTy,
                                                OpLoc, false, false);
    
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
  AddBuiltinOperatorCandidates(Op, &Args[0], NumArgs, CandidateSet);

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
        if (PerformCopyInitialization(Input,
                                      FnDecl->getParamDecl(0)->getType(),
                                      "passing"))
          return ExprError();
      }

      // Determine the result type
      QualType ResultTy
        = FnDecl->getType()->getAsFunctionType()->getResultType();
      ResultTy = ResultTy.getNonReferenceType();
      
      // Build the actual expression node.
      Expr *FnExpr = new (Context) DeclRefExpr(FnDecl, FnDecl->getType(),
                                               SourceLocation());
      UsualUnaryConversions(FnExpr);
      
      input.release();
      
      Expr *CE = new (Context) CXXOperatorCallExpr(Context, Op, FnExpr, 
                                                   &Input, 1, ResultTy, OpLoc);
      return MaybeBindToTemporary(CE);
    } else {
      // We matched a built-in operator. Convert the arguments, then
      // break out so that we will build the appropriate built-in
      // operator node.
        if (PerformImplicitConversion(Input, Best->BuiltinTypes.ParamTypes[0],
                                      Best->Conversions[0], "passing"))
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
      PrintOverloadCandidates(CandidateSet, /*OnlyViable=*/true);
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

  BinaryOperator::Opcode Opc = static_cast<BinaryOperator::Opcode>(OpcIn);
  OverloadedOperatorKind Op = BinaryOperator::getOverloadedOperator(Opc);
  DeclarationName OpName = Context.DeclarationNames.getCXXOperatorName(Op);

  // If either side is type-dependent, create an appropriate dependent
  // expression.
  if (LHS->isTypeDependent() || RHS->isTypeDependent()) {
    // .* cannot be overloaded.
    if (Opc == BinaryOperator::PtrMemD)
      return Owned(new (Context) BinaryOperator(LHS, RHS, Opc,
                                                Context.DependentTy, OpLoc));

    OverloadedFunctionDecl *Overloads 
      = OverloadedFunctionDecl::Create(Context, CurContext, OpName);
    for (FunctionSet::iterator Func = Functions.begin(), 
                            FuncEnd = Functions.end();
         Func != FuncEnd; ++Func)
      Overloads->addOverload(*Func);

    DeclRefExpr *Fn = new (Context) DeclRefExpr(Overloads, Context.OverloadTy,
                                                OpLoc, false, false);
    
    return Owned(new (Context) CXXOperatorCallExpr(Context, Op, Fn,
                                                   Args, 2, 
                                                   Context.DependentTy,
                                                   OpLoc));
  }

  // If this is the .* operator, which is not overloadable, just
  // create a built-in binary operator.
  if (Opc == BinaryOperator::PtrMemD)
    return CreateBuiltinBinOp(OpLoc, Opc, LHS, RHS);

  // If this is one of the assignment operators, we only perform
  // overload resolution if the left-hand side is a class or
  // enumeration type (C++ [expr.ass]p3).
  if (Opc >= BinaryOperator::Assign && Opc <= BinaryOperator::OrAssign &&
      !LHS->getType()->isOverloadableType())
    return CreateBuiltinBinOp(OpLoc, Opc, LHS, RHS);

  // Build an empty overload set.
  OverloadCandidateSet CandidateSet;

  // Add the candidates from the given function set.
  AddFunctionCandidates(Functions, Args, 2, CandidateSet, false);

  // Add operator candidates that are member functions.
  AddMemberOperatorCandidates(Op, OpLoc, Args, 2, CandidateSet);

  // Add builtin operator candidates.
  AddBuiltinOperatorCandidates(Op, Args, 2, CandidateSet);

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
          if (PerformObjectArgumentInitialization(LHS, Method) ||
              PerformCopyInitialization(RHS, FnDecl->getParamDecl(0)->getType(),
                                        "passing"))
            return ExprError();
        } else {
          // Convert the arguments.
          if (PerformCopyInitialization(LHS, FnDecl->getParamDecl(0)->getType(),
                                        "passing") ||
              PerformCopyInitialization(RHS, FnDecl->getParamDecl(1)->getType(),
                                        "passing"))
            return ExprError();
        }

        // Determine the result type
        QualType ResultTy
          = FnDecl->getType()->getAsFunctionType()->getResultType();
        ResultTy = ResultTy.getNonReferenceType();

        // Build the actual expression node.
        Expr *FnExpr = new (Context) DeclRefExpr(FnDecl, FnDecl->getType(),
                                                 OpLoc);
        UsualUnaryConversions(FnExpr);

        Expr *CE = new (Context) CXXOperatorCallExpr(Context, Op, FnExpr, 
                                                     Args, 2, ResultTy, OpLoc);
        return MaybeBindToTemporary(CE);
      } else {
        // We matched a built-in operator. Convert the arguments, then
        // break out so that we will build the appropriate built-in
        // operator node.
        if (PerformImplicitConversion(LHS, Best->BuiltinTypes.ParamTypes[0],
                                      Best->Conversions[0], "passing") ||
            PerformImplicitConversion(RHS, Best->BuiltinTypes.ParamTypes[1],
                                      Best->Conversions[1], "passing"))
          return ExprError();

        break;
      }
    }

    case OR_No_Viable_Function:
      // For class as left operand for assignment or compound assigment operator
      // do not fall through to handling in built-in, but report that no overloaded
      // assignment operator found
      if (LHS->getType()->isRecordType() && Opc >= BinaryOperator::Assign && Opc <= BinaryOperator::OrAssign) {
        Diag(OpLoc,  diag::err_ovl_no_viable_oper)
             << BinaryOperator::getOpcodeStr(Opc)
             << LHS->getSourceRange() << RHS->getSourceRange();
        return ExprError();
      }
      // No viable function; fall through to handling this as a
      // built-in operator, which will produce an error message for us.
      break;

    case OR_Ambiguous:
      Diag(OpLoc,  diag::err_ovl_ambiguous_oper)
          << BinaryOperator::getOpcodeStr(Opc)
          << LHS->getSourceRange() << RHS->getSourceRange();
      PrintOverloadCandidates(CandidateSet, /*OnlyViable=*/true);
      return ExprError();

    case OR_Deleted:
      Diag(OpLoc, diag::err_ovl_deleted_oper)
        << Best->Function->isDeleted()
        << BinaryOperator::getOpcodeStr(Opc)
        << LHS->getSourceRange() << RHS->getSourceRange();
      PrintOverloadCandidates(CandidateSet, /*OnlyViable=*/true);
      return ExprError();
    }

  // Either we found no viable overloaded operator or we matched a
  // built-in operator. In either case, try to build a built-in
  // operation.
  return CreateBuiltinBinOp(OpLoc, Opc, LHS, RHS);
}

/// BuildCallToMemberFunction - Build a call to a member
/// function. MemExpr is the expression that refers to the member
/// function (and includes the object parameter), Args/NumArgs are the
/// arguments to the function call (not including the object
/// parameter). The caller needs to validate that the member
/// expression refers to a member function or an overloaded member
/// function.
Sema::ExprResult
Sema::BuildCallToMemberFunction(Scope *S, Expr *MemExprE, 
                                SourceLocation LParenLoc, Expr **Args, 
                                unsigned NumArgs, SourceLocation *CommaLocs,
                                SourceLocation RParenLoc) {
  // Dig out the member expression. This holds both the object
  // argument and the member function we're referring to.
  MemberExpr *MemExpr = 0;
  if (ParenExpr *ParenE = dyn_cast<ParenExpr>(MemExprE))
    MemExpr = dyn_cast<MemberExpr>(ParenE->getSubExpr());
  else
    MemExpr = dyn_cast<MemberExpr>(MemExprE);
  assert(MemExpr && "Building member call without member expression");

  // Extract the object argument.
  Expr *ObjectArg = MemExpr->getBase();

  CXXMethodDecl *Method = 0;
  if (isa<OverloadedFunctionDecl>(MemExpr->getMemberDecl()) ||
      isa<FunctionTemplateDecl>(MemExpr->getMemberDecl())) {
    // Add overload candidates
    OverloadCandidateSet CandidateSet;
    DeclarationName DeclName = MemExpr->getMemberDecl()->getDeclName();
    
    if (OverloadedFunctionDecl *Ovl 
          = dyn_cast<OverloadedFunctionDecl>(MemExpr->getMemberDecl())) {
      for (OverloadedFunctionDecl::function_iterator 
                Func = Ovl->function_begin(),
             FuncEnd = Ovl->function_end();
           Func != FuncEnd; ++Func) {
        if (Method = dyn_cast<CXXMethodDecl>(*Func))
          AddMethodCandidate(Method, ObjectArg, Args, NumArgs, CandidateSet, 
                             /*SuppressUserConversions=*/false);
        else
          AddMethodTemplateCandidate(cast<FunctionTemplateDecl>(*Func), 
                                     /*FIXME:*/false, /*FIXME:*/0, 
                                     /*FIXME:*/0, ObjectArg, Args, NumArgs,
                                     CandidateSet,
                                     /*SuppressUsedConversions=*/false);
      }
    } else
      AddMethodTemplateCandidate(
                          cast<FunctionTemplateDecl>(MemExpr->getMemberDecl()), 
                                 /*FIXME:*/false, /*FIXME:*/0, 
                                 /*FIXME:*/0, ObjectArg, Args, NumArgs,
                                 CandidateSet,
                                 /*SuppressUsedConversions=*/false);
      
    OverloadCandidateSet::iterator Best;
    switch (BestViableFunction(CandidateSet, MemExpr->getLocStart(), Best)) {
    case OR_Success:
      Method = cast<CXXMethodDecl>(Best->Function);
      break;

    case OR_No_Viable_Function:
      Diag(MemExpr->getSourceRange().getBegin(), 
           diag::err_ovl_no_viable_member_function_in_call)
        << DeclName << MemExprE->getSourceRange();
      PrintOverloadCandidates(CandidateSet, /*OnlyViable=*/false);
      // FIXME: Leaking incoming expressions!
      return true;

    case OR_Ambiguous:
      Diag(MemExpr->getSourceRange().getBegin(), 
           diag::err_ovl_ambiguous_member_call)
        << DeclName << MemExprE->getSourceRange();
      PrintOverloadCandidates(CandidateSet, /*OnlyViable=*/false);
      // FIXME: Leaking incoming expressions!
      return true;

    case OR_Deleted:
      Diag(MemExpr->getSourceRange().getBegin(), 
           diag::err_ovl_deleted_member_call)
        << Best->Function->isDeleted()
        << DeclName << MemExprE->getSourceRange();
      PrintOverloadCandidates(CandidateSet, /*OnlyViable=*/false);
      // FIXME: Leaking incoming expressions!
      return true;
    }

    FixOverloadedFunctionReference(MemExpr, Method);
  } else {
    Method = dyn_cast<CXXMethodDecl>(MemExpr->getMemberDecl());
  }

  assert(Method && "Member call to something that isn't a method?");
  ExprOwningPtr<CXXMemberCallExpr> 
    TheCall(this, new (Context) CXXMemberCallExpr(Context, MemExpr, Args,
                                                  NumArgs, 
                                  Method->getResultType().getNonReferenceType(),
                                  RParenLoc));

  // Convert the object argument (for a non-static member function call).
  if (!Method->isStatic() && 
      PerformObjectArgumentInitialization(ObjectArg, Method))
    return true;
  MemExpr->setBase(ObjectArg);

  // Convert the rest of the arguments
  const FunctionProtoType *Proto = cast<FunctionProtoType>(Method->getType());
  if (ConvertArgumentsForCall(&*TheCall, MemExpr, Method, Proto, Args, NumArgs, 
                              RParenLoc))
    return true;

  if (CheckFunctionCall(Method, TheCall.get()))
    return true;

  return MaybeBindToTemporary(TheCall.release()).release();
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
  DeclContext::lookup_const_iterator Oper, OperEnd;
  for (llvm::tie(Oper, OperEnd) = Record->getDecl()->lookup(OpName);
       Oper != OperEnd; ++Oper)
    AddMethodCandidate(cast<CXXMethodDecl>(*Oper), Object, Args, NumArgs, 
                       CandidateSet, /*SuppressUserConversions=*/false);

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
  //
  // FIXME: Look in base classes for more conversion operators!
  OverloadedFunctionDecl *Conversions 
    = cast<CXXRecordDecl>(Record->getDecl())->getConversionFunctions();
  for (OverloadedFunctionDecl::function_iterator 
         Func = Conversions->function_begin(),
         FuncEnd = Conversions->function_end();
       Func != FuncEnd; ++Func) {
    CXXConversionDecl *Conv = cast<CXXConversionDecl>(*Func);

    // Strip the reference type (if any) and then the pointer type (if
    // any) to get down to what might be a function type.
    QualType ConvType = Conv->getConversionType().getNonReferenceType();
    if (const PointerType *ConvPtrType = ConvType->getAs<PointerType>())
      ConvType = ConvPtrType->getPointeeType();

    if (const FunctionProtoType *Proto = ConvType->getAsFunctionProtoType())
      AddSurrogateCandidate(Conv, Proto, Object, Args, NumArgs, CandidateSet);
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
    // FIXME: Represent the user-defined conversion in the AST!
    ImpCastExprToType(Object,
                      Conv->getConversionType().getNonReferenceType(),
                      CastExpr::CK_Unknown,
                      Conv->getConversionType()->isLValueReferenceType());
    return ActOnCallExpr(S, ExprArg(*this, Object), LParenLoc,
                         MultiExprArg(*this, (ExprTy**)Args, NumArgs),
                         CommaLocs, RParenLoc).release();
  }

  // We found an overloaded operator(). Build a CXXOperatorCallExpr
  // that calls this method, using Object for the implicit object
  // parameter and passing along the remaining arguments.
  CXXMethodDecl *Method = cast<CXXMethodDecl>(Best->Function);
  const FunctionProtoType *Proto = Method->getType()->getAsFunctionProtoType();

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
      IsError |= PerformCopyInitialization(Arg, ProtoArgType, "passing");
    } else {
      Arg = CXXDefaultArgExpr::Create(Context, Method->getParamDecl(i));
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
  // FIXME: look in base classes.
  DeclarationName OpName = Context.DeclarationNames.getCXXOperatorName(OO_Arrow);
  OverloadCandidateSet CandidateSet;
  const RecordType *BaseRecord = Base->getType()->getAs<RecordType>();

  DeclContext::lookup_const_iterator Oper, OperEnd;
  for (llvm::tie(Oper, OperEnd) 
         = BaseRecord->getDecl()->lookup(OpName); Oper != OperEnd; ++Oper)
    AddMethodCandidate(cast<CXXMethodDecl>(*Oper), Base, 0, 0, CandidateSet,
                       /*SuppressUserConversions=*/false);

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
      << "operator->" << Base->getSourceRange();
    PrintOverloadCandidates(CandidateSet, /*OnlyViable=*/true);
    return ExprError();

  case OR_Deleted:
    Diag(OpLoc,  diag::err_ovl_deleted_oper)
      << Best->Function->isDeleted()
      << "operator->" << Base->getSourceRange();
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
  Base = new (Context) CXXOperatorCallExpr(Context, OO_Arrow, FnExpr, &Base, 1, 
                                 Method->getResultType().getNonReferenceType(),
                                 OpLoc);
  return Owned(Base);
}

/// FixOverloadedFunctionReference - E is an expression that refers to
/// a C++ overloaded function (possibly with some parentheses and
/// perhaps a '&' around it). We have resolved the overloaded function
/// to the function declaration Fn, so patch up the expression E to
/// refer (possibly indirectly) to Fn.
void Sema::FixOverloadedFunctionReference(Expr *E, FunctionDecl *Fn) {
  if (ParenExpr *PE = dyn_cast<ParenExpr>(E)) {
    FixOverloadedFunctionReference(PE->getSubExpr(), Fn);
    E->setType(PE->getSubExpr()->getType());
  } else if (UnaryOperator *UnOp = dyn_cast<UnaryOperator>(E)) {
    assert(UnOp->getOpcode() == UnaryOperator::AddrOf && 
           "Can only take the address of an overloaded function");
    if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(Fn)) {
      if (Method->isStatic()) {
        // Do nothing: static member functions aren't any different
        // from non-member functions.
      } else if (QualifiedDeclRefExpr *DRE 
                 = dyn_cast<QualifiedDeclRefExpr>(UnOp->getSubExpr())) {
        // We have taken the address of a pointer to member
        // function. Perform the computation here so that we get the
        // appropriate pointer to member type.
        DRE->setDecl(Fn);
        DRE->setType(Fn->getType());
        QualType ClassType
          = Context.getTypeDeclType(cast<RecordDecl>(Method->getDeclContext()));
        E->setType(Context.getMemberPointerType(Fn->getType(), 
                                                ClassType.getTypePtr()));
        return;
      }
    }
    FixOverloadedFunctionReference(UnOp->getSubExpr(), Fn);
    E->setType(Context.getPointerType(UnOp->getSubExpr()->getType()));
  } else if (DeclRefExpr *DR = dyn_cast<DeclRefExpr>(E)) {
    assert((isa<OverloadedFunctionDecl>(DR->getDecl()) ||
            isa<FunctionTemplateDecl>(DR->getDecl())) && 
           "Expected overloaded function or function template");
    DR->setDecl(Fn);
    E->setType(Fn->getType());
  } else if (MemberExpr *MemExpr = dyn_cast<MemberExpr>(E)) {
    MemExpr->setMemberDecl(Fn);
    E->setType(Fn->getType());
  } else {
    assert(false && "Invalid reference to overloaded function");
  }
}

} // end namespace clang
