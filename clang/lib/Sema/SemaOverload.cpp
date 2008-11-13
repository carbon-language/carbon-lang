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
#include "clang/AST/TypeOrdering.h"
#include "llvm/ADT/SmallPtrSet.h"
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
    "Integral conversion",
    "Floating conversion",
    "Floating-integral conversion",
    "Pointer conversion",
    "Pointer-to-member conversion",
    "Boolean conversion",
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
      (FromType->isPointerType() ||
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
    if (const PointerType* ToPtrType = ToType->getAsPointerType())
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
  fprintf(stderr, "'%s'", ConversionFunction->getName());
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
  } else if (FunctionDecl* Old = dyn_cast<FunctionDecl>(OldD)) {
    // Is the function New an overload of the function Old?
    QualType OldQType = Context.getCanonicalType(Old->getType());
    QualType NewQType = Context.getCanonicalType(New->getType());

    // Compare the signatures (C++ 1.3.10) of the two functions to
    // determine whether they are overloads. If we find any mismatch
    // in the signature, they are overloads.

    // If either of these functions is a K&R-style function (no
    // prototype), then we consider them to have matching signatures.
    if (isa<FunctionTypeNoProto>(OldQType.getTypePtr()) ||
        isa<FunctionTypeNoProto>(NewQType.getTypePtr()))
      return false;

    FunctionTypeProto* OldType = cast<FunctionTypeProto>(OldQType.getTypePtr());
    FunctionTypeProto* NewType = cast<FunctionTypeProto>(NewQType.getTypePtr());

    // The signature of a function includes the types of its
    // parameters (C++ 1.3.10), which includes the presence or absence
    // of the ellipsis; see C++ DR 357).
    if (OldQType != NewQType &&
        (OldType->getNumArgs() != NewType->getNumArgs() ||
         OldType->isVariadic() != NewType->isVariadic() ||
         !std::equal(OldType->arg_type_begin(), OldType->arg_type_end(),
                     NewType->arg_type_begin())))
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
        OldQType.getCVRQualifiers() != NewQType.getCVRQualifiers())
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
ImplicitConversionSequence
Sema::TryImplicitConversion(Expr* From, QualType ToType,
                            bool SuppressUserConversions)
{
  ImplicitConversionSequence ICS;
  if (IsStandardConversion(From, ToType, ICS.Standard))
    ICS.ConversionKind = ImplicitConversionSequence::StandardConversion;
  else if (!SuppressUserConversions &&
           IsUserDefinedConversion(From, ToType, ICS.UserDefined)) {
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
      if (Constructor->isCopyConstructor(Context)) {
        // Turn this into a "standard" conversion sequence, so that it
        // gets ranked with standard conversion sequences.
        ICS.ConversionKind = ImplicitConversionSequence::StandardConversion;
        ICS.Standard.setAsIdentityConversion();
        ICS.Standard.FromTypePtr = From->getType().getAsOpaquePtr();
        ICS.Standard.ToTypePtr = ToType.getAsOpaquePtr();
        ICS.Standard.CopyConstructor = Constructor;
        if (IsDerivedFrom(From->getType().getUnqualifiedType(),
                          ToType.getUnqualifiedType()))
          ICS.Standard.Second = ICK_Derived_To_Base;
      }
    }
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

  // There are no standard conversions for class types, so abort early.
  if (FromType->isRecordType() || ToType->isRecordType())
    return false;

  // Standard conversions (C++ [conv])
  SCS.setAsIdentityConversion();
  SCS.Deprecated = false;
  SCS.FromTypePtr = FromType.getAsOpaquePtr();
  SCS.CopyConstructor = 0;

  // The first conversion can be an lvalue-to-rvalue conversion,
  // array-to-pointer conversion, or function-to-pointer conversion
  // (C++ 4p1).

  // Lvalue-to-rvalue conversion (C++ 4.1): 
  //   An lvalue (3.10) of a non-function, non-array type T can be
  //   converted to an rvalue.
  Expr::isLvalueResult argIsLvalue = From->isLvalue(Context);
  if (argIsLvalue == Expr::LV_Valid && 
      !FromType->isFunctionType() && !FromType->isArrayType() &&
      !FromType->isOverloadType()) {
    SCS.First = ICK_Lvalue_To_Rvalue;

    // If T is a non-class type, the type of the rvalue is the
    // cv-unqualified version of T. Otherwise, the type of the rvalue
    // is T (C++ 4.1p1).
    FromType = FromType.getUnqualifiedType();
  }
  // Array-to-pointer conversion (C++ 4.2)
  else if (FromType->isArrayType()) {
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
  }
  // Function-to-pointer conversion (C++ 4.3).
  else if (FromType->isFunctionType() && argIsLvalue == Expr::LV_Valid) {
    SCS.First = ICK_Function_To_Pointer;

    // An lvalue of function type T can be converted to an rvalue of
    // type "pointer to T." The result is a pointer to the
    // function. (C++ 4.3p1).
    FromType = Context.getPointerType(FromType);
  } 
  // Address of overloaded function (C++ [over.over]).
  else if (FunctionDecl *Fn 
             = ResolveAddressOfOverloadedFunction(From, ToType, false)) {
    SCS.First = ICK_Function_To_Pointer;

    // We were able to resolve the address of the overloaded function,
    // so we can convert to the type of that function.
    FromType = Fn->getType();
    if (ToType->isReferenceType())
      FromType = Context.getReferenceType(FromType);
    else
      FromType = Context.getPointerType(FromType);
  }
  // We don't require any conversions for the first step.
  else {
    SCS.First = ICK_Identity;
  }

  // The second conversion can be an integral promotion, floating
  // point promotion, integral conversion, floating point conversion,
  // floating-integral conversion, pointer conversion,
  // pointer-to-member conversion, or boolean conversion (C++ 4p1).
  if (Context.getCanonicalType(FromType).getUnqualifiedType() ==
      Context.getCanonicalType(ToType).getUnqualifiedType()) {
    // The unqualified versions of the types are the same: there's no
    // conversion to do.
    SCS.Second = ICK_Identity;
  }
  // Integral promotion (C++ 4.5).  
  else if (IsIntegralPromotion(From, FromType, ToType)) {
    SCS.Second = ICK_Integral_Promotion;
    FromType = ToType.getUnqualifiedType();
  } 
  // Floating point promotion (C++ 4.6).
  else if (IsFloatingPointPromotion(FromType, ToType)) {
    SCS.Second = ICK_Floating_Promotion;
    FromType = ToType.getUnqualifiedType();
  } 
  // Integral conversions (C++ 4.7).
  // FIXME: isIntegralType shouldn't be true for enums in C++.
  else if ((FromType->isIntegralType() || FromType->isEnumeralType()) &&
           (ToType->isIntegralType() && !ToType->isEnumeralType())) {
    SCS.Second = ICK_Integral_Conversion;
    FromType = ToType.getUnqualifiedType();
  }
  // Floating point conversions (C++ 4.8).
  else if (FromType->isFloatingType() && ToType->isFloatingType()) {
    SCS.Second = ICK_Floating_Conversion;
    FromType = ToType.getUnqualifiedType();
  }
  // Floating-integral conversions (C++ 4.9).
  // FIXME: isIntegralType shouldn't be true for enums in C++.
  else if ((FromType->isFloatingType() &&
            ToType->isIntegralType() && !ToType->isBooleanType() &&
                                        !ToType->isEnumeralType()) ||
           ((FromType->isIntegralType() || FromType->isEnumeralType()) && 
            ToType->isFloatingType())) {
    SCS.Second = ICK_Floating_Integral;
    FromType = ToType.getUnqualifiedType();
  }
  // Pointer conversions (C++ 4.10).
  else if (IsPointerConversion(From, FromType, ToType, FromType)) {
    SCS.Second = ICK_Pointer_Conversion;
  }
  // FIXME: Pointer to member conversions (4.11).
  // Boolean conversions (C++ 4.12).
  // FIXME: pointer-to-member type
  else if (ToType->isBooleanType() &&
           (FromType->isArithmeticType() ||
            FromType->isEnumeralType() ||
            FromType->isPointerType())) {
    SCS.Second = ICK_Boolean_Conversion;
    FromType = Context.BoolTy;
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
    QualType PromoteTypes[4] = { 
      Context.IntTy, Context.UnsignedIntTy, 
      Context.LongTy, Context.UnsignedLongTy 
    };
    for (int Idx = 0; Idx < 0; ++Idx) {
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
  if (MemberExpr *MemRef = dyn_cast<MemberExpr>(From)) {
    using llvm::APSInt;
    FieldDecl *MemberDecl = MemRef->getMemberDecl();
    APSInt BitWidth;
    if (MemberDecl->isBitField() &&
        FromType->isIntegralType() && !FromType->isEnumeralType() &&
        From->isIntegerConstantExpr(BitWidth, Context)) {
      APSInt ToSize(Context.getTypeSize(ToType));

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
    if (const BuiltinType *ToBuiltin = ToType->getAsBuiltinType())
      if (FromBuiltin->getKind() == BuiltinType::Float &&
          ToBuiltin->getKind() == BuiltinType::Double)
        return true;

  return false;
}

/// IsPointerConversion - Determines whether the conversion of the
/// expression From, which has the (possibly adjusted) type FromType,
/// can be converted to the type ToType via a pointer conversion (C++
/// 4.10). If so, returns true and places the converted type (that
/// might differ from ToType in its cv-qualifiers at some level) into
/// ConvertedType.
bool Sema::IsPointerConversion(Expr *From, QualType FromType, QualType ToType,
                               QualType& ConvertedType)
{
  const PointerType* ToTypePtr = ToType->getAsPointerType();
  if (!ToTypePtr)
    return false;

  // A null pointer constant can be converted to a pointer type (C++ 4.10p1).
  if (From->isNullPointerConstant(Context)) {
    ConvertedType = ToType;
    return true;
  }

  // An rvalue of type "pointer to cv T," where T is an object type,
  // can be converted to an rvalue of type "pointer to cv void" (C++
  // 4.10p2).
  if (FromType->isPointerType() &&
      FromType->getAsPointerType()->getPointeeType()->isObjectType() &&
      ToTypePtr->getPointeeType()->isVoidType()) {
    // We need to produce a pointer to cv void, where cv is the same
    // set of cv-qualifiers as we had on the incoming pointee type.
    QualType toPointee = ToTypePtr->getPointeeType();
    unsigned Quals = Context.getCanonicalType(FromType)->getAsPointerType()
                   ->getPointeeType().getCVRQualifiers();

    if (Context.getCanonicalType(ToTypePtr->getPointeeType()).getCVRQualifiers()
	  == Quals) {
      // ToType is exactly the type we want. Use it.
      ConvertedType = ToType;
    } else {
      // Build a new type with the right qualifiers.
      ConvertedType 
	= Context.getPointerType(Context.VoidTy.getQualifiedType(Quals));
    }
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
  if (const PointerType *FromPtrType = FromType->getAsPointerType())
    if (const PointerType *ToPtrType = ToType->getAsPointerType()) {
      if (FromPtrType->getPointeeType()->isRecordType() &&
          ToPtrType->getPointeeType()->isRecordType() &&
          IsDerivedFrom(FromPtrType->getPointeeType(), 
                        ToPtrType->getPointeeType())) {
        // The conversion is okay. Now, we need to produce the type
        // that results from this conversion, which will have the same
        // qualifiers as the incoming type.
        QualType CanonFromPointee 
          = Context.getCanonicalType(FromPtrType->getPointeeType());
        QualType ToPointee = ToPtrType->getPointeeType();
        QualType CanonToPointee = Context.getCanonicalType(ToPointee);
        unsigned Quals = CanonFromPointee.getCVRQualifiers();

        if (CanonToPointee.getCVRQualifiers() == Quals) {
          // ToType is exactly the type we want. Use it.
          ConvertedType = ToType;
        } else {
          // Build a new type with the right qualifiers.
          ConvertedType
            = Context.getPointerType(CanonToPointee.getQualifiedType(Quals));
        }
        return true;
      }
    }

  return false;
}

/// CheckPointerConversion - Check the pointer conversion from the
/// expression From to the type ToType. This routine checks for
/// ambiguous (FIXME: or inaccessible) derived-to-base pointer
/// conversions for which IsPointerConversion has already returned
/// true. It returns true and produces a diagnostic if there was an
/// error, or returns false otherwise.
bool Sema::CheckPointerConversion(Expr *From, QualType ToType) {
  QualType FromType = From->getType();

  if (const PointerType *FromPtrType = FromType->getAsPointerType())
    if (const PointerType *ToPtrType = ToType->getAsPointerType()) {
      BasePaths Paths(/*FindAmbiguities=*/true, /*RecordPaths=*/false,
                      /*DetectVirtual=*/false);
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

/// IsUserDefinedConversion - Determines whether there is a
/// user-defined conversion sequence (C++ [over.ics.user]) that
/// converts expression From to the type ToType. If such a conversion
/// exists, User will contain the user-defined conversion sequence
/// that performs such a conversion and this routine will return
/// true. Otherwise, this routine returns false and User is
/// unspecified.
bool Sema::IsUserDefinedConversion(Expr *From, QualType ToType, 
                                   UserDefinedConversionSequence& User)
{
  OverloadCandidateSet CandidateSet;
  if (const CXXRecordType *ToRecordType 
        = dyn_cast_or_null<CXXRecordType>(ToType->getAsRecordType())) {
    // C++ [over.match.ctor]p1:
    //   When objects of class type are direct-initialized (8.5), or
    //   copy-initialized from an expression of the same or a
    //   derived class type (8.5), overload resolution selects the
    //   constructor. [...] For copy-initialization, the candidate
    //   functions are all the converting constructors (12.3.1) of
    //   that class. The argument list is the expression-list within
    //   the parentheses of the initializer.
    CXXRecordDecl *ToRecordDecl = ToRecordType->getDecl();
    const OverloadedFunctionDecl *Constructors = ToRecordDecl->getConstructors();
    for (OverloadedFunctionDecl::function_const_iterator func 
           = Constructors->function_begin();
         func != Constructors->function_end(); ++func) {
      CXXConstructorDecl *Constructor = cast<CXXConstructorDecl>(*func);
      if (Constructor->isConvertingConstructor())
        AddOverloadCandidate(Constructor, &From, 1, CandidateSet,
                             /*SuppressUserConversions=*/true);
    }
  }

  if (const CXXRecordType *FromRecordType
        = dyn_cast_or_null<CXXRecordType>(From->getType()->getAsRecordType())) {
    // Add all of the conversion functions as candidates.
    // FIXME: Look for conversions in base classes!
    CXXRecordDecl *FromRecordDecl = FromRecordType->getDecl();
    OverloadedFunctionDecl *Conversions 
      = FromRecordDecl->getConversionFunctions();
    for (OverloadedFunctionDecl::function_iterator Func 
           = Conversions->function_begin();
         Func != Conversions->function_end(); ++Func) {
      CXXConversionDecl *Conv = cast<CXXConversionDecl>(*Func);
      AddConversionCandidate(Conv, From, ToType, CandidateSet);
    }
  }

  OverloadCandidateSet::iterator Best;
  switch (BestViableFunction(CandidateSet, Best)) {
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
          = ThisType->getAsPointerType()->getPointeeType().getAsOpaquePtr();
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
      = FromType1->getAsPointerType()->getPointeeType().getUnqualifiedType();
    QualType FromPointee2
      = FromType2->getAsPointerType()->getPointeeType().getUnqualifiedType();

    if (IsDerivedFrom(FromPointee2, FromPointee1))
      return ImplicitConversionSequence::Better;
    else if (IsDerivedFrom(FromPointee1, FromPointee2))
      return ImplicitConversionSequence::Worse;
  }

  // Compare based on qualification conversions (C++ 13.3.3.2p3,
  // bullet 3).
  if (ImplicitConversionSequence::CompareKind QualCK 
        = CompareQualificationConversions(SCS1, SCS2))
    return QualCK;

  // C++ [over.ics.rank]p3b4:
  //   -- S1 and S2 are reference bindings (8.5.3), and the types to
  //      which the references refer are the same type except for
  //      top-level cv-qualifiers, and the type to which the reference
  //      initialized by S2 refers is more cv-qualified than the type
  //      to which the reference initialized by S1 refers.
  if (SCS1.ReferenceBinding && SCS2.ReferenceBinding) {
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
/// various kinds of derived-to-base conversions (C++ [over.ics.rank]p4b3).
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

  // Compare based on pointer conversions.
  if (SCS1.Second == ICK_Pointer_Conversion && 
      SCS2.Second == ICK_Pointer_Conversion) {
    QualType FromPointee1 
      = FromType1->getAsPointerType()->getPointeeType().getUnqualifiedType();
    QualType ToPointee1 
      = ToType1->getAsPointerType()->getPointeeType().getUnqualifiedType();
    QualType FromPointee2
      = FromType2->getAsPointerType()->getPointeeType().getUnqualifiedType();
    QualType ToPointee2
      = ToType2->getAsPointerType()->getPointeeType().getUnqualifiedType();
    //   -- conversion of C* to B* is better than conversion of C* to A*,
    if (FromPointee1 == FromPointee2 && ToPointee1 != ToPointee2) {
      if (IsDerivedFrom(ToPointee1, ToPointee2))
        return ImplicitConversionSequence::Better;
      else if (IsDerivedFrom(ToPointee2, ToPointee1))
        return ImplicitConversionSequence::Worse;
    }

    //   -- conversion of B* to A* is better than conversion of C* to A*,
    if (FromPointee1 != FromPointee2 && ToPointee1 == ToPointee2) {
      if (IsDerivedFrom(FromPointee2, FromPointee1))
        return ImplicitConversionSequence::Better;
      else if (IsDerivedFrom(FromPointee1, FromPointee2))
        return ImplicitConversionSequence::Worse;
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
/// do not permit any user-defined conversion sequences.
ImplicitConversionSequence 
Sema::TryCopyInitialization(Expr *From, QualType ToType, 
                            bool SuppressUserConversions) {
  if (!getLangOptions().CPlusPlus) {
    // In C, copy initialization is the same as performing an assignment.
    AssignConvertType ConvTy =
      CheckSingleAssignmentConstraints(ToType, From);
    ImplicitConversionSequence ICS;
    if (getLangOptions().NoExtensions? ConvTy != Compatible
                                     : ConvTy == Incompatible)
      ICS.ConversionKind = ImplicitConversionSequence::BadConversion;
    else
      ICS.ConversionKind = ImplicitConversionSequence::StandardConversion;
    return ICS;
  } else if (ToType->isReferenceType()) {
    ImplicitConversionSequence ICS;
    CheckReferenceInit(From, ToType, &ICS, SuppressUserConversions);
    return ICS;
  } else {
    return TryImplicitConversion(From, ToType, SuppressUserConversions);
  }
}

/// PerformArgumentPassing - Pass the argument Arg into a parameter of
/// type ToType. Returns true (and emits a diagnostic) if there was
/// an error, returns false if the initialization succeeded.
bool Sema::PerformCopyInitialization(Expr *&From, QualType ToType, 
                                     const char* Flavor) {
  if (!getLangOptions().CPlusPlus) {
    // In C, argument passing is the same as performing an assignment.
    QualType FromType = From->getType();
    AssignConvertType ConvTy =
      CheckSingleAssignmentConstraints(ToType, From);

    return DiagnoseAssignmentResult(ConvTy, From->getLocStart(), ToType,
                                    FromType, From, Flavor);
  } else if (ToType->isReferenceType()) {
    return CheckReferenceInit(From, ToType);
  } else {
    if (PerformImplicitConversion(From, ToType))
      return Diag(From->getSourceRange().getBegin(),
                  diag::err_typecheck_convert_incompatible,
                  ToType.getAsString(), From->getType().getAsString(),
                  Flavor,
                  From->getSourceRange());
    else
      return false;
  }
}

/// AddOverloadCandidate - Adds the given function to the set of
/// candidate functions, using the given function call arguments.  If
/// @p SuppressUserConversions, then don't allow user-defined
/// conversions via constructors or conversion operators.
void 
Sema::AddOverloadCandidate(FunctionDecl *Function, 
                           Expr **Args, unsigned NumArgs,
                           OverloadCandidateSet& CandidateSet,
                           bool SuppressUserConversions)
{
  const FunctionTypeProto* Proto 
    = dyn_cast<FunctionTypeProto>(Function->getType()->getAsFunctionType());
  assert(Proto && "Functions without a prototype cannot be overloaded");
  assert(!isa<CXXConversionDecl>(Function) && 
         "Use AddConversionCandidate for conversion functions");

  // Add this candidate
  CandidateSet.push_back(OverloadCandidate());
  OverloadCandidate& Candidate = CandidateSet.back();
  Candidate.Function = Function;

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
  Candidate.Viable = true;
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
                                SuppressUserConversions);
      if (Candidate.Conversions[ArgIdx].ConversionKind 
            == ImplicitConversionSequence::BadConversion)
        Candidate.Viable = false;
    } else {
      // (C++ 13.3.2p2): For the purposes of overload resolution, any
      // argument for which there is no corresponding parameter is
      // considered to ""match the ellipsis" (C+ 13.3.3.1.3).
      Candidate.Conversions[ArgIdx].ConversionKind 
        = ImplicitConversionSequence::EllipsisConversion;
    }
  }
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
  Candidate.FinalConversion.setAsIdentityConversion();
  Candidate.FinalConversion.FromTypePtr 
    = Conversion->getConversionType().getAsOpaquePtr();
  Candidate.FinalConversion.ToTypePtr = ToType.getAsOpaquePtr();

  // Determine the implicit conversion sequences for each of the
  // arguments.
  Candidate.Viable = true;
  Candidate.Conversions.resize(1);

  // FIXME: We need to follow the rules for the implicit object
  // parameter.
  QualType ImplicitObjectType 
    = Context.getTypeDeclType(Conversion->getParent());
  ImplicitObjectType 
    = ImplicitObjectType.getQualifiedType(Conversion->getTypeQualifiers());
  ImplicitObjectType = Context.getReferenceType(ImplicitObjectType);
  Candidate.Conversions[0] = TryCopyInitialization(From, ImplicitObjectType, 
                                                   true);
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
                                &ConversionRef, false);
  CallExpr Call(&ConversionFn, 0, 0, 
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

/// AddBuiltinCandidate - Add a candidate for a built-in
/// operator. ResultTy and ParamTys are the result and parameter types
/// of the built-in candidate, respectively. Args and NumArgs are the
/// arguments being passed to the candidate.
void Sema::AddBuiltinCandidate(QualType ResultTy, QualType *ParamTys, 
                               Expr **Args, unsigned NumArgs,
                               OverloadCandidateSet& CandidateSet) {
  // Add this candidate
  CandidateSet.push_back(OverloadCandidate());
  OverloadCandidate& Candidate = CandidateSet.back();
  Candidate.Function = 0;
  Candidate.BuiltinTypes.ResultTy = ResultTy;
  for (unsigned ArgIdx = 0; ArgIdx < NumArgs; ++ArgIdx)
    Candidate.BuiltinTypes.ParamTypes[ArgIdx] = ParamTys[ArgIdx];

  // Determine the implicit conversion sequences for each of the
  // arguments.
  Candidate.Viable = true;
  Candidate.Conversions.resize(NumArgs);
  for (unsigned ArgIdx = 0; ArgIdx < NumArgs; ++ArgIdx) {
    Candidate.Conversions[ArgIdx] 
      = TryCopyInitialization(Args[ArgIdx], ParamTys[ArgIdx], false);
    if (Candidate.Conversions[ArgIdx].ConversionKind 
        == ImplicitConversionSequence::BadConversion)
      Candidate.Viable = false;
  }
}

/// BuiltinCandidateTypeSet - A set of types that will be used for the
/// candidate operator functions for built-in operators (C++
/// [over.built]). The types are separated into pointer types and
/// enumeration types.
class BuiltinCandidateTypeSet  {
  /// TypeSet - A set of types.
  typedef llvm::SmallPtrSet<void*, 8> TypeSet;

  /// PointerTypes - The set of pointer types that will be used in the
  /// built-in candidates.
  TypeSet PointerTypes;

  /// EnumerationTypes - The set of enumeration types that will be
  /// used in the built-in candidates.
  TypeSet EnumerationTypes;

  /// Context - The AST context in which we will build the type sets.
  ASTContext &Context;

  bool AddWithMoreQualifiedTypeVariants(QualType Ty);

public:
  /// iterator - Iterates through the types that are part of the set.
  class iterator {
    TypeSet::iterator Base;

  public:
    typedef QualType                 value_type;
    typedef QualType                 reference;
    typedef QualType                 pointer;
    typedef std::ptrdiff_t           difference_type;
    typedef std::input_iterator_tag  iterator_category;

    iterator(TypeSet::iterator B) : Base(B) { }

    iterator& operator++() {
      ++Base;
      return *this;
    }

    iterator operator++(int) {
      iterator tmp(*this);
      ++(*this);
      return tmp;
    }

    reference operator*() const {
      return QualType::getFromOpaquePtr(*Base);
    }

    pointer operator->() const {
      return **this;
    }

    friend bool operator==(iterator LHS, iterator RHS) {
      return LHS.Base == RHS.Base;
    }

    friend bool operator!=(iterator LHS, iterator RHS) {
      return LHS.Base != RHS.Base;
    }
  };

  BuiltinCandidateTypeSet(ASTContext &Context) : Context(Context) { }

  void AddTypesConvertedFrom(QualType Ty, bool AllowUserConversions = true);

  /// pointer_begin - First pointer type found;
  iterator pointer_begin() { return PointerTypes.begin(); }

  /// pointer_end - Last pointer type found;
  iterator pointer_end() { return PointerTypes.end(); }

  /// enumeration_begin - First enumeration type found;
  iterator enumeration_begin() { return EnumerationTypes.begin(); }

  /// enumeration_end - Last enumeration type found;
  iterator enumeration_end() { return EnumerationTypes.end(); }
};

/// AddWithMoreQualifiedTypeVariants - Add the pointer type @p Ty to
/// the set of pointer types along with any more-qualified variants of
/// that type. For example, if @p Ty is "int const *", this routine
/// will add "int const *", "int const volatile *", "int const
/// restrict *", and "int const volatile restrict *" to the set of
/// pointer types. Returns true if the add of @p Ty itself succeeded,
/// false otherwise.
bool BuiltinCandidateTypeSet::AddWithMoreQualifiedTypeVariants(QualType Ty) {
  // Insert this type.
  if (!PointerTypes.insert(Ty.getAsOpaquePtr()))
    return false;

  if (const PointerType *PointerTy = Ty->getAsPointerType()) {
    QualType PointeeTy = PointerTy->getPointeeType();
    // FIXME: Optimize this so that we don't keep trying to add the same types.

    // FIXME: Do we have to add CVR qualifiers at *all* levels to deal
    // with all pointer conversions that don't cast away constness?
    if (!PointeeTy.isConstQualified())
      AddWithMoreQualifiedTypeVariants
        (Context.getPointerType(PointeeTy.withConst()));
    if (!PointeeTy.isVolatileQualified())
      AddWithMoreQualifiedTypeVariants
        (Context.getPointerType(PointeeTy.withVolatile()));
    if (!PointeeTy.isRestrictQualified())
      AddWithMoreQualifiedTypeVariants
        (Context.getPointerType(PointeeTy.withRestrict()));
  }

  return true;
}

/// AddTypesConvertedFrom - Add each of the types to which the type @p
/// Ty can be implicit converted to the given set of @p Types. We're
/// primarily interested in pointer types, enumeration types,
void BuiltinCandidateTypeSet::AddTypesConvertedFrom(QualType Ty,
                                                    bool AllowUserConversions) {
  // Only deal with canonical types.
  Ty = Context.getCanonicalType(Ty);

  // Look through reference types; they aren't part of the type of an
  // expression for the purposes of conversions.
  if (const ReferenceType *RefTy = Ty->getAsReferenceType())
    Ty = RefTy->getPointeeType();

  // We don't care about qualifiers on the type.
  Ty = Ty.getUnqualifiedType();

  if (const PointerType *PointerTy = Ty->getAsPointerType()) {
    QualType PointeeTy = PointerTy->getPointeeType();

    // Insert our type, and its more-qualified variants, into the set
    // of types.
    if (!AddWithMoreQualifiedTypeVariants(Ty))
      return;

    // Add 'cv void*' to our set of types.
    if (!Ty->isVoidType()) {
      QualType QualVoid 
        = Context.VoidTy.getQualifiedType(PointeeTy.getCVRQualifiers());
      AddWithMoreQualifiedTypeVariants(Context.getPointerType(QualVoid));
    }

    // If this is a pointer to a class type, add pointers to its bases
    // (with the same level of cv-qualification as the original
    // derived class, of course).
    if (const RecordType *PointeeRec = PointeeTy->getAsRecordType()) {
      CXXRecordDecl *ClassDecl = cast<CXXRecordDecl>(PointeeRec->getDecl());
      for (CXXRecordDecl::base_class_iterator Base = ClassDecl->bases_begin();
           Base != ClassDecl->bases_end(); ++Base) {
        QualType BaseTy = Context.getCanonicalType(Base->getType());
        BaseTy = BaseTy.getQualifiedType(PointeeTy.getCVRQualifiers());

        // Add the pointer type, recursively, so that we get all of
        // the indirect base classes, too.
        AddTypesConvertedFrom(Context.getPointerType(BaseTy), false);
      }
    }
  } else if (Ty->isEnumeralType()) {
    EnumerationTypes.insert(Ty.getAsOpaquePtr());
  } else if (AllowUserConversions) {
    if (const RecordType *TyRec = Ty->getAsRecordType()) {
      CXXRecordDecl *ClassDecl = cast<CXXRecordDecl>(TyRec->getDecl());
      // FIXME: Visit conversion functions in the base classes, too.
      OverloadedFunctionDecl *Conversions 
        = ClassDecl->getConversionFunctions();
      for (OverloadedFunctionDecl::function_iterator Func 
             = Conversions->function_begin();
           Func != Conversions->function_end(); ++Func) {
        CXXConversionDecl *Conv = cast<CXXConversionDecl>(*Func);
        AddTypesConvertedFrom(Conv->getConversionType(), false);
      }
    }
  }
}

/// AddBuiltinCandidates - Add the appropriate built-in operator
/// overloads to the candidate set (C++ [over.built]), based on the
/// operator @p Op and the arguments given. For example, if the
/// operator is a binary '+', this routine might add
///   "int operator+(int, int)"
/// to cover integer addition.
void
Sema::AddBuiltinBinaryOperatorCandidates(OverloadedOperatorKind Op, 
                                         Expr **Args, 
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
      Op == OO_Plus || Op == OO_Minus || Op == OO_Equal ||
      Op == OO_PlusEqual || Op == OO_MinusEqual || Op == OO_Subscript ||
      Op == OO_ArrowStar) {
    for (unsigned ArgIdx = 0; ArgIdx < 2; ++ArgIdx)
      CandidateTypes.AddTypesConvertedFrom(Args[ArgIdx]->getType());
  }

  bool isComparison = false;
  switch (Op) {
  case OO_None:
  case NUM_OVERLOADED_OPERATORS:
    assert(false && "Expected an overloaded operator");
    break;

  case OO_New:
  case OO_Delete:
  case OO_Array_New:
  case OO_Array_Delete:
  case OO_Tilde:
  case OO_Exclaim:
  case OO_PlusPlus:
  case OO_MinusMinus:
  case OO_Arrow:
  case OO_Call:
    assert(false && "Expected a binary operator");
    break;

  case OO_Comma:
    // C++ [over.match.oper]p3:
    //   -- For the operator ',', the unary operator '&', or the
    //      operator '->', the built-in candidates set is empty.
    // We don't check '&' or '->' here, since they are unary operators.
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

  case OO_Plus:
  case OO_Minus:
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

  case OO_Star:
  case OO_Slash:
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
    for (unsigned Left = FirstPromotedArithmeticType; 
         Left < LastPromotedArithmeticType; ++Left) {
      for (unsigned Right = FirstPromotedArithmeticType; 
           Right < LastPromotedArithmeticType; ++Right) {
        QualType LandR[2] = { ArithmeticTypes[Left], ArithmeticTypes[Right] };
        QualType Result 
          = isComparison? Context.BoolTy 
                        : UsualArithmeticConversionsType(LandR[0], LandR[1]);
        AddBuiltinCandidate(Result, LandR, Args, 2, CandidateSet);
      }
    }
    break;

  case OO_Percent:
  case OO_Amp:
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
            : UsualArithmeticConversionsType(LandR[0], LandR[1]);
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
      ParamTypes[0] = Context.getReferenceType(*Enum);
      ParamTypes[1] = *Enum;
      AddBuiltinCandidate(ParamTypes[0], ParamTypes, Args, 2, CandidateSet);

      // volatile T& operator=(volatile T&, T)
      ParamTypes[0] = Context.getReferenceType((*Enum).withVolatile());
      ParamTypes[1] = *Enum;
      AddBuiltinCandidate(ParamTypes[0], ParamTypes, Args, 2, CandidateSet);
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
      ParamTypes[0] = Context.getReferenceType(*Ptr);
      AddBuiltinCandidate(ParamTypes[0], ParamTypes, Args, 2, CandidateSet);

      // volatile version
      ParamTypes[0] = Context.getReferenceType((*Ptr).withVolatile());
      AddBuiltinCandidate(ParamTypes[0], ParamTypes, Args, 2, CandidateSet);
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
        ParamTypes[0] = Context.getReferenceType(ArithmeticTypes[Left]);
        AddBuiltinCandidate(ParamTypes[0], ParamTypes, Args, 2, CandidateSet);

        // Add this built-in operator as a candidate (VQ is 'volatile').
        ParamTypes[0] = ArithmeticTypes[Left].withVolatile();
        ParamTypes[0] = Context.getReferenceType(ParamTypes[0]);
        AddBuiltinCandidate(ParamTypes[0], ParamTypes, Args, 2, CandidateSet);
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
        // FIXME: We should be caching these declarations somewhere,
        // rather than re-building them every time.
        ParamTypes[0] = Context.getReferenceType(ArithmeticTypes[Left]);
        AddBuiltinCandidate(ParamTypes[0], ParamTypes, Args, 2, CandidateSet);

        // Add this built-in operator as a candidate (VQ is 'volatile').
        ParamTypes[0] = ArithmeticTypes[Left];
        ParamTypes[0].addVolatile();
        ParamTypes[0] = Context.getReferenceType(ParamTypes[0]);
        AddBuiltinCandidate(ParamTypes[0], ParamTypes, Args, 2, CandidateSet);
      }
    }
    break;

  case OO_AmpAmp:
  case OO_PipePipe: {
    // C++ [over.operator]p23:
    //
    //   There also exist candidate operator functions of the form
    //
    //        bool        operator!(bool);            [In Unary version]
    //        bool        operator&&(bool, bool);
    //        bool        operator||(bool, bool);
    QualType ParamTypes[2] = { Context.BoolTy, Context.BoolTy };
    AddBuiltinCandidate(Context.BoolTy, ParamTypes, Args, 2, CandidateSet);
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
      QualType PointeeType = (*Ptr)->getAsPointerType()->getPointeeType();
      QualType ResultTy = Context.getReferenceType(PointeeType);

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
  }
}

/// AddOverloadCandidates - Add all of the function overloads in Ovl
/// to the candidate set.
void 
Sema::AddOverloadCandidates(const OverloadedFunctionDecl *Ovl, 
                            Expr **Args, unsigned NumArgs,
                            OverloadCandidateSet& CandidateSet,
                            bool SuppressUserConversions)
{
  for (OverloadedFunctionDecl::function_const_iterator Func 
         = Ovl->function_begin();
       Func != Ovl->function_end(); ++Func)
    AddOverloadCandidate(*Func, Args, NumArgs, CandidateSet,
                         SuppressUserConversions);
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

  // FIXME: Deal with the implicit object parameter for static member
  // functions. (C++ 13.3.3p1).

  // (C++ 13.3.3p1): a viable function F1 is defined to be a better
  // function than another viable function F2 if for all arguments i,
  // ICSi(F1) is not a worse conversion sequence than ICSi(F2), and
  // then...
  unsigned NumArgs = Cand1.Conversions.size();
  assert(Cand2.Conversions.size() == NumArgs && "Overload candidate mismatch");
  bool HasBetterConversion = false;
  for (unsigned ArgIdx = 0; ArgIdx < NumArgs; ++ArgIdx) {
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

  if (HasBetterConversion)
    return true;

  // FIXME: Several other bullets in (C++ 13.3.3p1) need to be
  // implemented, but they require template support.

  // C++ [over.match.best]p1b4:
  //
  //   -- the context is an initialization by user-defined conversion
  //      (see 8.5, 13.3.1.5) and the standard conversion sequence
  //      from the return type of F1 to the destination type (i.e.,
  //      the type of the entity being initialized) is a better
  //      conversion sequence than the standard conversion sequence
  //      from the return type of F2 to the destination type.
  if (isa<CXXConversionDecl>(Cand1.Function) && 
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

/// BestViableFunction - Computes the best viable function (C++ 13.3.3) 
/// within an overload candidate set. If overloading is successful,
/// the result will be OR_Success and Best will be set to point to the
/// best viable function within the candidate set. Otherwise, one of
/// several kinds of errors will be returned; see
/// Sema::OverloadingResult.
Sema::OverloadingResult 
Sema::BestViableFunction(OverloadCandidateSet& CandidateSet,
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
        !isBetterOverloadCandidate(*Best, *Cand))
      return OR_Ambiguous;
  }
  
  // Best is the best viable function.
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
        // Normal function
        Diag(Cand->Function->getLocation(), diag::err_ovl_candidate);
      } else {
        // FIXME: We need to get the identifier in here
        // FIXME: Do we want the error message to point at the 
        // operator? (built-ins won't have a location)
        QualType FnType 
          = Context.getFunctionType(Cand->BuiltinTypes.ResultTy,
                                    Cand->BuiltinTypes.ParamTypes,
                                    Cand->Conversions.size(),
                                    false, 0);

        Diag(SourceLocation(), diag::err_ovl_builtin_candidate,
             FnType.getAsString());
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
  if (const PointerLikeType *ToTypePtr = ToType->getAsPointerLikeType())
    FunctionType = ToTypePtr->getPointeeType();

  // We only look at pointers or references to functions.
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
  if (DeclRefExpr *DR = dyn_cast<DeclRefExpr>(OvlExpr))
    Ovl = dyn_cast<OverloadedFunctionDecl>(DR->getDecl());

  // If there's no overloaded function declaration, we're done.
  if (!Ovl)
    return 0;
 
  // Look through all of the overloaded functions, searching for one
  // whose type matches exactly.
  // FIXME: When templates or using declarations come along, we'll actually
  // have to deal with duplicates, partial ordering, etc. For now, we 
  // can just do a simple search.
  FunctionType = Context.getCanonicalType(FunctionType.getUnqualifiedType());
  for (OverloadedFunctionDecl::function_iterator Fun = Ovl->function_begin();
       Fun != Ovl->function_end(); ++Fun) {
    // C++ [over.over]p3:
    //   Non-member functions and static member functions match
    //   targets of type “pointer-to-function”or
    //   “reference-to-function.”
    if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(*Fun))
      if (!Method->isStatic())
        continue;

    if (FunctionType == Context.getCanonicalType((*Fun)->getType()))
      return *Fun;
  }

  return 0;
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
    FixOverloadedFunctionReference(UnOp->getSubExpr(), Fn);
    E->setType(Context.getPointerType(E->getType()));
  } else if (DeclRefExpr *DR = dyn_cast<DeclRefExpr>(E)) {
    assert(isa<OverloadedFunctionDecl>(DR->getDecl()) && 
           "Expected overloaded function");
    DR->setDecl(Fn);
    E->setType(Fn->getType());
  } else {
    assert(false && "Invalid reference to overloaded function");
  }
}

} // end namespace clang
