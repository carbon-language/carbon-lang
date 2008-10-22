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
#include "clang/Basic/Diagnostic.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
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
    "Boolean conversion"
  };
  return Name[Kind];
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

/// TryCopyInitialization - Attempt to copy-initialize a value of the
/// given type (ToType) from the given expression (Expr), as one would
/// do when copy-initializing a function parameter. This function
/// returns an implicit conversion sequence that can be used to
/// perform the initialization. Given
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
ImplicitConversionSequence
Sema::TryCopyInitialization(Expr* From, QualType ToType)
{
  ImplicitConversionSequence ICS;

  QualType FromType = From->getType();

  // Standard conversions (C++ 4)
  ICS.ConversionKind = ImplicitConversionSequence::StandardConversion;
  ICS.Standard.Deprecated = false;
  ICS.Standard.FromTypePtr = FromType.getAsOpaquePtr();

  if (const ReferenceType *ToTypeRef = ToType->getAsReferenceType()) {
    // FIXME: This is a hack to deal with the initialization of
    // references the way that the C-centric code elsewhere deals with
    // references, by only allowing them if the referred-to type is
    // exactly the same. This means that we're only handling the
    // direct-binding case. The code will be replaced by an
    // implementation of C++ 13.3.3.1.4 once we have the
    // initialization of references implemented.
    QualType ToPointee = Context.getCanonicalType(ToTypeRef->getPointeeType());

    // Get down to the canonical type that we're converting from.
    if (const ReferenceType *FromTypeRef = FromType->getAsReferenceType())
      FromType = FromTypeRef->getPointeeType();
    FromType = Context.getCanonicalType(FromType);

    ICS.Standard.First = ICK_Identity;
    ICS.Standard.Second = ICK_Identity;
    ICS.Standard.Third = ICK_Identity;
    ICS.Standard.ToTypePtr = ToType.getAsOpaquePtr();

    if (FromType != ToPointee)
      ICS.ConversionKind = ImplicitConversionSequence::BadConversion;

    return ICS;
  }

  // The first conversion can be an lvalue-to-rvalue conversion,
  // array-to-pointer conversion, or function-to-pointer conversion
  // (C++ 4p1).

  // Lvalue-to-rvalue conversion (C++ 4.1): 
  //   An lvalue (3.10) of a non-function, non-array type T can be
  //   converted to an rvalue.
  Expr::isLvalueResult argIsLvalue = From->isLvalue(Context);
  if (argIsLvalue == Expr::LV_Valid && 
      !FromType->isFunctionType() && !FromType->isArrayType()) {
    ICS.Standard.First = ICK_Lvalue_To_Rvalue;

    // If T is a non-class type, the type of the rvalue is the
    // cv-unqualified version of T. Otherwise, the type of the rvalue
    // is T (C++ 4.1p1).
    if (!FromType->isRecordType())
      FromType = FromType.getUnqualifiedType();
  }
  // Array-to-pointer conversion (C++ 4.2)
  else if (FromType->isArrayType()) {
    ICS.Standard.First = ICK_Array_To_Pointer;

    // An lvalue or rvalue of type "array of N T" or "array of unknown
    // bound of T" can be converted to an rvalue of type "pointer to
    // T" (C++ 4.2p1).
    FromType = Context.getArrayDecayedType(FromType);

    if (IsStringLiteralToNonConstPointerConversion(From, ToType)) {
      // This conversion is deprecated. (C++ D.4).
      ICS.Standard.Deprecated = true;

      // For the purpose of ranking in overload resolution
      // (13.3.3.1.1), this conversion is considered an
      // array-to-pointer conversion followed by a qualification
      // conversion (4.4). (C++ 4.2p2)
      ICS.Standard.Second = ICK_Identity;
      ICS.Standard.Third = ICK_Qualification;
      ICS.Standard.ToTypePtr = ToType.getAsOpaquePtr();
      return ICS;
    }
  }
  // Function-to-pointer conversion (C++ 4.3).
  else if (FromType->isFunctionType() && argIsLvalue == Expr::LV_Valid) {
    ICS.Standard.First = ICK_Function_To_Pointer;

    // An lvalue of function type T can be converted to an rvalue of
    // type "pointer to T." The result is a pointer to the
    // function. (C++ 4.3p1).
    FromType = Context.getPointerType(FromType);

    // FIXME: Deal with overloaded functions here (C++ 4.3p2).
  } 
  // We don't require any conversions for the first step.
  else {
    ICS.Standard.First = ICK_Identity;
  }

  // The second conversion can be an integral promotion, floating
  // point promotion, integral conversion, floating point conversion,
  // floating-integral conversion, pointer conversion,
  // pointer-to-member conversion, or boolean conversion (C++ 4p1).
  if (Context.getCanonicalType(FromType).getUnqualifiedType() ==
      Context.getCanonicalType(ToType).getUnqualifiedType()) {
    // The unqualified versions of the types are the same: there's no
    // conversion to do.
    ICS.Standard.Second = ICK_Identity;
  }
  // Integral promotion (C++ 4.5).  
  else if (IsIntegralPromotion(From, FromType, ToType)) {
    ICS.Standard.Second = ICK_Integral_Promotion;
    FromType = ToType.getUnqualifiedType();
  } 
  // Floating point promotion (C++ 4.6).
  else if (IsFloatingPointPromotion(FromType, ToType)) {
    ICS.Standard.Second = ICK_Floating_Promotion;
    FromType = ToType.getUnqualifiedType();
  } 
  // Integral conversions (C++ 4.7).
  else if ((FromType->isIntegralType() || FromType->isEnumeralType()) &&
           (ToType->isIntegralType() || ToType->isEnumeralType())) {
    ICS.Standard.Second = ICK_Integral_Conversion;
    FromType = ToType.getUnqualifiedType();
  }
  // Floating point conversions (C++ 4.8).
  else if (FromType->isFloatingType() && ToType->isFloatingType()) {
    ICS.Standard.Second = ICK_Floating_Conversion;
    FromType = ToType.getUnqualifiedType();
  }
  // Floating-integral conversions (C++ 4.9).
  else if ((FromType->isFloatingType() &&
            ToType->isIntegralType() && !ToType->isBooleanType()) ||
           ((FromType->isIntegralType() || FromType->isEnumeralType()) && 
            ToType->isFloatingType())) {
    ICS.Standard.Second = ICK_Floating_Integral;
    FromType = ToType.getUnqualifiedType();
  }
  // Pointer conversions (C++ 4.10).
  else if (IsPointerConversion(From, FromType, ToType, FromType))
    ICS.Standard.Second = ICK_Pointer_Conversion;
  // FIXME: Pointer to member conversions (4.11).
  // Boolean conversions (C++ 4.12).
  // FIXME: pointer-to-member type
  else if (ToType->isBooleanType() &&
           (FromType->isArithmeticType() ||
            FromType->isEnumeralType() ||
            FromType->isPointerType())) {
    ICS.Standard.Second = ICK_Boolean_Conversion;
    FromType = Context.BoolTy;
  } else {
    // No second conversion required.
    ICS.Standard.Second = ICK_Identity;
  }

  // The third conversion can be a qualification conversion (C++ 4p1).
  if (IsQualificationConversion(FromType, ToType)) {
    ICS.Standard.Third = ICK_Qualification;
    FromType = ToType;
  } else {
    // No conversion required
    ICS.Standard.Third = ICK_Identity;
  }

  // If we have not converted the argument type to the parameter type,
  // this is a bad conversion sequence.
  if (Context.getCanonicalType(FromType) != Context.getCanonicalType(ToType))
    ICS.ConversionKind = ImplicitConversionSequence::BadConversion;

  ICS.Standard.ToTypePtr = FromType.getAsOpaquePtr();
  return ICS;
}

/// IsIntegralPromotion - Determines whether the conversion from the
/// expression From (whose potentially-adjusted type is FromType) to
/// ToType is an integral promotion (C++ 4.5). If so, returns true and
/// sets PromotedType to the promoted type.
bool Sema::IsIntegralPromotion(Expr *From, QualType FromType, QualType ToType)
{
  const BuiltinType *To = ToType->getAsBuiltinType();

  // An rvalue of type char, signed char, unsigned char, short int, or
  // unsigned short int can be converted to an rvalue of type int if
  // int can represent all the values of the source type; otherwise,
  // the source rvalue can be converted to an rvalue of type unsigned
  // int (C++ 4.5p1).
  if (FromType->isPromotableIntegerType() && !FromType->isBooleanType() && To) {
    if (// We can promote any signed, promotable integer type to an int
        (FromType->isSignedIntegerType() ||
         // We can promote any unsigned integer type whose size is
         // less than int to an int.
         (!FromType->isSignedIntegerType() && 
          Context.getTypeSize(FromType) < Context.getTypeSize(ToType))))
      return To->getKind() == BuiltinType::Int;
        
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
        return Context.getCanonicalType(FromType).getUnqualifiedType()
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
          (FromType->isSignedIntegerType() && BitWidth <= ToSize))
        return To->getKind() == BuiltinType::Int;
        
      // Are we promoting to an unsigned int from an unsigned bitfield
      // that fits into an unsigned int?
      if (FromType->isUnsignedIntegerType() && BitWidth <= ToSize)
        return To->getKind() == BuiltinType::UInt;

      return false;
    }
  }

  // An rvalue of type bool can be converted to an rvalue of type int,
  // with false becoming zero and true becoming one (C++ 4.5p4).
  if (FromType->isBooleanType() && To && To->getKind() == BuiltinType::Int)
    return true;

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

  // FIXME: An rvalue of type "pointer to cv D," where D is a class
  // type, can be converted to an rvalue of type "pointer to cv B,"
  // where B is a base class (clause 10) of D (C++ 4.10p3).
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

  // FIXME: The other bullets in (C++ 13.3.3.2p4) require support
  // for derived classes.

  // Compare based on qualification conversions (C++ 13.3.3.2p3,
  // bullet 3).
  if (ImplicitConversionSequence::CompareKind CK 
        = CompareQualificationConversions(SCS1, SCS2))
    return CK;

  // FIXME: Handle comparison of reference bindings.

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

/// AddOverloadCandidate - Adds the given function to the set of
/// candidate functions, using the given function call arguments.
void 
Sema::AddOverloadCandidate(FunctionDecl *Function, 
                           Expr **Args, unsigned NumArgs,
                           OverloadCandidateSet& CandidateSet)
{
  const FunctionTypeProto* Proto 
    = dyn_cast<FunctionTypeProto>(Function->getType()->getAsFunctionType());
  assert(Proto && "Functions without a prototype cannot be overloaded");

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
        = TryCopyInitialization(Args[ArgIdx], ParamType);
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

/// AddOverloadCandidates - Add all of the function overloads in Ovl
/// to the candidate set.
void 
Sema::AddOverloadCandidates(OverloadedFunctionDecl *Ovl, 
                            Expr **Args, unsigned NumArgs,
                            OverloadCandidateSet& CandidateSet)
{
  for (OverloadedFunctionDecl::function_iterator Func = Ovl->function_begin();
       Func != Ovl->function_end(); ++Func)
    AddOverloadCandidate(*Func, Args, NumArgs, CandidateSet);
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

  // FIXME: Several other bullets in (C++ 13.3.3p1) need to be implemented.

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
    if (Cand->Viable ||!OnlyViable)
      Diag(Cand->Function->getLocation(), diag::err_ovl_candidate);
  }
}

} // end namespace clang
