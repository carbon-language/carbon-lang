//===--- Overload.h - C++ Overloading ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the data structures and types used in C++
// overload resolution.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_OVERLOAD_H
#define LLVM_CLANG_SEMA_OVERLOAD_H

#include "llvm/ADT/SmallVector.h"

namespace clang {
  class CXXConstructorDecl;
  class FunctionDecl;

  /// ImplicitConversionKind - The kind of implicit conversion used to
  /// convert an argument to a parameter's type. The enumerator values
  /// match with Table 9 of (C++ 13.3.3.1.1) and are listed such that
  /// better conversion kinds have smaller values.
  enum ImplicitConversionKind {
    ICK_Identity = 0,          ///< Identity conversion (no conversion)
    ICK_Lvalue_To_Rvalue,      ///< Lvalue-to-rvalue conversion (C++ 4.1)
    ICK_Array_To_Pointer,      ///< Array-to-pointer conversion (C++ 4.2)
    ICK_Function_To_Pointer,   ///< Function-to-pointer (C++ 4.3)
    ICK_Qualification,         ///< Qualification conversions (C++ 4.4)
    ICK_Integral_Promotion,    ///< Integral promotions (C++ 4.5)
    ICK_Floating_Promotion,    ///< Floating point promotions (C++ 4.6)
    ICK_Complex_Promotion,     ///< Complex promotions (Clang extension)
    ICK_Integral_Conversion,   ///< Integral conversions (C++ 4.7)
    ICK_Floating_Conversion,   ///< Floating point conversions (C++ 4.8)
    ICK_Complex_Conversion,    ///< Complex conversions (C99 6.3.1.6)
    ICK_Floating_Integral,     ///< Floating-integral conversions (C++ 4.9)
    ICK_Complex_Real,          ///< Complex-real conversions (C99 6.3.1.7)
    ICK_Pointer_Conversion,    ///< Pointer conversions (C++ 4.10)
    ICK_Pointer_Member,        ///< Pointer-to-member conversions (C++ 4.11)
    ICK_Boolean_Conversion,    ///< Boolean conversions (C++ 4.12)
    ICK_Compatible_Conversion, ///< Conversions between compatible types in C99
    ICK_Derived_To_Base,       ///< Derived-to-base (C++ [over.best.ics])
    ICK_Num_Conversion_Kinds   ///< The number of conversion kinds
  };

  /// ImplicitConversionCategory - The category of an implicit
  /// conversion kind. The enumerator values match with Table 9 of
  /// (C++ 13.3.3.1.1) and are listed such that better conversion
  /// categories have smaller values.
  enum ImplicitConversionCategory {
    ICC_Identity = 0,              ///< Identity
    ICC_Lvalue_Transformation,     ///< Lvalue transformation
    ICC_Qualification_Adjustment,  ///< Qualification adjustment
    ICC_Promotion,                 ///< Promotion
    ICC_Conversion                 ///< Conversion
  };

  ImplicitConversionCategory
  GetConversionCategory(ImplicitConversionKind Kind);

  /// ImplicitConversionRank - The rank of an implicit conversion
  /// kind. The enumerator values match with Table 9 of (C++
  /// 13.3.3.1.1) and are listed such that better conversion ranks
  /// have smaller values.
  enum ImplicitConversionRank {
    ICR_Exact_Match = 0, ///< Exact Match
    ICR_Promotion,       ///< Promotion
    ICR_Conversion       ///< Conversion
  };

  ImplicitConversionRank GetConversionRank(ImplicitConversionKind Kind);

  /// StandardConversionSequence - represents a standard conversion
  /// sequence (C++ 13.3.3.1.1). A standard conversion sequence
  /// contains between zero and three conversions. If a particular
  /// conversion is not needed, it will be set to the identity conversion
  /// (ICK_Identity). Note that the three conversions are
  /// specified as separate members (rather than in an array) so that
  /// we can keep the size of a standard conversion sequence to a
  /// single word.
  struct StandardConversionSequence {
    /// First -- The first conversion can be an lvalue-to-rvalue
    /// conversion, array-to-pointer conversion, or
    /// function-to-pointer conversion.
    ImplicitConversionKind First : 8;

    /// Second - The second conversion can be an integral promotion,
    /// floating point promotion, integral conversion, floating point
    /// conversion, floating-integral conversion, pointer conversion,
    /// pointer-to-member conversion, or boolean conversion.
    ImplicitConversionKind Second : 8;

    /// Third - The third conversion can be a qualification conversion.
    ImplicitConversionKind Third : 8;

    /// Deprecated - Whether this the deprecated conversion of a
    /// string literal to a pointer to non-const character data
    /// (C++ 4.2p2).
    bool Deprecated : 1;

    /// IncompatibleObjC - Whether this is an Objective-C conversion
    /// that we should warn about (if we actually use it).
    bool IncompatibleObjC : 1;

    /// ReferenceBinding - True when this is a reference binding
    /// (C++ [over.ics.ref]).
    bool ReferenceBinding : 1;

    /// DirectBinding - True when this is a reference binding that is a
    /// direct binding (C++ [dcl.init.ref]).
    bool DirectBinding : 1;

    /// RRefBinding - True when this is a reference binding of an rvalue
    /// reference to an rvalue (C++0x [over.ics.rank]p3b4).
    bool RRefBinding : 1;

    /// FromType - The type that this conversion is converting
    /// from. This is an opaque pointer that can be translated into a
    /// QualType.
    void *FromTypePtr;

    /// ToType - The type that this conversion is converting to. This
    /// is an opaque pointer that can be translated into a QualType.
    void *ToTypePtr;

    /// CopyConstructor - The copy constructor that is used to perform
    /// this conversion, when the conversion is actually just the
    /// initialization of an object via copy constructor. Such
    /// conversions are either identity conversions or derived-to-base
    /// conversions.
    CXXConstructorDecl *CopyConstructor;

    void setAsIdentityConversion();
    ImplicitConversionRank getRank() const;
    bool isPointerConversionToBool() const;
    bool isPointerConversionToVoidPointer(ASTContext& Context) const;
    void DebugPrint() const;
  };

  /// UserDefinedConversionSequence - Represents a user-defined
  /// conversion sequence (C++ 13.3.3.1.2).
  struct UserDefinedConversionSequence {
    /// Before - Represents the standard conversion that occurs before
    /// the actual user-defined conversion. (C++ 13.3.3.1.2p1):
    ///
    ///   If the user-defined conversion is specified by a constructor
    ///   (12.3.1), the initial standard conversion sequence converts
    ///   the source type to the type required by the argument of the
    ///   constructor. If the user-defined conversion is specified by
    ///   a conversion function (12.3.2), the initial standard
    ///   conversion sequence converts the source type to the implicit
    ///   object parameter of the conversion function.
    StandardConversionSequence Before;

    /// After - Represents the standard conversion that occurs after
    /// the actual user-defined conversion.
    StandardConversionSequence After;

    /// ConversionFunction - The function that will perform the
    /// user-defined conversion.
    FunctionDecl* ConversionFunction;

    void DebugPrint() const;
  };

  /// ImplicitConversionSequence - Represents an implicit conversion
  /// sequence, which may be a standard conversion sequence
  /// (C++ 13.3.3.1.1), user-defined conversion sequence (C++ 13.3.3.1.2),
  /// or an ellipsis conversion sequence (C++ 13.3.3.1.3).
  struct ImplicitConversionSequence {
    /// Kind - The kind of implicit conversion sequence. BadConversion
    /// specifies that there is no conversion from the source type to
    /// the target type. The enumerator values are ordered such that
    /// better implicit conversions have smaller values.
    enum Kind {
      StandardConversion = 0,
      UserDefinedConversion,
      EllipsisConversion,
      BadConversion
    };

    /// ConversionKind - The kind of implicit conversion sequence.
    Kind ConversionKind;

    union {
      /// When ConversionKind == StandardConversion, provides the
      /// details of the standard conversion sequence.
      StandardConversionSequence Standard;

      /// When ConversionKind == UserDefinedConversion, provides the
      /// details of the user-defined conversion sequence.
      UserDefinedConversionSequence UserDefined;
    };

    // The result of a comparison between implicit conversion
    // sequences. Use Sema::CompareImplicitConversionSequences to
    // actually perform the comparison.
    enum CompareKind {
      Better = -1,
      Indistinguishable = 0,
      Worse = 1
    };

    void DebugPrint() const;
  };

  /// OverloadCandidate - A single candidate in an overload set (C++ 13.3).
  struct OverloadCandidate {
    /// Function - The actual function that this candidate
    /// represents. When NULL, this is a built-in candidate
    /// (C++ [over.oper]) or a surrogate for a conversion to a
    /// function pointer or reference (C++ [over.call.object]).
    FunctionDecl *Function;

    // BuiltinTypes - Provides the return and parameter types of a
    // built-in overload candidate. Only valid when Function is NULL.
    struct {
      QualType ResultTy;
      QualType ParamTypes[3];
    } BuiltinTypes;

    /// Surrogate - The conversion function for which this candidate
    /// is a surrogate, but only if IsSurrogate is true.
    CXXConversionDecl *Surrogate;

    /// Conversions - The conversion sequences used to convert the
    /// function arguments to the function parameters.
    llvm::SmallVector<ImplicitConversionSequence, 4> Conversions;

    /// Viable - True to indicate that this overload candidate is viable.
    bool Viable;

    /// IsSurrogate - True to indicate that this candidate is a
    /// surrogate for a conversion to a function pointer or reference
    /// (C++ [over.call.object]).
    bool IsSurrogate;

    /// IgnoreObjectArgument - True to indicate that the first
    /// argument's conversion, which for this function represents the
    /// implicit object argument, should be ignored. This will be true
    /// when the candidate is a static member function (where the
    /// implicit object argument is just a placeholder) or a
    /// non-static member function when the call doesn't have an
    /// object argument.
    bool IgnoreObjectArgument;

    /// FinalConversion - For a conversion function (where Function is
    /// a CXXConversionDecl), the standard conversion that occurs
    /// after the call to the overload candidate to convert the result
    /// of calling the conversion function to the required type.
    StandardConversionSequence FinalConversion;
  };

  /// OverloadCandidateSet - A set of overload candidates, used in C++
  /// overload resolution (C++ 13.3).
  typedef llvm::SmallVector<OverloadCandidate, 16> OverloadCandidateSet;
} // end namespace clang

#endif // LLVM_CLANG_SEMA_OVERLOAD_H
