//===--- Specifiers.h - Declaration and Type Specifiers ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines various enumerations that describe declaration and
// type specifiers.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_SPECIFIERS_H
#define LLVM_CLANG_BASIC_SPECIFIERS_H

namespace clang {
  /// \brief Specifies the width of a type, e.g., short, long, or long long.
  enum TypeSpecifierWidth {
    TSW_unspecified,
    TSW_short,
    TSW_long,
    TSW_longlong
  };
  
  /// \brief Specifies the signedness of a type, e.g., signed or unsigned.
  enum TypeSpecifierSign {
    TSS_unspecified,
    TSS_signed,
    TSS_unsigned
  };
  
  /// \brief Specifies the kind of type.
  enum TypeSpecifierType {
    TST_unspecified,
    TST_void,
    TST_char,
    TST_wchar,        // C++ wchar_t
    TST_char16,       // C++0x char16_t
    TST_char32,       // C++0x char32_t
    TST_int,
    TST_float,
    TST_double,
    TST_bool,         // _Bool
    TST_decimal32,    // _Decimal32
    TST_decimal64,    // _Decimal64
    TST_decimal128,   // _Decimal128
    TST_enum,
    TST_union,
    TST_struct,
    TST_class,        // C++ class type
    TST_typename,     // Typedef, C++ class-name or enum name, etc.
    TST_typeofType,
    TST_typeofExpr,
    TST_decltype,     // C++0x decltype
    TST_underlyingType, // __underlying_type for C++0x
    TST_auto,         // C++0x auto
    TST_unknown_anytype, // __unknown_anytype extension
    TST_error         // erroneous type
  };
  
  /// WrittenBuiltinSpecs - Structure that packs information about the 
  /// type specifiers that were written in a particular type specifier
  /// sequence.
  struct WrittenBuiltinSpecs {
    /*DeclSpec::TST*/ unsigned Type  : 5;
    /*DeclSpec::TSS*/ unsigned Sign  : 2;
    /*DeclSpec::TSW*/ unsigned Width : 2;
    bool ModeAttr : 1;
  };  

  /// AccessSpecifier - A C++ access specifier (public, private,
  /// protected), plus the special value "none" which means
  /// different things in different contexts.
  enum AccessSpecifier {
    AS_public,
    AS_protected,
    AS_private,
    AS_none
  };

  /// ExprValueKind - The categorization of expression values,
  /// currently following the C++0x scheme.
  enum ExprValueKind {
    /// An r-value expression (a gr-value in the C++0x taxonomy)
    /// produces a temporary value.
    VK_RValue,

    /// An l-value expression is a reference to an object with
    /// independent storage.
    VK_LValue,

    /// An x-value expression is a reference to an object with
    /// independent storage but which can be "moved", i.e.
    /// efficiently cannibalized for its resources.
    VK_XValue
  };

  /// A further classification of the kind of object referenced by an
  /// l-value or x-value.
  enum ExprObjectKind {
    /// An ordinary object is located at an address in memory.
    OK_Ordinary,

    /// A bitfield object is a bitfield on a C or C++ record.
    OK_BitField,

    /// A vector component is an element or range of elements on a vector.
    OK_VectorComponent,

    /// An Objective C property is a logical field of an Objective-C
    /// object which is read and written via Objective C method calls.
    OK_ObjCProperty
  };

  // \brief Describes the kind of template specialization that a
  // particular template specialization declaration represents.
  enum TemplateSpecializationKind {
    /// This template specialization was formed from a template-id but
    /// has not yet been declared, defined, or instantiated.
    TSK_Undeclared = 0,
    /// This template specialization was implicitly instantiated from a
    /// template. (C++ [temp.inst]).
    TSK_ImplicitInstantiation,
    /// This template specialization was declared or defined by an
    /// explicit specialization (C++ [temp.expl.spec]) or partial
    /// specialization (C++ [temp.class.spec]).
    TSK_ExplicitSpecialization,
    /// This template specialization was instantiated from a template
    /// due to an explicit instantiation declaration request
    /// (C++0x [temp.explicit]).
    TSK_ExplicitInstantiationDeclaration,
    /// This template specialization was instantiated from a template
    /// due to an explicit instantiation definition request
    /// (C++ [temp.explicit]).
    TSK_ExplicitInstantiationDefinition
  };

  /// \brief Storage classes.
  enum StorageClass {
    // These are legal on both functions and variables.
    SC_None,
    SC_Extern,
    SC_Static,
    SC_PrivateExtern,

    // These are only legal on variables.
    SC_Auto,
    SC_Register
  };

  /// Checks whether the given storage class is legal for functions.
  inline bool isLegalForFunction(StorageClass SC) {
    return SC <= SC_PrivateExtern;
  }

  /// Checks whether the given storage class is legal for variables.
  inline bool isLegalForVariable(StorageClass SC) {
    return true;
  }
} // end namespace clang

#endif // LLVM_CLANG_BASIC_SPECIFIERS_H
