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
    TST_auto,         // C++0x auto
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
}

#endif // LLVM_CLANG_BASIC_SPECIFIERS_H
