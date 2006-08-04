//===--- Declarations.h - Declaration Representation ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines interfaces used for Declarations.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_PARSE_DECLARATIONS_H
#define LLVM_CLANG_PARSE_DECLARATIONS_H

#include "clang/Basic/Diagnostic.h"

namespace llvm {
namespace clang {
  class LangOptions;
  
/// DeclSpec - This class captures information about "declaration specifiers",
/// which encompases storage-class-specifiers, type-specifiers, type-qualifiers,
/// and function-specifiers.
class DeclSpec {
public:
  // storage-class-specifier
  enum SCS {
    SCS_unspecified,
    SCS_typedef,
    SCS_extern,
    SCS_static,
    SCS_auto,
    SCS_register
  } StorageClassSpec : 3;
  
  // type-specifier
  enum TSW {
    TSW_unspecified,
    TSW_short,
    TSW_long,
    TSW_longlong
  } TypeSpecWidth : 2;
  
  enum TSC {
    TSC_unspecified,
    TSC_imaginary,
    TSC_complex
  } TypeSpecComplex : 2;
  
  enum TSS {
    TSS_unspecified,
    TSS_signed,
    TSS_unsigned
  } TypeSpecSign : 2;
  
  enum TST {
    TST_unspecified,
    TST_void,
    TST_char,
    TST_int,
    TST_float,
    TST_double,
    TST_bool,         // _Bool
    TST_decimal32,    // _Decimal32
    TST_decimal64,    // _Decimal64
    TST_decimal128    // _Decimal128
  } TypeSpecType : 4;
  
  // type-qualifiers
  enum TQ {
    TQ_unspecified = 0,
    TQ_const       = 1,
    TQ_restrict    = 2,
    TQ_volatile    = 4
  };
  unsigned TypeQualifiers : 3;  // Bitwise OR of typequals.
  
  // function-specifier
  enum FS {
    FS_unspecified,
    FS_inline
  } FuncSpec : 1;
  
  // attributes.
  // FIXME: implement declspec attributes.
  
  // Flags to query which specifiers were applied.
  enum ParsedSpecifiers {
    PQ_None                  = 0,
    PQ_StorageClassSpecifier = 1,
    PQ_TypeSpecifier         = 2,
    PQ_TypeQualifier         = 4,
    PQ_FunctionSpecifier     = 8
  };
  
  DeclSpec()
    : StorageClassSpec(SCS_unspecified),
      TypeSpecWidth(TSW_unspecified),
      TypeSpecComplex(TSC_unspecified),
      TypeSpecSign(TSS_unspecified),
      TypeSpecType(TST_unspecified),
      TypeQualifiers(TSS_unspecified),
      FuncSpec(FS_unspecified) {
  }
  
  /// getParsedSpecifiers - Return a bitmask of which flavors of specifiers this
  /// DeclSpec includes.
  ///
  unsigned getParsedSpecifiers() const;
  
  /// These methods set the specified attribute of the DeclSpec, but return true
  /// and ignore the request if invalid (e.g. "extern" then "auto" is
  /// specified).  The name of the previous specifier is returned in prevspec.
  bool SetTypeSpecWidth(TSW W, const char *&PrevSpec);
  bool SetTypeSpecComplex(TSC C, const char *&PrevSpec);
  bool SetTypeSpecSign(TSS S, const char *&PrevSpec);
  bool SetTypeSpecType(TST T, const char *&PrevSpec);
  bool SetFuncSpec(FS F, const char *&PrevSpec);
  
  bool SetTypeQual(TQ T, const char *&PrevSpec, const LangOptions &Lang);
  
  /// Finish - This does final analysis of the declspec, rejecting things like
  /// "_Imaginary" (lacking an FP type).  This returns a diagnostic to issue or
  /// diag::NUM_DIAGNOSTICS if there is no error.  After calling this method,
  /// DeclSpec is guaranteed self-consistent, even if an error occurred.
  diag::kind Finish(const LangOptions &Lang);
};
  
}  // end namespace clang
}  // end namespace llvm

#endif
