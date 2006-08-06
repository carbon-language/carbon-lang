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
#include "clang/Basic/SourceLocation.h"

namespace llvm {
namespace clang {
  class LangOptions;
  class IdentifierInfo;
  
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
  
  // storage-class-specifier
  bool SCS_thread_specified : 1;
  
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
  bool FS_inline_specified : 1;
  
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
      SCS_thread_specified(false),
      TypeSpecWidth(TSW_unspecified),
      TypeSpecComplex(TSC_unspecified),
      TypeSpecSign(TSS_unspecified),
      TypeSpecType(TST_unspecified),
      TypeQualifiers(TSS_unspecified),
      FS_inline_specified(false) {
  }
  
  /// getParsedSpecifiers - Return a bitmask of which flavors of specifiers this
  /// DeclSpec includes.
  ///
  unsigned getParsedSpecifiers() const;
  
  /// These methods set the specified attribute of the DeclSpec, but return true
  /// and ignore the request if invalid (e.g. "extern" then "auto" is
  /// specified).  The name of the previous specifier is returned in prevspec.
  bool SetStorageClassSpec(SCS S, const char *&PrevSpec);
  bool SetTypeSpecWidth(TSW W, const char *&PrevSpec);
  bool SetTypeSpecComplex(TSC C, const char *&PrevSpec);
  bool SetTypeSpecSign(TSS S, const char *&PrevSpec);
  bool SetTypeSpecType(TST T, const char *&PrevSpec);
  
  bool SetTypeQual(TQ T, const char *&PrevSpec, const LangOptions &Lang);
  
  /// Finish - This does final analysis of the declspec, issuing diagnostics for
  /// things like "_Imaginary" (lacking an FP type).  After calling this method,
  /// DeclSpec is guaranteed self-consistent, even if an error occurred.
  void Finish(SourceLocation Loc, Diagnostic &D,const LangOptions &Lang);
};


/// DeclaratorInfo - Information about one declarator, including the parsed type
/// information and the identifier.  When the declarator is fully formed, this
/// is turned into the appropriate Decl object.
///
/// Declarators come in two types: normal declarators and abstract declarators.
/// Abstract declarators are used when parsing types, and don't have an
/// identifier.  Normal declarators do have ID's.  One strange case s 
class Declarator {
  const DeclSpec &DS;
  IdentifierInfo *Identifier;
  SourceLocation IdentifierLoc;
  
public:
  enum TheContext {
    FileContext,         // File scope declaration.
    PrototypeContext,    // Within a function prototype.
    KNRTypeListContext,  // K&R type definition list for formals.
    TypeNameContext,     // Abstract declarator for types.
    MemberContext,       // Struct/Union field.
    BlockContext,        // Declaration within a block in a function.
    ForContext           // Declaration within first part of a for loop.
  };
private:
  /// Context - Where we are parsing this declarator.
  ///
  TheContext Context;
public:
  Declarator(const DeclSpec &ds, TheContext C)
    : DS(ds), Identifier(0), Context(C) {
  }
  
  /// mayOmitIdentifier - Return true if the identifier is either optional or
  /// not allowed.  This is true for typenames and prototypes.
  bool mayOmitIdentifier() const {
    return Context == TypeNameContext || Context == PrototypeContext;
  }

  /// mayHaveIdentifier - Return true if the identifier is either optional or
  /// required.  This is true for normal declarators and prototypes, but not
  /// typenames.
  bool mayHaveIdentifier() const {
    return Context != TypeNameContext;
  }
  
  /// isPastIdentifier - Return true if we have parsed beyond the point where
  /// the
  bool isPastIdentifier() const { return IdentifierLoc.isValid(); }
  
  IdentifierInfo *getIdentifier() const { return Identifier; }
  SourceLocation getIdentifierLoc() const { return IdentifierLoc; }
  
  void SetIdentifier(IdentifierInfo *ID, SourceLocation Loc) {
    Identifier = ID;
    IdentifierLoc = Loc;
  }
};

  
}  // end namespace clang
}  // end namespace llvm

#endif
