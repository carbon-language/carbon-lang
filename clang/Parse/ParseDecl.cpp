//===--- ParserDeclarations.cpp - Declaration Parsing ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the Declaration portions of the Parser interfaces.
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/Parser.h"
#include "clang/Parse/Declarations.h"
using namespace llvm;
using namespace clang;

//===----------------------------------------------------------------------===//
// C99 6.7: Declarations.
//===----------------------------------------------------------------------===//

/// ParseDeclarationSpecifiers
///       declaration-specifiers: [C99 6.7]
///         storage-class-specifier declaration-specifiers [opt] [TODO]
///             //__thread
///         type-specifier declaration-specifiers [opt]
///         type-qualifier declaration-specifiers [opt]          [TODO]
/// [C99]   function-specifier declaration-specifiers [opt]      [TODO]
///
///       type-specifier: [C99 6.7.2]
///         'void'
///         'char'
///         'short'
///         'int'
///         'long'
///         'float'
///         'double'
///         'signed'
///         'unsigned'
/// [C99]   '_Bool'
/// [C99]   '_Complex'
/// [C99]   '_Imaginary'
///         struct-or-union-specifier [TODO]
///         enum-specifier [TODO]
///         typedef-name [TODO]
///       function-specifier: [C99 6.7.4]
/// [C99]   inline
///
void Parser::ParseDeclarationSpecifiers(DeclSpec &DS) {
  SourceLocation StartLoc = Tok.getLocation();
  while (1) {
    bool isInvalid = false;
    const char *PrevSpec = 0;
    switch (Tok.getKind()) {
    default:
      // If this is not a declaration specifier token, we're done reading decl
      // specifiers.  First verify that DeclSpec's are consistent.
      diag::kind Res = DS.Finish();
      if (Res != diag::NUM_DIAGNOSTICS)
        Diag(StartLoc, Res);
      return;
    // type-specifiers
    case tok::kw_short:
      isInvalid = DS.SetTypeSpecWidth(DeclSpec::TSW_short, PrevSpec);
      break;
    case tok::kw_long:
      if (DS.TypeSpecWidth != DeclSpec::TSW_long) {
        isInvalid = DS.SetTypeSpecWidth(DeclSpec::TSW_long, PrevSpec);
      } else {
        DS.TypeSpecWidth = DeclSpec::TSW_unspecified;
        isInvalid = DS.SetTypeSpecWidth(DeclSpec::TSW_longlong, PrevSpec);
      }
      break;
    case tok::kw_signed:
      isInvalid = DS.SetTypeSpecSign(DeclSpec::TSS_signed, PrevSpec);
      break;
    case tok::kw_unsigned:
      isInvalid = DS.SetTypeSpecSign(DeclSpec::TSS_unsigned, PrevSpec);
      break;
    case tok::kw__Complex:
      isInvalid = DS.SetTypeSpecComplex(DeclSpec::TSC_complex, PrevSpec);
      break;
    case tok::kw__Imaginary:
      isInvalid = DS.SetTypeSpecComplex(DeclSpec::TSC_imaginary, PrevSpec);
      break;
    case tok::kw_void:
      isInvalid = DS.SetTypeSpecType(DeclSpec::TST_void, PrevSpec);
      break;
    case tok::kw_char:
      isInvalid = DS.SetTypeSpecType(DeclSpec::TST_char, PrevSpec);
      break;
    case tok::kw_int:
      isInvalid = DS.SetTypeSpecType(DeclSpec::TST_int, PrevSpec);
      break;
    case tok::kw_float:
      isInvalid = DS.SetTypeSpecType(DeclSpec::TST_float, PrevSpec);
      break;
    case tok::kw_double:
      isInvalid = DS.SetTypeSpecType(DeclSpec::TST_double, PrevSpec);
      break;
    case tok::kw__Bool:
      isInvalid = DS.SetTypeSpecType(DeclSpec::TST_bool, PrevSpec);
      break;
    case tok::kw__Decimal32:
      isInvalid = DS.SetTypeSpecType(DeclSpec::TST_decimal32, PrevSpec);
      break;
    case tok::kw__Decimal64:
      isInvalid = DS.SetTypeSpecType(DeclSpec::TST_decimal64, PrevSpec);
      break;
    case tok::kw__Decimal128:
      isInvalid = DS.SetTypeSpecType(DeclSpec::TST_decimal128, PrevSpec);
      break;
      
    //case tok::kw_struct:
    //case tok::kw_union:
    //case tok::kw_enum:
      
    // function-specifier
    case tok::kw_inline:
      isInvalid = DS.SetFuncSpec(DeclSpec::FS_inline, PrevSpec);
      break;
    }
    // If the specifier combination wasn't legal, issue a diagnostic.
    if (isInvalid) {
      assert(PrevSpec && "Method did not return previous specifier!");
      Diag(Tok, diag::err_invalid_decl_spec_combination, PrevSpec);
    }
    ConsumeToken();
  }
}


/// ParseDeclarator
///       declarator: [C99 6.7.5]
///         pointer[opt] direct-declarator
///
///       pointer: [C99 6.7.5]
///         '*' type-qualifier-list[opt]
///         '*' type-qualifier-list[opt] pointer
///
void Parser::ParseDeclarator() {
  while (Tok.getKind() == tok::star) {  // '*' -> pointer.
    ConsumeToken();  // Eat the *.
    ParseTypeQualifierListOpt();
  }
  
  ParseDirectDeclarator();
}

/// ParseTypeQualifierListOpt
///       type-qualifier-list: [C99 6.7.5]
///         type-qualifier
/// [GNU]   attributes                         [TODO]
///         type-qualifier-list type-qualifier
/// [GNU]   type-qualifier-list attributes     [TODO]
///
void Parser::ParseTypeQualifierListOpt() {
  while (1) {
    switch (Tok.getKind()) {
    default: break;
      // TODO: attributes.
    case tok::kw_const:
    case tok::kw_volatile:
    case tok::kw_restrict:
      ConsumeToken();
      break;
    }
  }
}


/// ParseDirectDeclarator
///       direct-declarator: [C99 6.7.5]
///         identifier
///         '(' declarator ')'
/// [GNU]   '(' attributes declarator ')'
///         direct-declarator array-declarator
///         direct-declarator '(' parameter-type-list ')'
///         direct-declarator '(' identifier-list[opt] ')'
/// [GNU]   direct-declarator '(' parameter-forward-declarations
///                    parameter-type-list[opt] ')'
///
///       parameter-type-list: [C99 6.7.5]
///         parameter-list
///         parameter-list ',' '...'
///
///       parameter-list: [C99 6.7.5]
///         parameter-declaration
///         parameter-list ',' parameter-declaration
///
///       parameter-declaration: [C99 6.7.5]
///         declaration-specifiers declarator
/// [GNU]   declaration-specifiers declarator attributes
///         declaration-specifiers abstract-declarator[opt] 
/// [GNU]   declaration-specifiers abstract-declarator[opt] attributes
///
///       identifier-list: [C99 6.7.5]
///         identifier
///         identifier-list ',' identifier
///
void Parser::ParseDirectDeclarator() {
  if (Tok.getKind() == tok::identifier) {
    ConsumeToken();
    return;
  }
  // FIXME: missing most stuff.
  assert(0 && "Unknown token!");
}
