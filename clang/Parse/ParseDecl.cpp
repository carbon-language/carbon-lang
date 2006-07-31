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
using namespace llvm;
using namespace clang;

//===----------------------------------------------------------------------===//
// C99 6.7: Declarations.
//===----------------------------------------------------------------------===//

/// ParseDeclarationSpecifiers
///       declaration-specifiers: [C99 6.7]
///         storage-class-specifier declaration-specifiers [opt] [TODO]
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

void Parser::ParseDeclarationSpecifiers() {
  while (1) {
    switch (Tok.getKind()) {
    default: return;  // Not a declaration specifier token.
    // type-specifiers
    case tok::kw_void:       //  SetTypeSpecifier(); break;
    case tok::kw_char:
    case tok::kw_short:     // Width
    case tok::kw_int:
    case tok::kw_long:      // Width
    case tok::kw_float:     // Type specifier
    case tok::kw_double:
    case tok::kw_signed:    // Signedness
    case tok::kw_unsigned:  // Signedness
    case tok::kw__Bool:
    case tok::kw__Complex:   // Complexity
    case tok::kw__Imaginary: // Complexity
                             // FIXME: Read these, handle them!
      ConsumeToken();
      break;
      
    //case tok::kw_struct:
    //case tok::kw_union:
    //case tok::kw_enum:
    }
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
