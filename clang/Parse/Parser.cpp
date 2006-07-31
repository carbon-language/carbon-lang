//===--- Parser.cpp - C Language Family Parser ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the Parser interfaces.
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/Parser.h"
#include "clang/Basic/Diagnostic.h"
using namespace llvm;
using namespace clang;

Parser::Parser(Preprocessor &pp, ParserActions &actions)
  : PP(pp), Actions(actions), Diags(PP.getDiagnostics()) {}

void Parser::Diag(const LexerToken &Tok, unsigned DiagID,
                  const std::string &Msg) {
  Diags.Report(Tok.getLocation(), DiagID, Msg);
}

//===----------------------------------------------------------------------===//
// C99 6.9: External Definitions.
//===----------------------------------------------------------------------===//

/// ParseTranslationUnit:
///       translation-unit: [C99 6.9]
///         external-declaration 
///         translation-unit external-declaration 
void Parser::ParseTranslationUnit() {

  if (Tok.getKind() == tok::eof)  // Empty source file is an extension.
    Diag(diag::ext_empty_source_file);
  
  while (Tok.getKind() != tok::eof)
    ParseExternalDeclaration();
}

/// ParseExternalDeclaration:
///       external-declaration: [C99 6.9]
///         function-definition        [TODO]
///         declaration                [TODO]
/// [EXT]   ';'
/// [GNU]   asm-definition             [TODO]
/// [GNU]   __extension__ external-declaration     [TODO]
/// [OBJC]  objc-class-definition      [TODO]
/// [OBJC]  objc-class-declaration     [TODO]
/// [OBJC]  objc-alias-declaration     [TODO]
/// [OBJC]  objc-protocol-definition   [TODO]
/// [OBJC]  objc-method-definition     [TODO]
/// [OBJC]  @end                       [TODO]
///
void Parser::ParseExternalDeclaration() {
  switch (Tok.getKind()) {
  case tok::semi:
    Diag(diag::ext_top_level_semi);
    ConsumeToken();
    break;
  default:
    // We can't tell whether this is a function-definition or declaration yet.
    ParseDeclarationOrFunctionDefinition();
    break;
  }
}

/// ParseDeclarationOrFunctionDefinition - Parse either a function-definition or
/// a declaration.  We can't tell which we have until we read up to the
/// compound-statement in function-definition.
///
///       function-definition: [C99 6.9.1]
///         declaration-specifiers[opt] declarator declaration-list[opt] 
///                 compound-statement                           [TODO]
///       declaration: [C99 6.7]
///         declaration-specifiers init-declarator-list[opt] ';' [TODO]
/// [!C99]  init-declarator-list ';' [TODO]
/// [OMP]   threadprivate-directive                              [TODO]
///
///       init-declarator-list: [C99 6.7]
///         init-declarator
///         init-declarator-list ',' init-declarator
///       init-declarator: [C99 6.7]
///         declarator
///         declarator '=' initializer
///
void Parser::ParseDeclarationOrFunctionDefinition() {
  // Parse the common declaration-specifiers piece.
  // NOTE: this can not be missing for C99 declaration's.
  ParseDeclarationSpecifiers();
  
  // Parse the common declarator piece.
  ParseDeclarator();

  // If the declarator was a function type...
  
  switch (Tok.getKind()) {
  case tok::equal:   // must be: decl-spec[opt] declarator init-declarator-list
  case tok::comma:   // must be: decl-spec[opt] declarator init-declarator-list
  default:
    assert(0 && "unimp!");
  case tok::semi:
    ConsumeToken();
    break;
  }
}

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
