//===--- Parser.h - C Language Parser ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Parser interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_PARSE_PARSER_H
#define LLVM_CLANG_PARSE_PARSER_H

#include "clang/Lex/Preprocessor.h"
#include "clang/Parse/ParserActions.h"

namespace llvm {
namespace clang {
  class ParserActions;

/// Parser - This implements a parser for the C family of languages.  After
/// parsing units of the grammar, productions are invoked to handle whatever has
/// been read.
///
class Parser {
  // SYMBOL TABLE
  Preprocessor &PP;
  ParserActions &Actions;
  Diagnostic &Diags;
  
  /// Tok - The current token we are peeking head.  All parsing methods assume
  /// that this is valid.
  LexerToken Tok;
public:
  Parser(Preprocessor &PP, ParserActions &Actions);

  // Type forwarding.  All of these are statically 'void*', but they may all be
  // different actual classes based on the actions in place.
  typedef ParserActions::ExprTy ExprTy;
  
  // Parsing methods.
  void ParseTranslationUnit();
  ExprTy ParseExpression();
  

  // Diagnostics.
  void Diag(const LexerToken &Tok, unsigned DiagID, const std::string &Msg="");
  void Diag(unsigned DiagID, const std::string &Msg="") {
    Diag(Tok, DiagID, Msg);
  }

  
  /// ConsumeToken - Consume the current 'peek token', lexing a new one and
  /// returning the token kind.
  tok::TokenKind ConsumeToken() {
    PP.Lex(Tok);
    return Tok.getKind();
  }
  
private:
  //===--------------------------------------------------------------------===//
  // C99 6.9: External Definitions.
  void ParseExternalDeclaration();
  void ParseDeclarationOrFunctionDefinition();

  
  //===--------------------------------------------------------------------===//
  // C99 6.7: Declarations.
  void ParseDeclarationSpecifiers();
  
  void ParseDeclarator();
  void ParseTypeQualifierListOpt();
  void ParseDirectDeclarator();
  
};

}  // end namespace clang
}  // end namespace llvm

#endif
