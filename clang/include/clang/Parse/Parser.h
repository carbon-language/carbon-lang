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
  class DeclSpec;
  class Declarator;
  class Scope;

/// Parser - This implements a parser for the C family of languages.  After
/// parsing units of the grammar, productions are invoked to handle whatever has
/// been read.
///
class Parser {
  Preprocessor &PP;
  ParserActions &Actions;
  Diagnostic &Diags;
  Scope *CurScope;
  unsigned short ParenCount, BracketCount, BraceCount;
  
  /// Tok - The current token we are peeking head.  All parsing methods assume
  /// that this is valid.
  LexerToken Tok;
public:
  Parser(Preprocessor &PP, ParserActions &Actions);
  ~Parser();

  const LangOptions &getLang() const { return PP.getLangOptions(); }
  
  // Type forwarding.  All of these are statically 'void*', but they may all be
  // different actual classes based on the actions in place.
  typedef ParserActions::ExprTy ExprTy;
  
  // Parsing methods.
  void ParseTranslationUnit();
  ExprTy ParseExpression();
  

  // Diagnostics.
  void Diag(SourceLocation Loc, unsigned DiagID, const std::string &Msg = "");
  void Diag(const LexerToken &Tok, unsigned DiagID, const std::string &M = "") {
    Diag(Tok.getLocation(), DiagID, M);
  }
  void Diag(unsigned DiagID, const std::string &Msg = "") {
    Diag(Tok, DiagID, Msg);
  }
  
  /// isTokenParen - Return true if the cur token is '(' or ')'.
  bool isTokenParen() const {
    return Tok.getKind() == tok::l_paren || Tok.getKind() == tok::r_paren;
  }
  /// isTokenBracket - Return true if the cur token is '[' or ']'.
  bool isTokenBracket() const {
    return Tok.getKind() == tok::l_square || Tok.getKind() == tok::r_square;
  }
  /// isTokenBrace - Return true if the cur token is '{' or '}'.
  bool isTokenBrace() const {
    return Tok.getKind() == tok::l_brace || Tok.getKind() == tok::r_brace;
  }
  
  /// ConsumeToken - Consume the current 'peek token' and lex the next one.
  /// This does not work will all kinds of tokens: strings and specific other
  /// tokens must be consumed with custom methods below.
  void ConsumeToken() {
    // Note: update Parser::SkipUntil if any other special tokens are added.
    assert(Tok.getKind() != tok::string_literal &&
           !isTokenParen() && !isTokenBracket() && !isTokenBrace() &&
           "Should consume special tokens with Consume*Token");
    PP.Lex(Tok);
  }
  
  /// ConsumeParen - This consume method keeps the paren count up-to-date.
  ///
  void ConsumeParen() {
    assert(isTokenParen() && "wrong consume method");
    if (Tok.getKind() == tok::l_paren)
      ++ParenCount;
    else if (ParenCount)
      --ParenCount;       // Don't let unbalanced )'s drive the count negative.
    PP.Lex(Tok);
  }
  
  /// ConsumeBracket - This consume method keeps the bracket count up-to-date.
  ///
  void ConsumeBracket() {
    assert(isTokenBracket() && "wrong consume method");
    if (Tok.getKind() == tok::l_square)
      ++BracketCount;
    else if (BracketCount)
      --BracketCount;     // Don't let unbalanced ]'s drive the count negative.
    
    PP.Lex(Tok);
  }
      
  /// ConsumeBrace - This consume method keeps the brace count up-to-date.
  ///
  void ConsumeBrace() {
    assert(isTokenBrace() && "wrong consume method");
    if (Tok.getKind() == tok::l_brace)
      ++BraceCount;
    else if (BraceCount)
      --BraceCount;     // Don't let unbalanced }'s drive the count negative.
    
    PP.Lex(Tok);
  }
  
  
  /// ConsumeStringToken - Consume the current 'peek token', lexing a new one
  /// and returning the token kind.  This method is specific to strings, as it
  /// handles string literal concatenation, as per C99 5.1.1.2, translation
  /// phase #6.
  void ConsumeStringToken() {
    assert(Tok.getKind() != tok::string_literal &&
           "Should consume special tokens with Consume*Token");
    // Due to string literal concatenation, all consequtive string literals are
    // a single token.
    while (Tok.getKind() == tok::string_literal)
      PP.Lex(Tok);
  }
  
private:
  //===--------------------------------------------------------------------===//
  // Error recovery.
    
  /// SkipUntil - Read tokens until we get to the specified token, then consume
  /// it (unless DontConsume is false).  Because we cannot guarantee that the
  /// token will ever occur, this skips to the next token, or to some likely
  /// good stopping point.  If StopAtSemi is true, skipping will stop at a ';'
  /// character.
  /// 
  /// If SkipUntil finds the specified token, it returns true, otherwise it
  /// returns false.  
  bool SkipUntil(tok::TokenKind T, bool StopAtSemi = true,
                 bool DontConsume = false);
    
  //===--------------------------------------------------------------------===//
  // C99 6.9: External Definitions.
  void ParseExternalDeclaration();
  void ParseDeclarationOrFunctionDefinition();
  void ParseFunctionDefinition(Declarator &D);

  //===--------------------------------------------------------------------===//
  // C99 6.8: Statements and Blocks.
  void ParseStatement() { ParseStatementOrDeclaration(true); }
  void ParseStatementOrDeclaration(bool OnlyStatement = false);
  void ParseIdentifierStatement(bool OnlyStatement);
  void ParseCaseStatement();
  void ParseDefaultStatement();
  void ParseCompoundStatement();
  void ParseIfStatement();
  void ParseSwitchStatement();
  void ParseWhileStatement();
  void ParseDoStatement();
  void ParseForStatement();
  void ParseGotoStatement();
  void ParseReturnStatement();

  //===--------------------------------------------------------------------===//
  // C99 6.7: Declarations.

  void ParseDeclaration(unsigned Context);
  void ParseInitDeclaratorListAfterFirstDeclarator(Declarator &D);
  void ParseDeclarationSpecifiers(DeclSpec &DS);
  bool isDeclarationSpecifier() const;
  
  //===--------------------------------------------------------------------===//
  // C99 6.5: Expressions.
  //ExprTy ParseExpression();  // Above.
  void ParseParenExpression();
  
  /// ParseDeclarator - Parse and verify a newly-initialized declarator.
  void ParseDeclarator(Declarator &D);
  void ParseDeclaratorInternal(Declarator &D);
  void ParseTypeQualifierListOpt(DeclSpec &DS);
  void ParseDirectDeclarator(Declarator &D);
  void ParseParenDeclarator(Declarator &D);
  void ParseBracketDeclarator(Declarator &D);
};

}  // end namespace clang
}  // end namespace llvm

#endif
