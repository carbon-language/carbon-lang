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
#include "clang/Parse/Action.h"

namespace llvm {
namespace clang {
  class DeclSpec;
  class Declarator;
  class Scope;

/// Parser - This implements a parser for the C family of languages.  After
/// parsing units of the grammar, productions are invoked to handle whatever has
/// been read.
///
class Parser {
  Preprocessor &PP;
  
  /// Tok - The current token we are peeking head.  All parsing methods assume
  /// that this is valid.
  LexerToken Tok;
  
  unsigned short ParenCount, BracketCount, BraceCount;

  /// Actions - These are the callbacks we invoke as we parse various constructs
  /// in the file.  This refers to the common base class between MinimalActions
  /// and SemaActions for those uses that don't matter.
  Action &Actions;
  
  /// MinimalActions/SemaActions - Exactly one of these two pointers is non-null
  /// depending on whether the client of the parser wants semantic analysis,
  /// name binding, and Decl creation performed or not.
  MinimalAction  *MinimalActions;
  SemanticAction *SemaActions;
  
  Scope *CurScope;
  Diagnostic &Diags;
public:
  Parser(Preprocessor &PP, MinimalAction &MinActions);
  Parser(Preprocessor &PP, SemanticAction &SemaActions);
  ~Parser();

  const LangOptions &getLang() const { return PP.getLangOptions(); }
  TargetInfo &getTargetInfo() const { return PP.getTargetInfo(); }
  Action &getActions() const { return Actions; }
  
  // Type forwarding.  All of these are statically 'void*', but they may all be
  // different actual classes based on the actions in place.
  typedef Action::ExprTy ExprTy;
  typedef Action::StmtTy StmtTy;
  typedef Action::DeclTy DeclTy;
  typedef Action::TypeTy TypeTy;
  
  // Parsing methods.
  
  /// ParseTranslationUnit - All in one method that initializes parses, and
  /// shuts down the parser.
  void ParseTranslationUnit();
  
  /// Initialize - Warm up the parser.
  ///
  void Initialize();
  
  /// ParseTopLevelDecl - Parse one top-level declaration, return whatever the
  /// action tells us to.  This returns true if the EOF was encountered.
  bool ParseTopLevelDecl(DeclTy*& Result);
  
  /// Finalize - Shut down the parser.
  ///
  void Finalize();
  
private:
  //===--------------------------------------------------------------------===//
  // Low-Level token peeking and consumption methods.
  //
  
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
  
  /// isTokenStringLiteral - True if this token is a string-literal.
  ///
  bool isTokenStringLiteral() const {
    return Tok.getKind() == tok::string_literal ||
           Tok.getKind() == tok::wide_string_literal;
  }
  
  /// ConsumeToken - Consume the current 'peek token' and lex the next one.
  /// This does not work will all kinds of tokens: strings and specific other
  /// tokens must be consumed with custom methods below.  This returns the
  /// location of the consumed token.
  SourceLocation ConsumeToken() {
    assert(!isTokenStringLiteral() && !isTokenParen() && !isTokenBracket() &&
           !isTokenBrace() &&
           "Should consume special tokens with Consume*Token");
    SourceLocation L = Tok.getLocation();
    PP.Lex(Tok);
    return L;
  }
  
  /// ConsumeAnyToken - Dispatch to the right Consume* method based on the
  /// current token type.  This should only be used in cases where the type of
  /// the token really isn't known, e.g. in error recovery.
  SourceLocation ConsumeAnyToken() {
    if (isTokenParen())
      return ConsumeParen();
    else if (isTokenBracket())
      return ConsumeBracket();
    else if (isTokenBrace())
      return ConsumeBrace();
    else
      return ConsumeToken();
  }
  
  /// ConsumeParen - This consume method keeps the paren count up-to-date.
  ///
  SourceLocation ConsumeParen() {
    assert(isTokenParen() && "wrong consume method");
    if (Tok.getKind() == tok::l_paren)
      ++ParenCount;
    else if (ParenCount)
      --ParenCount;       // Don't let unbalanced )'s drive the count negative.
    SourceLocation L = Tok.getLocation();
    PP.Lex(Tok);
    return L;
  }
  
  /// ConsumeBracket - This consume method keeps the bracket count up-to-date.
  ///
  SourceLocation ConsumeBracket() {
    assert(isTokenBracket() && "wrong consume method");
    if (Tok.getKind() == tok::l_square)
      ++BracketCount;
    else if (BracketCount)
      --BracketCount;     // Don't let unbalanced ]'s drive the count negative.
    
    SourceLocation L = Tok.getLocation();
    PP.Lex(Tok);
    return L;
  }
      
  /// ConsumeBrace - This consume method keeps the brace count up-to-date.
  ///
  SourceLocation ConsumeBrace() {
    assert(isTokenBrace() && "wrong consume method");
    if (Tok.getKind() == tok::l_brace)
      ++BraceCount;
    else if (BraceCount)
      --BraceCount;     // Don't let unbalanced }'s drive the count negative.
    
    SourceLocation L = Tok.getLocation();
    PP.Lex(Tok);
    return L;
  }
  
  
  /// ConsumeStringToken - Consume the current 'peek token', lexing a new one
  /// and returning the token kind.  This method is specific to strings, as it
  /// handles string literal concatenation, as per C99 5.1.1.2, translation
  /// phase #6.
  SourceLocation ConsumeStringToken() {
    assert(isTokenStringLiteral() &&
           "Should only consume string literals with this method");
    SourceLocation L = Tok.getLocation();
    PP.Lex(Tok);
    return L;
  }
  
  /// MatchRHSPunctuation - For punctuation with a LHS and RHS (e.g. '['/']'),
  /// this helper function matches and consumes the specified RHS token if
  /// present.  If not present, it emits the specified diagnostic indicating
  /// that the parser failed to match the RHS of the token at LHSLoc.  LHSName
  /// should be the name of the unmatched LHS token.  This returns the location
  /// of the consumed token.
  SourceLocation MatchRHSPunctuation(tok::TokenKind RHSTok,
                                     SourceLocation LHSLoc);
  
  /// ExpectAndConsume - The parser expects that 'ExpectedTok' is next in the
  /// input.  If so, it is consumed and false is returned.
  ///
  /// If the input is malformed, this emits the specified diagnostic.  Next, if
  /// SkipToTok is specified, it calls SkipUntil(SkipToTok).  Finally, true is
  /// returned.
  bool ExpectAndConsume(tok::TokenKind ExpectedTok, unsigned Diag,
                        const char *DiagMsg = "",
                        tok::TokenKind SkipToTok = tok::unknown);

  //===--------------------------------------------------------------------===//
  // Scope manipulation
  
  /// EnterScope - Start a new scope.
  void EnterScope(unsigned ScopeFlags);
  
  /// ExitScope - Pop a scope off the scope stack.
  void ExitScope();

  //===--------------------------------------------------------------------===//
  // Diagnostic Emission and Error recovery.
    
  void Diag(SourceLocation Loc, unsigned DiagID, const std::string &Msg = "");
  void Diag(const LexerToken &Tok, unsigned DiagID, const std::string &M = "") {
    Diag(Tok.getLocation(), DiagID, M);
  }
  void Diag(unsigned DiagID, const std::string &Msg = "") {
    Diag(Tok, DiagID, Msg);
  }
  
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
  DeclTy *ParseExternalDeclaration();
  DeclTy *ParseDeclarationOrFunctionDefinition();
  DeclTy *ParseFunctionDefinition(Declarator &D);
  void ParseSimpleAsm();
  void ParseAsmStringLiteral();

  // Objective-C External Declarations 
  void ObjCParseAtDirectives(); 
  void ObjCParseAtClassDeclaration(SourceLocation atLoc);
  void ObjCParseAtInterfaceDeclaration();
  void ObjCParseAtProtocolDeclaration();
  void ObjCParseAtImplementationDeclaration();
  void ObjCParseAtEndDeclaration();
  void ObjCParseAtAliasDeclaration();
  
  void ObjCParseInstanceMethodDeclaration();
  void ObjCParseClassMethodDeclaration();

  //===--------------------------------------------------------------------===//
  // C99 6.5: Expressions.

  typedef Action::ExprResult ExprResult;
  typedef Action::StmtResult StmtResult;
  
  ExprResult ParseExpression();
  ExprResult ParseConstantExpression();
  ExprResult ParseAssignmentExpression();  // Expr that doesn't include commas.
  
  ExprResult ParseExpressionWithLeadingIdentifier(const LexerToken &Tok);
  ExprResult ParseAssignmentExprWithLeadingIdentifier(const LexerToken &Tok);
  ExprResult ParseAssignmentExpressionWithLeadingStar(const LexerToken &Tok);

  ExprResult ParseRHSOfBinaryExpression(ExprResult LHS, unsigned MinPrec);
  ExprResult ParseCastExpression(bool isUnaryExpression);
  ExprResult ParsePostfixExpressionSuffix(ExprResult LHS);
  ExprResult ParseSizeofAlignofExpression();
  ExprResult ParseBuiltinPrimaryExpression();
  
  /// ParenParseOption - Control what ParseParenExpression will parse.
  enum ParenParseOption {
    SimpleExpr,      // Only parse '(' expression ')'
    CompoundStmt,    // Also allow '(' compound-statement ')'
    CompoundLiteral, // Also allow '(' type-name ')' '{' ... '}'
    CastExpr         // Also allow '(' type-name ')' <anything>
  };
  ExprResult ParseParenExpression(ParenParseOption &ExprType, TypeTy *&CastTy,
                                  SourceLocation &RParenLoc);
  
  ExprResult ParseSimpleParenExpression() {  // Parse SimpleExpr only.
    ParenParseOption Op = SimpleExpr;
    TypeTy *CastTy;
    SourceLocation RParenLoc;
    return ParseParenExpression(Op, CastTy, RParenLoc);
  }
  ExprResult ParseStringLiteralExpression();
  
  //===--------------------------------------------------------------------===//
  // C99 6.7.8: Initialization.
  ExprResult ParseInitializer();
  ExprResult ParseInitializerWithPotentialDesignator();
  
  //===--------------------------------------------------------------------===//
  // C99 6.8: Statements and Blocks.
  
  StmtResult ParseStatement() { return ParseStatementOrDeclaration(true); }
  StmtResult ParseStatementOrDeclaration(bool OnlyStatement = false);
  StmtResult ParseIdentifierStatement(bool OnlyStatement);
  StmtResult ParseCaseStatement();
  StmtResult ParseDefaultStatement();
  StmtResult ParseCompoundStatement();
  StmtResult ParseIfStatement();
  StmtResult ParseSwitchStatement();
  StmtResult ParseWhileStatement();
  StmtResult ParseDoStatement();
  StmtResult ParseForStatement();
  StmtResult ParseGotoStatement();
  StmtResult ParseContinueStatement();
  StmtResult ParseBreakStatement();
  StmtResult ParseReturnStatement();
  StmtResult ParseAsmStatement();
  void ParseAsmOperandsOpt();

  //===--------------------------------------------------------------------===//
  // C99 6.7: Declarations.
  
  void ParseDeclaration(unsigned Context);
  DeclTy *ParseInitDeclaratorListAfterFirstDeclarator(Declarator &D);
  void ParseDeclarationSpecifiers(DeclSpec &DS);
  void ParseSpecifierQualifierList(DeclSpec &DS);

  void ParseEnumSpecifier(DeclSpec &DS);
  void ParseStructUnionSpecifier(DeclSpec &DS);

  bool isDeclarationSpecifier() const;
  bool isTypeSpecifierQualifier() const;

  TypeTy *ParseTypeName();
  void ParseAttributes();
  
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
