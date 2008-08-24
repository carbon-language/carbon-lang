//===--- Parser.h - C Language Parser ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include <stack>

namespace clang {
  class DeclSpec;
  class ObjCDeclSpec;
  class Declarator;
  class FieldDeclarator;
  class AttributeList;
  class Scope;

/// Parser - This implements a parser for the C family of languages.  After
/// parsing units of the grammar, productions are invoked to handle whatever has
/// been read.
///
class Parser {
  Preprocessor &PP;
  
  /// Tok - The current token we are peeking ahead.  All parsing methods assume
  /// that this is valid.
  Token Tok;
  
  unsigned short ParenCount, BracketCount, BraceCount;

  /// Actions - These are the callbacks we invoke as we parse various constructs
  /// in the file.  This refers to the common base class between MinimalActions
  /// and SemaActions for those uses that don't matter.
  Action &Actions;
  
  Scope *CurScope;
  Diagnostic &Diags;
  
  /// ScopeCache - Cache scopes to reduce malloc traffic.
  enum { ScopeCacheSize = 16 };
  unsigned NumCachedScopes;
  Scope *ScopeCache[ScopeCacheSize];

  /// Ident_super - IdentifierInfo for "super", to support fast
  /// comparison.
  IdentifierInfo *Ident_super;

public:
  Parser(Preprocessor &PP, Action &Actions);
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
  
  /// ParseTopLevelDecl - Parse one top-level declaration. Returns true if 
  /// the EOF was encountered.
  bool ParseTopLevelDecl(DeclTy*& Result);
  
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
    else if (isTokenStringLiteral())
      return ConsumeStringToken();
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
  
  /// GetLookAheadToken - This peeks ahead N tokens and returns that token
  /// without consuming any tokens.  LookAhead(0) returns 'Tok', LookAhead(1)
  /// returns the token after Tok, etc.
  ///
  /// Note that this differs from the Preprocessor's LookAhead method, because
  /// the Parser always has one token lexed that the preprocessor doesn't.
  ///
  const Token &GetLookAheadToken(unsigned N) {
    if (N == 0 || Tok.is(tok::eof)) return Tok;
    return PP.LookAhead(N-1);
  }

  /// NextToken - This peeks ahead one token and returns it without
  /// consuming it.
  const Token &NextToken() {
    return PP.LookAhead(0);
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
    
  bool Diag(SourceLocation Loc, unsigned DiagID,
            const std::string &Msg = std::string());
  bool Diag(SourceLocation Loc, unsigned DiagID, const SourceRange &R);
  bool Diag(SourceLocation Loc, unsigned DiagID, const std::string &Msg,
            const SourceRange& R1);
  bool Diag(const Token &Tok, unsigned DiagID,
            const std::string &M = std::string()) {
    return Diag(Tok.getLocation(), DiagID, M);
  }
  
  /// SkipUntil - Read tokens until we get to the specified token, then consume
  /// it (unless DontConsume is true).  Because we cannot guarantee that the
  /// token will ever occur, this skips to the next token, or to some likely
  /// good stopping point.  If StopAtSemi is true, skipping will stop at a ';'
  /// character.
  /// 
  /// If SkipUntil finds the specified token, it returns true, otherwise it
  /// returns false.  
  bool SkipUntil(tok::TokenKind T, bool StopAtSemi = true,
                 bool DontConsume = false) {
    return SkipUntil(&T, 1, StopAtSemi, DontConsume);
  }
  bool SkipUntil(tok::TokenKind T1, tok::TokenKind T2, bool StopAtSemi = true,
                 bool DontConsume = false) {
    tok::TokenKind TokArray[] = {T1, T2};
    return SkipUntil(TokArray, 2, StopAtSemi, DontConsume);
  }
  bool SkipUntil(const tok::TokenKind *Toks, unsigned NumToks,
                 bool StopAtSemi = true, bool DontConsume = false);
 
  typedef Action::ExprResult ExprResult;
  typedef Action::StmtResult StmtResult;
    
  //===--------------------------------------------------------------------===//
  // Lexing and parsing of C++ inline methods.

  typedef llvm::SmallVector<Token, 32> TokensTy;
  struct LexedMethod {
    Action::DeclTy *D;
    TokensTy Toks;
    explicit LexedMethod(Action::DeclTy *MD) : D(MD) {}
  };

  /// LexedMethodsForTopClass - During parsing of a top (non-nested) C++ class,
  /// its inline method definitions and the inline method definitions of its
  /// nested classes are lexed and stored here.
  typedef std::stack<LexedMethod> LexedMethodsForTopClass;

  /// TopClassStacks - This is initialized with one LexedMethodsForTopClass used
  /// for lexing all top classes, until a local class in an inline method is
  /// encountered, at which point a new LexedMethodsForTopClass is pushed here
  /// and used until the parsing of that local class is finished.
  std::stack<LexedMethodsForTopClass> TopClassStacks;

  LexedMethodsForTopClass &getCurTopClassStack() {
    assert(!TopClassStacks.empty() && "No lexed method stacks!");
    return TopClassStacks.top();
  }

  void PushTopClassStack() {
    TopClassStacks.push(LexedMethodsForTopClass());
  }
  void PopTopClassStack() { TopClassStacks.pop(); }

  DeclTy *ParseCXXInlineMethodDef(AccessSpecifier AS, Declarator &D);
  void ParseLexedMethodDefs();
  bool ConsumeAndStoreUntil(tok::TokenKind T, TokensTy &Toks);

  //===--------------------------------------------------------------------===//
  // C99 6.9: External Definitions.
  DeclTy *ParseExternalDeclaration();
  DeclTy *ParseDeclarationOrFunctionDefinition();
  DeclTy *ParseFunctionDefinition(Declarator &D);
  void ParseKNRParamDeclarations(Declarator &D);
  ExprResult ParseSimpleAsm();
  ExprResult ParseAsmStringLiteral();

  // Objective-C External Declarations
  DeclTy *ParseObjCAtDirectives(); 
  DeclTy *ParseObjCAtClassDeclaration(SourceLocation atLoc);
  DeclTy *ParseObjCAtInterfaceDeclaration(SourceLocation atLoc, 
                                          AttributeList *prefixAttrs = 0);
  void ParseObjCClassInstanceVariables(DeclTy *interfaceDecl, 
                                       SourceLocation atLoc);
  bool ParseObjCProtocolReferences(llvm::SmallVectorImpl<Action::DeclTy*> &P,
                                   bool WarnOnDeclarations, 
                                   SourceLocation &EndProtoLoc);
  void ParseObjCInterfaceDeclList(DeclTy *interfaceDecl,
                                  tok::ObjCKeywordKind contextKey);
  DeclTy *ParseObjCAtProtocolDeclaration(SourceLocation atLoc);
  
  DeclTy *ObjCImpDecl;

  DeclTy *ParseObjCAtImplementationDeclaration(SourceLocation atLoc);
  DeclTy *ParseObjCAtEndDeclaration(SourceLocation atLoc);
  DeclTy *ParseObjCAtAliasDeclaration(SourceLocation atLoc);
  DeclTy *ParseObjCPropertySynthesize(SourceLocation atLoc);
  DeclTy *ParseObjCPropertyDynamic(SourceLocation atLoc);
  
  IdentifierInfo *ParseObjCSelector(SourceLocation &MethodLocation);
  // Definitions for Objective-c context sensitive keywords recognition.
  enum ObjCTypeQual {
    objc_in=0, objc_out, objc_inout, objc_oneway, objc_bycopy, objc_byref,
    objc_NumQuals
  };
  IdentifierInfo *ObjCTypeQuals[objc_NumQuals];
  // Definitions for ObjC2's @property attributes.
  enum ObjCPropertyAttr {
    objc_readonly=0, objc_getter, objc_setter, objc_assign, 
    objc_readwrite, objc_retain, objc_copy, objc_nonatomic, objc_NumAttrs
  };
  IdentifierInfo *ObjCPropertyAttrs[objc_NumAttrs];
  bool isObjCPropertyAttribute();
  
  bool isTokIdentifier_in() const;

  TypeTy *ParseObjCTypeName(ObjCDeclSpec &DS);
  void ParseObjCMethodRequirement();
  DeclTy *ParseObjCMethodPrototype(DeclTy *classOrCat,
            tok::ObjCKeywordKind MethodImplKind = tok::objc_not_keyword);
  DeclTy *ParseObjCMethodDecl(SourceLocation mLoc, tok::TokenKind mType,
                              DeclTy *classDecl,
            tok::ObjCKeywordKind MethodImplKind = tok::objc_not_keyword);
  void ParseObjCPropertyAttribute(ObjCDeclSpec &DS);
  
  DeclTy *ParseObjCMethodDefinition();
  
  //===--------------------------------------------------------------------===//
  // C99 6.5: Expressions.

  ExprResult ParseExpression();
  ExprResult ParseConstantExpression();
  ExprResult ParseAssignmentExpression();  // Expr that doesn't include commas.
  
  ExprResult ParseExpressionWithLeadingAt(SourceLocation AtLoc);

  ExprResult ParseRHSOfBinaryExpression(ExprResult LHS, unsigned MinPrec);
  ExprResult ParseCastExpression(bool isUnaryExpression);
  ExprResult ParsePostfixExpressionSuffix(ExprResult LHS);
  ExprResult ParseSizeofAlignofExpression();
  ExprResult ParseBuiltinPrimaryExpression();

  typedef llvm::SmallVector<ExprTy*, 8> ExprListTy;
  typedef llvm::SmallVector<SourceLocation, 8> CommaLocsTy;

  /// ParseExpressionList - Used for C/C++ (argument-)expression-list.
  bool ParseExpressionList(ExprListTy &Exprs, CommaLocsTy &CommaLocs);
  
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
    SourceLocation RParenLoc;
    return ParseSimpleParenExpression(RParenLoc);
  }
  ExprResult ParseSimpleParenExpression(SourceLocation &RParenLoc) {
    ParenParseOption Op = SimpleExpr;
    TypeTy *CastTy;
    return ParseParenExpression(Op, CastTy, RParenLoc);
  }
  ExprResult ParseStringLiteralExpression();
  
  //===--------------------------------------------------------------------===//
  // C++ 5.2p1: C++ Casts
  ExprResult ParseCXXCasts();

  //===--------------------------------------------------------------------===//
  // C++ 9.3.2: C++ 'this' pointer
  ExprResult ParseCXXThis();

  //===--------------------------------------------------------------------===//
  // C++ 15: C++ Throw Expression
  ExprResult ParseThrowExpression();

  //===--------------------------------------------------------------------===//
  // C++ 2.13.5: C++ Boolean Literals
  ExprResult ParseCXXBoolLiteral();

  //===--------------------------------------------------------------------===//
  // C++ 5.2.3: Explicit type conversion (functional notation)
  ExprResult ParseCXXTypeConstructExpression(const DeclSpec &DS);

  /// ParseCXXSimpleTypeSpecifier - [C++ 7.1.5.2] Simple type specifiers.
  /// This should only be called when the current token is known to be part of
  /// simple-type-specifier.
  void ParseCXXSimpleTypeSpecifier(DeclSpec &DS);

  //===--------------------------------------------------------------------===//
  // C99 6.7.8: Initialization.
  ExprResult ParseInitializer();
  ExprResult ParseInitializerWithPotentialDesignator();
  
  //===--------------------------------------------------------------------===//
  // Objective-C Expressions
  
  bool isTokObjCMessageIdentifierReceiver() const {
    if (!Tok.is(tok::identifier))
      return false;
    
    if (Actions.isTypeName(*Tok.getIdentifierInfo(), CurScope))
      return true;
    
    return Tok.getIdentifierInfo() == Ident_super;
  }
  
  ExprResult ParseObjCAtExpression(SourceLocation AtLocation);
  ExprResult ParseObjCStringLiteral(SourceLocation AtLoc);
  ExprResult ParseObjCEncodeExpression(SourceLocation AtLoc);
  ExprResult ParseObjCSelectorExpression(SourceLocation AtLoc);
  ExprResult ParseObjCProtocolExpression(SourceLocation AtLoc);
  ExprResult ParseObjCMessageExpression();
  ExprResult ParseObjCMessageExpressionBody(SourceLocation LBracloc,
                                            IdentifierInfo *ReceiverName,
                                            ExprTy *ReceiverExpr);
  ExprResult ParseAssignmentExprWithObjCMessageExprStart(SourceLocation LBracloc,
                                                         IdentifierInfo *ReceiverName,
                                                         ExprTy *ReceiverExpr);
    
  //===--------------------------------------------------------------------===//
  // C99 6.8: Statements and Blocks.
  
  StmtResult ParseStatement() { return ParseStatementOrDeclaration(true); }
  StmtResult ParseStatementOrDeclaration(bool OnlyStatement = false);
  StmtResult ParseLabeledStatement();
  StmtResult ParseCaseStatement();
  StmtResult ParseDefaultStatement();
  StmtResult ParseCompoundStatement(bool isStmtExpr = false);
  StmtResult ParseCompoundStatementBody(bool isStmtExpr = false);
  StmtResult ParseIfStatement();
  StmtResult ParseSwitchStatement();
  StmtResult ParseWhileStatement();
  StmtResult ParseDoStatement();
  StmtResult ParseForStatement();
  StmtResult ParseGotoStatement();
  StmtResult ParseContinueStatement();
  StmtResult ParseBreakStatement();
  StmtResult ParseReturnStatement();
  StmtResult ParseAsmStatement(bool &msAsm);
  StmtResult FuzzyParseMicrosoftAsmStatement();
  StmtResult ParseObjCAtStatement(SourceLocation atLoc);
  StmtResult ParseObjCTryStmt(SourceLocation atLoc);
  StmtResult ParseObjCThrowStmt(SourceLocation atLoc);
  StmtResult ParseObjCSynchronizedStmt(SourceLocation atLoc);
  bool ParseAsmOperandsOpt(llvm::SmallVectorImpl<std::string> &Names,
                           llvm::SmallVectorImpl<ExprTy*> &Constraints,
                           llvm::SmallVectorImpl<ExprTy*> &Exprs);


  //===--------------------------------------------------------------------===//
  // C99 6.7: Declarations.
  
  DeclTy *ParseDeclaration(unsigned Context);
  DeclTy *ParseSimpleDeclaration(unsigned Context);
  DeclTy *ParseInitDeclaratorListAfterFirstDeclarator(Declarator &D);
  DeclTy *ParseFunctionStatementBody(DeclTy *Decl, 
                                     SourceLocation L, SourceLocation R);
  void ParseDeclarationSpecifiers(DeclSpec &DS);
  void ParseSpecifierQualifierList(DeclSpec &DS);
  
  void ParseObjCTypeQualifierList(ObjCDeclSpec &DS);

  bool ParseTag(DeclTy *&Decl, unsigned TagType, SourceLocation StartLoc);
  void ParseEnumSpecifier(DeclSpec &DS);
  void ParseEnumBody(SourceLocation StartLoc, DeclTy *TagDecl);
  void ParseStructUnionBody(SourceLocation StartLoc, unsigned TagType,
                            DeclTy *TagDecl);
  void ParseStructDeclaration(DeclSpec &DS,
                              llvm::SmallVectorImpl<FieldDeclarator> &Fields);
                              
  bool isDeclarationSpecifier() const;
  bool isTypeSpecifierQualifier() const;
  bool isTypeQualifier() const;

  TypeTy *ParseTypeName();
  AttributeList *ParseAttributes();
  void ParseTypeofSpecifier(DeclSpec &DS);
  
  /// ParseDeclarator - Parse and verify a newly-initialized declarator.
  void ParseDeclarator(Declarator &D);
  void ParseDeclaratorInternal(Declarator &D);
  void ParseTypeQualifierListOpt(DeclSpec &DS);
  void ParseDirectDeclarator(Declarator &D);
  void ParseParenDeclarator(Declarator &D);
  void ParseFunctionDeclarator(SourceLocation LParenLoc, Declarator &D);
  void ParseFunctionDeclaratorIdentifierList(SourceLocation LParenLoc,
                                             Declarator &D);
  void ParseBracketDeclarator(Declarator &D);
  
  //===--------------------------------------------------------------------===//
  // C++ 7: Declarations [dcl.dcl]
  
  DeclTy *ParseNamespace(unsigned Context);
  DeclTy *ParseLinkage(unsigned Context);

  //===--------------------------------------------------------------------===//
  // C++ 9: classes [class] and C structs/unions.
  void ParseClassSpecifier(DeclSpec &DS);
  void ParseCXXMemberSpecification(SourceLocation StartLoc, unsigned TagType,
                                   DeclTy *TagDecl);
  DeclTy *ParseCXXClassMemberDeclaration(AccessSpecifier AS);

  //===--------------------------------------------------------------------===//
  // C++ 10: Derived classes [class.derived]
  void ParseBaseClause(DeclTy *ClassDecl);
  bool ParseBaseSpecifier(DeclTy *ClassDecl);
  AccessSpecifier getAccessSpecifierIfPresent() const;
};

}  // end namespace clang

#endif
