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
#include "clang/Parse/DeclSpec.h"
#include <stack>

namespace clang {
  class AttributeList;
  class DeclSpec;
  class Declarator;
  class FieldDeclarator;
  class ObjCDeclSpec;
  class PragmaHandler;
  class Scope;
  class DiagnosticBuilder;

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

  PragmaHandler *PackHandler;

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
  typedef Action::BaseTy BaseTy;
  typedef Action::MemInitTy MemInitTy;
  typedef Action::CXXScopeTy CXXScopeTy;
  typedef Action::TemplateParamsTy TemplateParamsTy;
  typedef Action::TemplateArgTy TemplateArgTy;

  typedef llvm::SmallVector<TemplateParamsTy *, 4> TemplateParameterLists;

  typedef Action::ExprResult        ExprResult;
  typedef Action::StmtResult        StmtResult;
  typedef Action::BaseResult        BaseResult;
  typedef Action::MemInitResult     MemInitResult;

  typedef Action::OwningExprResult OwningExprResult;
  typedef Action::OwningStmtResult OwningStmtResult;
  typedef Action::OwningTemplateArgResult OwningTemplateArgResult;

  typedef Action::ExprArg ExprArg;
  typedef Action::MultiStmtArg MultiStmtArg;

  /// Adorns a ExprResult with Actions to make it an OwningExprResult
  OwningExprResult Owned(ExprResult res) {
    return OwningExprResult(Actions, res);
  }
  /// Adorns a StmtResult with Actions to make it an OwningStmtResult
  OwningStmtResult Owned(StmtResult res) {
    return OwningStmtResult(Actions, res);
  }

  OwningExprResult ExprError() { return OwningExprResult(Actions, true); }
  OwningStmtResult StmtError() { return OwningStmtResult(Actions, true); }
  OwningTemplateArgResult TemplateArgError() { 
    return OwningTemplateArgResult(Actions, true); 
  }

  OwningExprResult ExprError(const DiagnosticBuilder &) { return ExprError(); }
  OwningStmtResult StmtError(const DiagnosticBuilder &) { return StmtError(); }
  OwningTemplateArgResult TemplateArgError(const DiagnosticBuilder &) {
    return TemplateArgError();
  }

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
  /// This does not work with all kinds of tokens: strings and specific other
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

  /// TryAnnotateTypeOrScopeToken - If the current token position is on a
  /// typename (possibly qualified in C++) or a C++ scope specifier not followed
  /// by a typename, TryAnnotateTypeOrScopeToken will replace one or more tokens
  /// with a single annotation token representing the typename or C++ scope
  /// respectively.
  /// This simplifies handling of C++ scope specifiers and allows efficient
  /// backtracking without the need to re-parse and resolve nested-names and
  /// typenames.
  /// It will mainly be called when we expect to treat identifiers as typenames
  /// (if they are typenames). For example, in C we do not expect identifiers
  /// inside expressions to be treated as typenames so it will not be called
  /// for expressions in C.
  ///
  /// This returns true if the token was annotated.
  bool TryAnnotateTypeOrScopeToken(const Token *GlobalQualifier = 0);

  /// TryAnnotateCXXScopeToken - Like TryAnnotateTypeOrScopeToken but only
  /// annotates C++ scope specifiers.  This returns true if the token was
  /// annotated.
  bool TryAnnotateCXXScopeToken();

  /// TentativeParsingAction - An object that is used as a kind of "tentative
  /// parsing transaction". It gets instantiated to mark the token position and
  /// after the token consumption is done, Commit() or Revert() is called to
  /// either "commit the consumed tokens" or revert to the previously marked
  /// token position. Example:
  ///
  ///   TentativeParsingAction TPA;
  ///   ConsumeToken();
  ///   ....
  ///   TPA.Revert();
  ///
  class TentativeParsingAction {
    Parser &P;
    Token PrevTok;
    bool isActive;

  public:
    explicit TentativeParsingAction(Parser& p) : P(p) {
      PrevTok = P.Tok;
      P.PP.EnableBacktrackAtThisPos();
      isActive = true;
    }
    void Commit() {
      assert(isActive && "Parsing action was finished!");
      P.PP.CommitBacktrackedTokens();
      isActive = false;
    }
    void Revert() {
      assert(isActive && "Parsing action was finished!");
      P.PP.Backtrack();
      P.Tok = PrevTok;
      isActive = false;
    }
    ~TentativeParsingAction() {
      assert(!isActive && "Forgot to call Commit or Revert!");
    }
  };
  
  
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
  
  /// ParseScope - Introduces a new scope for parsing. The kind of
  /// scope is determined by ScopeFlags. Objects of this type should
  /// be created on the stack to coincide with the position where the
  /// parser enters the new scope, and this object's constructor will
  /// create that new scope. Similarly, once the object is destroyed
  /// the parser will exit the scope.
  class ParseScope {
    Parser *Self;
    ParseScope(const ParseScope&); // do not implement
    ParseScope& operator=(const ParseScope&); // do not implement

  public:
    // ParseScope - Construct a new object to manage a scope in the
    // parser Self where the new Scope is created with the flags
    // ScopeFlags, but only when ManageScope is true (the default). If
    // ManageScope is false, this object does nothing.
    ParseScope(Parser *Self, unsigned ScopeFlags, bool ManageScope = true) 
      : Self(Self) {
      if (ManageScope)
        Self->EnterScope(ScopeFlags);
      else
        this->Self = 0;
    }

    // Exit - Exit the scope associated with this object now, rather
    // than waiting until the object is destroyed.
    void Exit() {
      if (Self) {
        Self->ExitScope();
        Self = 0;
      }
    }

    ~ParseScope() {
      Exit();
    }
  };

  /// EnterScope - Start a new scope.
  void EnterScope(unsigned ScopeFlags);
  
  /// ExitScope - Pop a scope off the scope stack.
  void ExitScope();

  //===--------------------------------------------------------------------===//
  // Diagnostic Emission and Error recovery.

  DiagnosticBuilder Diag(SourceLocation Loc, unsigned DiagID);
  DiagnosticBuilder Diag(const Token &Tok, unsigned DiagID);

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

  //===--------------------------------------------------------------------===//
  // Lexing and parsing of C++ inline methods.

  struct LexedMethod {
    Action::DeclTy *D;
    CachedTokens Toks;
    explicit LexedMethod(Action::DeclTy *MD) : D(MD) {}
  };

  /// LateParsedDefaultArgument - Keeps track of a parameter that may
  /// have a default argument that cannot be parsed yet because it
  /// occurs within a member function declaration inside the class
  /// (C++ [class.mem]p2).
  struct LateParsedDefaultArgument {
    explicit LateParsedDefaultArgument(Action::DeclTy *P, 
                                       CachedTokens *Toks = 0)
      : Param(P), Toks(Toks) { }

    /// Param - The parameter declaration for this parameter.
    Action::DeclTy *Param;

    /// Toks - The sequence of tokens that comprises the default
    /// argument expression, not including the '=' or the terminating
    /// ')' or ','. This will be NULL for parameters that have no
    /// default argument.
    CachedTokens *Toks;
  };
  
  /// LateParsedMethodDeclaration - A method declaration inside a class that
  /// contains at least one entity whose parsing needs to be delayed
  /// until the class itself is completely-defined, such as a default
  /// argument (C++ [class.mem]p2).
  struct LateParsedMethodDeclaration {
    explicit LateParsedMethodDeclaration(Action::DeclTy *M) : Method(M) { }

    /// Method - The method declaration.
    Action::DeclTy *Method;

    /// DefaultArgs - Contains the parameters of the function and
    /// their default arguments. At least one of the parameters will
    /// have a default argument, but all of the parameters of the
    /// method will be stored so that they can be reintroduced into
    /// scope at the appropriate times. 
    llvm::SmallVector<LateParsedDefaultArgument, 8> DefaultArgs;
  };

  /// LateParsedMethodDecls - During parsing of a top (non-nested) C++
  /// class, its method declarations that contain parts that won't be
  /// parsed until after the definiton is completed (C++ [class.mem]p2),
  /// the method declarations will be stored here with the tokens that
  /// will be parsed to create those entities.
  typedef std::list<LateParsedMethodDeclaration> LateParsedMethodDecls;

  /// LexedMethodsForTopClass - During parsing of a top (non-nested) C++ class,
  /// its inline method definitions and the inline method definitions of its
  /// nested classes are lexed and stored here.
  typedef std::list<LexedMethod> LexedMethodsForTopClass;


  /// TopClass - Contains information about parts of the top
  /// (non-nested) C++ class that will need to be parsed after the
  /// class is fully defined.
  struct TopClass {
    /// MethodDecls - Method declarations that contain pieces whose
    /// parsing will be delayed until the class is fully defined.
    LateParsedMethodDecls MethodDecls;

    /// MethodDefs - Methods whose definitions will be parsed once the
    /// class has been fully defined.
    LexedMethodsForTopClass MethodDefs;
  };

  /// TopClassStacks - This is initialized with one TopClass used
  /// for lexing all top classes, until a local class in an inline method is
  /// encountered, at which point a new TopClass is pushed here
  /// and used until the parsing of that local class is finished.
  std::stack<TopClass> TopClassStacks;

  TopClass &getCurTopClassStack() {
    assert(!TopClassStacks.empty() && "No lexed method stacks!");
    return TopClassStacks.top();
  }

  void PushTopClassStack() {
    TopClassStacks.push(TopClass());
  }
  void PopTopClassStack() { TopClassStacks.pop(); }

  DeclTy *ParseCXXInlineMethodDef(AccessSpecifier AS, Declarator &D);
  void ParseLexedMethodDeclarations();
  void ParseLexedMethodDefs();
  bool ConsumeAndStoreUntil(tok::TokenKind T1, tok::TokenKind T2, 
                            CachedTokens &Toks,
                            tok::TokenKind EarlyAbortIf = tok::unknown,
                            bool ConsumeFinalToken = true);

  //===--------------------------------------------------------------------===//
  // C99 6.9: External Definitions.
  DeclTy *ParseExternalDeclaration();
  DeclTy *ParseDeclarationOrFunctionDefinition(
            TemplateParameterLists *TemplateParams = 0);
  DeclTy *ParseFunctionDefinition(Declarator &D);
  void ParseKNRParamDeclarations(Declarator &D);
  OwningExprResult ParseSimpleAsm();
  OwningExprResult ParseAsmStringLiteral();

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
  DeclTy *ParseObjCAtProtocolDeclaration(SourceLocation atLoc,
                                         AttributeList *prefixAttrs = 0);
  
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

  OwningExprResult ParseExpression();
  OwningExprResult ParseConstantExpression();
  // Expr that doesn't include commas.
  OwningExprResult ParseAssignmentExpression();

  OwningExprResult ParseExpressionWithLeadingAt(SourceLocation AtLoc);

  OwningExprResult ParseRHSOfBinaryExpression(OwningExprResult LHS,
                                              unsigned MinPrec);
  OwningExprResult ParseCastExpression(bool isUnaryExpression);
  OwningExprResult ParsePostfixExpressionSuffix(OwningExprResult LHS);
  OwningExprResult ParseSizeofAlignofExpression();
  OwningExprResult ParseBuiltinPrimaryExpression();

  static const unsigned ExprListSize = 12;
  typedef llvm::SmallVector<ExprTy*, ExprListSize> ExprListTy;
  typedef llvm::SmallVector<SourceLocation, ExprListSize> CommaLocsTy;

  /// ParseExpressionList - Used for C/C++ (argument-)expression-list.
  bool ParseExpressionList(ExprListTy &Exprs, CommaLocsTy &CommaLocs);

  /// ParenParseOption - Control what ParseParenExpression will parse.
  enum ParenParseOption {
    SimpleExpr,      // Only parse '(' expression ')'
    CompoundStmt,    // Also allow '(' compound-statement ')'
    CompoundLiteral, // Also allow '(' type-name ')' '{' ... '}'
    CastExpr         // Also allow '(' type-name ')' <anything>
  };
  OwningExprResult ParseParenExpression(ParenParseOption &ExprType,
                                        TypeTy *&CastTy,
                                        SourceLocation &RParenLoc);

  OwningExprResult ParseSimpleParenExpression() {  // Parse SimpleExpr only.
    SourceLocation RParenLoc;
    return ParseSimpleParenExpression(RParenLoc);
  }
  OwningExprResult ParseSimpleParenExpression(SourceLocation &RParenLoc) {
    ParenParseOption Op = SimpleExpr;
    TypeTy *CastTy;
    return ParseParenExpression(Op, CastTy, RParenLoc);
  }
  
  OwningExprResult ParseStringLiteralExpression();

  //===--------------------------------------------------------------------===//
  // C++ Expressions
  OwningExprResult ParseCXXIdExpression();

  /// MaybeParseCXXScopeSpecifier - Parse global scope or nested-name-specifier.
  /// Returns true if a nested-name-specifier was parsed from the token stream.
  ///
  /// If GlobalQualifier is non-null, then it is a :: token we should use as the
  /// global qualifier.
  bool MaybeParseCXXScopeSpecifier(CXXScopeSpec &SS,
                                   const Token *GlobalQualifier = 0);
  
  //===--------------------------------------------------------------------===//
  // C++ 5.2p1: C++ Casts
  OwningExprResult ParseCXXCasts();

  //===--------------------------------------------------------------------===//
  // C++ 5.2p1: C++ Type Identification
  OwningExprResult ParseCXXTypeid();

  //===--------------------------------------------------------------------===//
  // C++ 9.3.2: C++ 'this' pointer
  OwningExprResult ParseCXXThis();

  //===--------------------------------------------------------------------===//
  // C++ 15: C++ Throw Expression
  OwningExprResult ParseThrowExpression();
  bool ParseExceptionSpecification();

  //===--------------------------------------------------------------------===//
  // C++ 2.13.5: C++ Boolean Literals
  OwningExprResult ParseCXXBoolLiteral();

  //===--------------------------------------------------------------------===//
  // C++ 5.2.3: Explicit type conversion (functional notation)
  OwningExprResult ParseCXXTypeConstructExpression(const DeclSpec &DS);

  /// ParseCXXSimpleTypeSpecifier - [C++ 7.1.5.2] Simple type specifiers.
  /// This should only be called when the current token is known to be part of
  /// simple-type-specifier.
  void ParseCXXSimpleTypeSpecifier(DeclSpec &DS);

  bool ParseCXXTypeSpecifierSeq(DeclSpec &DS);

  //===--------------------------------------------------------------------===//
  // C++ 5.3.4 and 5.3.5: C++ new and delete
  bool ParseExpressionListOrTypeId(ExprListTy &Exprs, Declarator &D);
  void ParseDirectNewDeclarator(Declarator &D);
  OwningExprResult ParseCXXNewExpression(bool UseGlobal, SourceLocation Start);
  OwningExprResult ParseCXXDeleteExpression(bool UseGlobal,
                                            SourceLocation Start);

  //===--------------------------------------------------------------------===//
  // C++ if/switch/while/for condition expression.
  OwningExprResult ParseCXXCondition();

  //===--------------------------------------------------------------------===//
  // C++ types

  //===--------------------------------------------------------------------===//
  // C99 6.7.8: Initialization.
  
  /// ParseInitializer
  ///       initializer: [C99 6.7.8]
  ///         assignment-expression
  ///         '{' ...
  OwningExprResult ParseInitializer() {
    if (Tok.isNot(tok::l_brace))
      return ParseAssignmentExpression();
    return ParseBraceInitializer();
  }
  OwningExprResult ParseBraceInitializer();
  OwningExprResult ParseInitializerWithPotentialDesignator(
                       InitListDesignations &D, unsigned InitNum);

  //===--------------------------------------------------------------------===//
  // clang Expressions

  OwningExprResult ParseBlockLiteralExpression();  // ^{...}

  //===--------------------------------------------------------------------===//
  // Objective-C Expressions
  
  bool isTokObjCMessageIdentifierReceiver() const {
    if (!Tok.is(tok::identifier))
      return false;
    
    if (Actions.isTypeName(*Tok.getIdentifierInfo(), CurScope))
      return true;
    
    return Tok.getIdentifierInfo() == Ident_super;
  }

  OwningExprResult ParseObjCAtExpression(SourceLocation AtLocation);
  OwningExprResult ParseObjCStringLiteral(SourceLocation AtLoc);
  OwningExprResult ParseObjCEncodeExpression(SourceLocation AtLoc);
  OwningExprResult ParseObjCSelectorExpression(SourceLocation AtLoc);
  OwningExprResult ParseObjCProtocolExpression(SourceLocation AtLoc);
  OwningExprResult ParseObjCMessageExpression();
  OwningExprResult ParseObjCMessageExpressionBody(SourceLocation LBracloc,
                                                  SourceLocation NameLoc,
                                                  IdentifierInfo *ReceiverName,
                                                  ExprArg ReceiverExpr);
  OwningExprResult ParseAssignmentExprWithObjCMessageExprStart(
      SourceLocation LBracloc, SourceLocation NameLoc,
      IdentifierInfo *ReceiverName, ExprArg ReceiverExpr);

  //===--------------------------------------------------------------------===//
  // C99 6.8: Statements and Blocks.

  OwningStmtResult ParseStatement() {
    return ParseStatementOrDeclaration(true);
  }
  OwningStmtResult ParseStatementOrDeclaration(bool OnlyStatement = false);
  OwningStmtResult ParseLabeledStatement();
  OwningStmtResult ParseCaseStatement();
  OwningStmtResult ParseDefaultStatement();
  OwningStmtResult ParseCompoundStatement(bool isStmtExpr = false);
  OwningStmtResult ParseCompoundStatementBody(bool isStmtExpr = false);
  bool ParseParenExprOrCondition(OwningExprResult &CondExp,
                                 bool OnlyAllowCondition = false);
  OwningStmtResult ParseIfStatement();
  OwningStmtResult ParseSwitchStatement();
  OwningStmtResult ParseWhileStatement();
  OwningStmtResult ParseDoStatement();
  OwningStmtResult ParseForStatement();
  OwningStmtResult ParseGotoStatement();
  OwningStmtResult ParseContinueStatement();
  OwningStmtResult ParseBreakStatement();
  OwningStmtResult ParseReturnStatement();
  OwningStmtResult ParseAsmStatement(bool &msAsm);
  OwningStmtResult FuzzyParseMicrosoftAsmStatement();
  bool ParseAsmOperandsOpt(llvm::SmallVectorImpl<std::string> &Names,
                           llvm::SmallVectorImpl<ExprTy*> &Constraints,
                           llvm::SmallVectorImpl<ExprTy*> &Exprs);

  //===--------------------------------------------------------------------===//
  // C++ 6: Statements and Blocks

  OwningStmtResult ParseCXXTryBlock();
  OwningStmtResult ParseCXXCatchBlock();

  //===--------------------------------------------------------------------===//
  // Objective-C Statements

  OwningStmtResult ParseObjCAtStatement(SourceLocation atLoc);
  OwningStmtResult ParseObjCTryStmt(SourceLocation atLoc);
  OwningStmtResult ParseObjCThrowStmt(SourceLocation atLoc);
  OwningStmtResult ParseObjCSynchronizedStmt(SourceLocation atLoc);


  //===--------------------------------------------------------------------===//
  // C99 6.7: Declarations.
  
  DeclTy *ParseDeclaration(unsigned Context);
  DeclTy *ParseSimpleDeclaration(unsigned Context);
  DeclTy *ParseInitDeclaratorListAfterFirstDeclarator(Declarator &D);
  DeclTy *ParseFunctionStatementBody(DeclTy *Decl, 
                                     SourceLocation L, SourceLocation R);
  void ParseDeclarationSpecifiers(DeclSpec &DS, 
                                  TemplateParameterLists *TemplateParams = 0);
  bool MaybeParseTypeSpecifier(DeclSpec &DS, int &isInvalid, 
                               const char *&PrevSpec,
                               TemplateParameterLists *TemplateParams = 0);
  void ParseSpecifierQualifierList(DeclSpec &DS);
  
  void ParseObjCTypeQualifierList(ObjCDeclSpec &DS);

  void ParseEnumSpecifier(DeclSpec &DS);
  void ParseEnumBody(SourceLocation StartLoc, DeclTy *TagDecl);
  void ParseStructUnionBody(SourceLocation StartLoc, unsigned TagType,
                            DeclTy *TagDecl);
  void ParseStructDeclaration(DeclSpec &DS,
                              llvm::SmallVectorImpl<FieldDeclarator> &Fields);
                              
  bool isDeclarationSpecifier();
  bool isTypeSpecifierQualifier();
  bool isTypeQualifier() const;

  /// isDeclarationStatement - Disambiguates between a declaration or an
  /// expression statement, when parsing function bodies.
  /// Returns true for declaration, false for expression.
  bool isDeclarationStatement() {
    if (getLang().CPlusPlus)
      return isCXXDeclarationStatement();
    return isDeclarationSpecifier();
  }

  /// isSimpleDeclaration - Disambiguates between a declaration or an
  /// expression, mainly used for the C 'clause-1' or the C++
  // 'for-init-statement' part of a 'for' statement.
  /// Returns true for declaration, false for expression.
  bool isSimpleDeclaration() {
    if (getLang().CPlusPlus)
      return isCXXSimpleDeclaration();
    return isDeclarationSpecifier();
  }

  /// isTypeIdInParens - Assumes that a '(' was parsed and now we want to know
  /// whether the parens contain an expression or a type-id.
  /// Returns true for a type-id and false for an expression.
  bool isTypeIdInParens() {
    if (getLang().CPlusPlus)
      return isCXXTypeIdInParens();
    return isTypeSpecifierQualifier();
  }

  /// isCXXDeclarationStatement - C++-specialized function that disambiguates
  /// between a declaration or an expression statement, when parsing function
  /// bodies. Returns true for declaration, false for expression.
  bool isCXXDeclarationStatement();

  /// isCXXSimpleDeclaration - C++-specialized function that disambiguates
  /// between a simple-declaration or an expression-statement.
  /// If during the disambiguation process a parsing error is encountered,
  /// the function returns true to let the declaration parsing code handle it.
  /// Returns false if the statement is disambiguated as expression.
  bool isCXXSimpleDeclaration();

  /// isCXXFunctionDeclarator - Disambiguates between a function declarator or
  /// a constructor-style initializer, when parsing declaration statements.
  /// Returns true for function declarator and false for constructor-style
  /// initializer. If 'warnIfAmbiguous' is true a warning will be emitted to
  /// indicate that the parens were disambiguated as function declarator.
  /// If during the disambiguation process a parsing error is encountered,
  /// the function returns true to let the declaration parsing code handle it.
  bool isCXXFunctionDeclarator(bool warnIfAmbiguous);

  /// isCXXConditionDeclaration - Disambiguates between a declaration or an
  /// expression for a condition of a if/switch/while/for statement.
  /// If during the disambiguation process a parsing error is encountered,
  /// the function returns true to let the declaration parsing code handle it.
  bool isCXXConditionDeclaration();

  /// isCXXTypeIdInParens - Assumes that a '(' was parsed and now we want to
  /// know whether the parens contain an expression or a type-id.
  /// Returns true for a type-id and false for an expression.
  /// If during the disambiguation process a parsing error is encountered,
  /// the function returns true to let the declaration parsing code handle it.
  bool isCXXTypeIdInParens();

  /// TPResult - Used as the result value for functions whose purpose is to
  /// disambiguate C++ constructs by "tentatively parsing" them.
  /// This is a class instead of a simple enum because the implicit enum-to-bool
  /// conversions may cause subtle bugs.
  class TPResult {
    enum Result {
      TPR_true,
      TPR_false,
      TPR_ambiguous,
      TPR_error
    };
    Result Res;
    TPResult(Result result) : Res(result) {}
  public:
    static TPResult True() { return TPR_true; }
    static TPResult False() { return TPR_false; }
    static TPResult Ambiguous() { return TPR_ambiguous; }
    static TPResult Error() { return TPR_error; }

    bool operator==(const TPResult &RHS) const { return Res == RHS.Res; }
    bool operator!=(const TPResult &RHS) const { return Res != RHS.Res; }
  };

  /// isCXXDeclarationSpecifier - Returns TPResult::True() if it is a
  /// declaration specifier, TPResult::False() if it is not,
  /// TPResult::Ambiguous() if it could be either a decl-specifier or a
  /// function-style cast, and TPResult::Error() if a parsing error was
  /// encountered.
  /// Doesn't consume tokens.
  TPResult isCXXDeclarationSpecifier();

  // "Tentative parsing" functions, used for disambiguation. If a parsing error
  // is encountered they will return TPResult::Error().
  // Returning TPResult::True()/False() indicates that the ambiguity was
  // resolved and tentative parsing may stop. TPResult::Ambiguous() indicates
  // that more tentative parsing is necessary for disambiguation.
  // They all consume tokens, so backtracking should be used after calling them.

  TPResult TryParseDeclarationSpecifier();
  TPResult TryParseSimpleDeclaration();
  TPResult TryParseTypeofSpecifier();
  TPResult TryParseInitDeclaratorList();
  TPResult TryParseDeclarator(bool mayBeAbstract, bool mayHaveIdentifier=true);
  TPResult TryParseParameterDeclarationClause();
  TPResult TryParseFunctionDeclarator();
  TPResult TryParseBracketDeclarator();


  TypeTy *ParseTypeName();
  AttributeList *ParseAttributes();
  void FuzzyParseMicrosoftDeclSpec();
  void ParseTypeofSpecifier(DeclSpec &DS);

  /// DeclaratorScopeObj - RAII object used in Parser::ParseDirectDeclarator to
  /// enter a new C++ declarator scope and exit it when the function is
  /// finished.
  class DeclaratorScopeObj {
    Parser &P;
    CXXScopeSpec &SS;
  public:
    DeclaratorScopeObj(Parser &p, CXXScopeSpec &ss) : P(p), SS(ss) {}

    void EnterDeclaratorScope() {
      if (SS.isSet())
        P.Actions.ActOnCXXEnterDeclaratorScope(P.CurScope, SS);
    }

    ~DeclaratorScopeObj() {
      if (SS.isSet())
        P.Actions.ActOnCXXExitDeclaratorScope(P.CurScope, SS);
    }
  };
  
  /// ParseDeclarator - Parse and verify a newly-initialized declarator.
  void ParseDeclarator(Declarator &D);
  /// A function that parses a variant of direct-declarator.
  typedef void (Parser::*DirectDeclParseFunction)(Declarator&);
  void ParseDeclaratorInternal(Declarator &D,
                               DirectDeclParseFunction DirectDeclParser);
  void ParseTypeQualifierListOpt(DeclSpec &DS, bool AttributesAllowed = true);
  void ParseDirectDeclarator(Declarator &D);
  void ParseParenDeclarator(Declarator &D);
  void ParseFunctionDeclarator(SourceLocation LParenLoc, Declarator &D,
                               AttributeList *AttrList = 0,
                               bool RequiresArg = false);
  void ParseFunctionDeclaratorIdentifierList(SourceLocation LParenLoc,
                                             Declarator &D);
  void ParseBracketDeclarator(Declarator &D);
  
  //===--------------------------------------------------------------------===//
  // C++ 7: Declarations [dcl.dcl]
  
  DeclTy *ParseNamespace(unsigned Context);
  DeclTy *ParseLinkage(unsigned Context);
  DeclTy *ParseUsingDirectiveOrDeclaration(unsigned Context);
  DeclTy *ParseUsingDirective(unsigned Context, SourceLocation UsingLoc);
  DeclTy *ParseUsingDeclaration(unsigned Context, SourceLocation UsingLoc);

  //===--------------------------------------------------------------------===//
  // C++ 9: classes [class] and C structs/unions.
  TypeTy *ParseClassName(const CXXScopeSpec *SS = 0);
  void ParseClassSpecifier(DeclSpec &DS, 
                           TemplateParameterLists *TemplateParams = 0);
  void ParseCXXMemberSpecification(SourceLocation StartLoc, unsigned TagType,
                                   DeclTy *TagDecl);
  DeclTy *ParseCXXClassMemberDeclaration(AccessSpecifier AS);
  void ParseConstructorInitializer(DeclTy *ConstructorDecl);
  MemInitResult ParseMemInitializer(DeclTy *ConstructorDecl);

  //===--------------------------------------------------------------------===//
  // C++ 10: Derived classes [class.derived]
  void ParseBaseClause(DeclTy *ClassDecl);
  BaseResult ParseBaseSpecifier(DeclTy *ClassDecl);
  AccessSpecifier getAccessSpecifierIfPresent() const;

  //===--------------------------------------------------------------------===//
  // C++ 13.5: Overloaded operators [over.oper]
  OverloadedOperatorKind TryParseOperatorFunctionId();
  TypeTy *ParseConversionFunctionId();

  //===--------------------------------------------------------------------===//
  // C++ 14: Templates [temp]
  typedef llvm::SmallVector<DeclTy *, 4> TemplateParameterList;

  // C++ 14.1: Template Parameters [temp.param]
  DeclTy *ParseTemplateDeclaration(unsigned Context);
  bool ParseTemplateParameters(unsigned Depth, 
                               TemplateParameterList &TemplateParams,
                               SourceLocation &LAngleLoc, 
                               SourceLocation &RAngleLoc);
  bool ParseTemplateParameterList(unsigned Depth,
                                  TemplateParameterList &TemplateParams);
  DeclTy *ParseTemplateParameter(unsigned Depth, unsigned Position);
  DeclTy *ParseTypeParameter(unsigned Depth, unsigned Position);
  DeclTy *ParseTemplateTemplateParameter(unsigned Depth, unsigned Position);
  DeclTy *ParseNonTypeTemplateParameter(unsigned Depth, unsigned Position);
  // C++ 14.3: Template arguments [temp.arg]
  typedef llvm::SmallVector<TemplateArgTy*, 8> TemplateArgList;
  void AnnotateTemplateIdToken(DeclTy *Template, const CXXScopeSpec *SS = 0);
  bool ParseTemplateArgumentList(TemplateArgList &TemplateArgs);
  OwningTemplateArgResult ParseTemplateArgument();
  
};

}  // end namespace clang

#endif
