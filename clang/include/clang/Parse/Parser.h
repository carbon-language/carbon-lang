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
#include "clang/Parse/AccessSpecifier.h"
#include "clang/Parse/Action.h"
#include "clang/Parse/DeclSpec.h"
#include "llvm/ADT/OwningPtr.h"
#include <stack>
#include <list>

namespace clang {
  class AttributeList;
  struct CXX0XAttributeList;
  class PragmaHandler;
  class Scope;
  class DiagnosticBuilder;
  class Parser;
  class PragmaUnusedHandler;
  class ColonProtectionRAIIObject;

/// PrettyStackTraceParserEntry - If a crash happens while the parser is active,
/// an entry is printed for it.
class PrettyStackTraceParserEntry : public llvm::PrettyStackTraceEntry {
  const Parser &P;
public:
  PrettyStackTraceParserEntry(const Parser &p) : P(p) {}
  virtual void print(llvm::raw_ostream &OS) const;
};


/// Parser - This implements a parser for the C family of languages.  After
/// parsing units of the grammar, productions are invoked to handle whatever has
/// been read.
///
class Parser {
  friend class PragmaUnusedHandler;
  friend class ColonProtectionRAIIObject;
  PrettyStackTraceParserEntry CrashInfo;

  Preprocessor &PP;

  /// Tok - The current token we are peeking ahead.  All parsing methods assume
  /// that this is valid.
  Token Tok;

  // PrevTokLocation - The location of the token we previously
  // consumed. This token is used for diagnostics where we expected to
  // see a token following another token (e.g., the ';' at the end of
  // a statement).
  SourceLocation PrevTokLocation;

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

  llvm::OwningPtr<PragmaHandler> PackHandler;
  llvm::OwningPtr<PragmaHandler> UnusedHandler;
  llvm::OwningPtr<PragmaHandler> WeakHandler;
  llvm::OwningPtr<clang::CommentHandler> CommentHandler;

  /// Whether the '>' token acts as an operator or not. This will be
  /// true except when we are parsing an expression within a C++
  /// template argument list, where the '>' closes the template
  /// argument list.
  bool GreaterThanIsOperator;
  
  /// ColonIsSacred - When this is false, we aggressively try to recover from
  /// code like "foo : bar" as if it were a typo for "foo :: bar".  This is not
  /// safe in case statements and a few other things.  This is managed by the
  /// ColonProtectionRAIIObject RAII object.
  bool ColonIsSacred;

  /// The "depth" of the template parameters currently being parsed.
  unsigned TemplateParameterDepth;

public:
  Parser(Preprocessor &PP, Action &Actions);
  ~Parser();

  const LangOptions &getLang() const { return PP.getLangOptions(); }
  const TargetInfo &getTargetInfo() const { return PP.getTargetInfo(); }
  Preprocessor &getPreprocessor() const { return PP; }
  Action &getActions() const { return Actions; }

  const Token &getCurToken() const { return Tok; }

  // Type forwarding.  All of these are statically 'void*', but they may all be
  // different actual classes based on the actions in place.
  typedef Action::ExprTy ExprTy;
  typedef Action::StmtTy StmtTy;
  typedef Action::DeclPtrTy DeclPtrTy;
  typedef Action::DeclGroupPtrTy DeclGroupPtrTy;
  typedef Action::TypeTy TypeTy;
  typedef Action::BaseTy BaseTy;
  typedef Action::MemInitTy MemInitTy;
  typedef Action::CXXScopeTy CXXScopeTy;
  typedef Action::TemplateParamsTy TemplateParamsTy;
  typedef Action::TemplateTy TemplateTy;

  typedef llvm::SmallVector<TemplateParamsTy *, 4> TemplateParameterLists;

  typedef Action::ExprResult        ExprResult;
  typedef Action::StmtResult        StmtResult;
  typedef Action::BaseResult        BaseResult;
  typedef Action::MemInitResult     MemInitResult;
  typedef Action::TypeResult        TypeResult;

  typedef Action::OwningExprResult OwningExprResult;
  typedef Action::OwningStmtResult OwningStmtResult;

  typedef Action::ExprArg ExprArg;
  typedef Action::MultiStmtArg MultiStmtArg;
  typedef Action::FullExprArg FullExprArg;

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

  OwningExprResult ExprError(const DiagnosticBuilder &) { return ExprError(); }
  OwningStmtResult StmtError(const DiagnosticBuilder &) { return StmtError(); }

  OwningExprResult ExprEmpty() { return OwningExprResult(Actions, false); }

  // Parsing methods.

  /// ParseTranslationUnit - All in one method that initializes parses, and
  /// shuts down the parser.
  void ParseTranslationUnit();

  /// Initialize - Warm up the parser.
  ///
  void Initialize();

  /// ParseTopLevelDecl - Parse one top-level declaration. Returns true if
  /// the EOF was encountered.
  bool ParseTopLevelDecl(DeclGroupPtrTy &Result);

  DeclGroupPtrTy RetrievePendingObjCImpDecl();

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
    PrevTokLocation = Tok.getLocation();
    PP.Lex(Tok);
    return PrevTokLocation;
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
    PrevTokLocation = Tok.getLocation();
    PP.Lex(Tok);
    return PrevTokLocation;
  }

  /// ConsumeBracket - This consume method keeps the bracket count up-to-date.
  ///
  SourceLocation ConsumeBracket() {
    assert(isTokenBracket() && "wrong consume method");
    if (Tok.getKind() == tok::l_square)
      ++BracketCount;
    else if (BracketCount)
      --BracketCount;     // Don't let unbalanced ]'s drive the count negative.

    PrevTokLocation = Tok.getLocation();
    PP.Lex(Tok);
    return PrevTokLocation;
  }

  /// ConsumeBrace - This consume method keeps the brace count up-to-date.
  ///
  SourceLocation ConsumeBrace() {
    assert(isTokenBrace() && "wrong consume method");
    if (Tok.getKind() == tok::l_brace)
      ++BraceCount;
    else if (BraceCount)
      --BraceCount;     // Don't let unbalanced }'s drive the count negative.

    PrevTokLocation = Tok.getLocation();
    PP.Lex(Tok);
    return PrevTokLocation;
  }

  /// ConsumeStringToken - Consume the current 'peek token', lexing a new one
  /// and returning the token kind.  This method is specific to strings, as it
  /// handles string literal concatenation, as per C99 5.1.1.2, translation
  /// phase #6.
  SourceLocation ConsumeStringToken() {
    assert(isTokenStringLiteral() &&
           "Should only consume string literals with this method");
    PrevTokLocation = Tok.getLocation();
    PP.Lex(Tok);
    return PrevTokLocation;
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
  bool TryAnnotateTypeOrScopeToken(bool EnteringContext = false);

  /// TryAnnotateCXXScopeToken - Like TryAnnotateTypeOrScopeToken but only
  /// annotates C++ scope specifiers.  This returns true if the token was
  /// annotated.
  bool TryAnnotateCXXScopeToken(bool EnteringContext = false);

  /// TentativeParsingAction - An object that is used as a kind of "tentative
  /// parsing transaction". It gets instantiated to mark the token position and
  /// after the token consumption is done, Commit() or Revert() is called to
  /// either "commit the consumed tokens" or revert to the previously marked
  /// token position. Example:
  ///
  ///   TentativeParsingAction TPA(*this);
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

  void SuggestParentheses(SourceLocation Loc, unsigned DK,
                          SourceRange ParenRange);

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
    Action::DeclPtrTy D;
    CachedTokens Toks;

    /// \brief Whether this member function had an associated template
    /// scope. When true, D is a template declaration.
    /// othewise, it is a member function declaration.
    bool TemplateScope;

    explicit LexedMethod(Action::DeclPtrTy MD) : D(MD), TemplateScope(false) {}
  };

  /// LateParsedDefaultArgument - Keeps track of a parameter that may
  /// have a default argument that cannot be parsed yet because it
  /// occurs within a member function declaration inside the class
  /// (C++ [class.mem]p2).
  struct LateParsedDefaultArgument {
    explicit LateParsedDefaultArgument(Action::DeclPtrTy P,
                                       CachedTokens *Toks = 0)
      : Param(P), Toks(Toks) { }

    /// Param - The parameter declaration for this parameter.
    Action::DeclPtrTy Param;

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
    explicit LateParsedMethodDeclaration(Action::DeclPtrTy M)
      : Method(M), TemplateScope(false) { }

    /// Method - The method declaration.
    Action::DeclPtrTy Method;

    /// \brief Whether this member function had an associated template
    /// scope. When true, D is a template declaration.
    /// othewise, it is a member function declaration.
    bool TemplateScope;

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

  /// \brief Representation of a class that has been parsed, including
  /// any member function declarations or definitions that need to be
  /// parsed after the corresponding top-level class is complete.
  struct ParsingClass {
    ParsingClass(DeclPtrTy TagOrTemplate, bool TopLevelClass)
      : TopLevelClass(TopLevelClass), TemplateScope(false),
        TagOrTemplate(TagOrTemplate) { }

    /// \brief Whether this is a "top-level" class, meaning that it is
    /// not nested within another class.
    bool TopLevelClass : 1;

    /// \brief Whether this class had an associated template
    /// scope. When true, TagOrTemplate is a template declaration;
    /// othewise, it is a tag declaration.
    bool TemplateScope : 1;

    /// \brief The class or class template whose definition we are parsing.
    DeclPtrTy TagOrTemplate;

    /// MethodDecls - Method declarations that contain pieces whose
    /// parsing will be delayed until the class is fully defined.
    LateParsedMethodDecls MethodDecls;

    /// MethodDefs - Methods whose definitions will be parsed once the
    /// class has been fully defined.
    LexedMethodsForTopClass MethodDefs;

    /// \brief Nested classes inside this class.
    llvm::SmallVector<ParsingClass*, 4> NestedClasses;
  };

  /// \brief The stack of classes that is currently being
  /// parsed. Nested and local classes will be pushed onto this stack
  /// when they are parsed, and removed afterward.
  std::stack<ParsingClass *> ClassStack;

  ParsingClass &getCurrentClass() {
    assert(!ClassStack.empty() && "No lexed method stacks!");
    return *ClassStack.top();
  }

  /// \brief RAII object used to inform the actions that we're
  /// currently parsing a declaration.  This is active when parsing a
  /// variable's initializer, but not when parsing the body of a
  /// class or function definition.
  class ParsingDeclRAIIObject {
    Action &Actions;
    Action::ParsingDeclStackState State;
    bool Popped;
    
  public:
    ParsingDeclRAIIObject(Parser &P) : Actions(P.Actions) {
      push();
    }

    ~ParsingDeclRAIIObject() {
      abort();
    }

    /// Resets the RAII object for a new declaration.
    void reset() {
      abort();
      push();
    }

    /// Signals that the context was completed without an appropriate
    /// declaration being parsed.
    void abort() {
      pop(DeclPtrTy());
    }

    void complete(DeclPtrTy D) {
      assert(!Popped && "ParsingDeclaration has already been popped!");
      pop(D);
    }

  private:
    void push() {
      State = Actions.PushParsingDeclaration();
      Popped = false;
    }

    void pop(DeclPtrTy D) {
      if (!Popped) {
        Actions.PopParsingDeclaration(State, D);
        Popped = true;
      }
    }
  };

  /// A class for parsing a DeclSpec.
  class ParsingDeclSpec : public DeclSpec {
    ParsingDeclRAIIObject ParsingRAII;

  public:
    ParsingDeclSpec(Parser &P) : ParsingRAII(P) {
    }

    void complete(DeclPtrTy D) {
      ParsingRAII.complete(D);
    }

    void abort() {
      ParsingRAII.abort();
    }
  };

  /// A class for parsing a declarator.
  class ParsingDeclarator : public Declarator {
    ParsingDeclRAIIObject ParsingRAII;

  public:
    ParsingDeclarator(Parser &P, const ParsingDeclSpec &DS, TheContext C)
      : Declarator(DS, C), ParsingRAII(P) {
    }

    const ParsingDeclSpec &getDeclSpec() const {
      return static_cast<const ParsingDeclSpec&>(Declarator::getDeclSpec());
    }

    ParsingDeclSpec &getMutableDeclSpec() const {
      return const_cast<ParsingDeclSpec&>(getDeclSpec());
    }

    void clear() {
      Declarator::clear();
      ParsingRAII.reset();
    }

    void complete(DeclPtrTy D) {
      ParsingRAII.complete(D);
    }
  };

  /// \brief RAII object used to
  class ParsingClassDefinition {
    Parser &P;
    bool Popped;

  public:
    ParsingClassDefinition(Parser &P, DeclPtrTy TagOrTemplate, bool TopLevelClass)
      : P(P), Popped(false) {
      P.PushParsingClass(TagOrTemplate, TopLevelClass);
    }

    /// \brief Pop this class of the stack.
    void Pop() {
      assert(!Popped && "Nested class has already been popped");
      Popped = true;
      P.PopParsingClass();
    }

    ~ParsingClassDefinition() {
      if (!Popped)
        P.PopParsingClass();
    }
  };

  /// \brief Contains information about any template-specific
  /// information that has been parsed prior to parsing declaration
  /// specifiers.
  struct ParsedTemplateInfo {
    ParsedTemplateInfo()
      : Kind(NonTemplate), TemplateParams(0), TemplateLoc() { }

    ParsedTemplateInfo(TemplateParameterLists *TemplateParams,
                       bool isSpecialization,
                       bool lastParameterListWasEmpty = false)
      : Kind(isSpecialization? ExplicitSpecialization : Template),
        TemplateParams(TemplateParams), 
        LastParameterListWasEmpty(lastParameterListWasEmpty) { }

    explicit ParsedTemplateInfo(SourceLocation ExternLoc,
                                SourceLocation TemplateLoc)
      : Kind(ExplicitInstantiation), TemplateParams(0),
        ExternLoc(ExternLoc), TemplateLoc(TemplateLoc),
        LastParameterListWasEmpty(false){ }

    /// \brief The kind of template we are parsing.
    enum {
      /// \brief We are not parsing a template at all.
      NonTemplate = 0,
      /// \brief We are parsing a template declaration.
      Template,
      /// \brief We are parsing an explicit specialization.
      ExplicitSpecialization,
      /// \brief We are parsing an explicit instantiation.
      ExplicitInstantiation
    } Kind;

    /// \brief The template parameter lists, for template declarations
    /// and explicit specializations.
    TemplateParameterLists *TemplateParams;

    /// \brief The location of the 'extern' keyword, if any, for an explicit
    /// instantiation
    SourceLocation ExternLoc;

    /// \brief The location of the 'template' keyword, for an explicit
    /// instantiation.
    SourceLocation TemplateLoc;
    
    /// \brief Whether the last template parameter list was empty.
    bool LastParameterListWasEmpty;
  };

  void PushParsingClass(DeclPtrTy TagOrTemplate, bool TopLevelClass);
  void DeallocateParsedClasses(ParsingClass *Class);
  void PopParsingClass();

  DeclPtrTy ParseCXXInlineMethodDef(AccessSpecifier AS, Declarator &D,
                                    const ParsedTemplateInfo &TemplateInfo);
  void ParseLexedMethodDeclarations(ParsingClass &Class);
  void ParseLexedMethodDefs(ParsingClass &Class);
  bool ConsumeAndStoreUntil(tok::TokenKind T1, tok::TokenKind T2,
                            CachedTokens &Toks,
                            tok::TokenKind EarlyAbortIf = tok::unknown,
                            bool ConsumeFinalToken = true);

  //===--------------------------------------------------------------------===//
  // C99 6.9: External Definitions.
  DeclGroupPtrTy ParseExternalDeclaration(CXX0XAttributeList Attr);
  bool isDeclarationAfterDeclarator();
  bool isStartOfFunctionDefinition();
  DeclGroupPtrTy ParseDeclarationOrFunctionDefinition(AttributeList *Attr,
            AccessSpecifier AS = AS_none);
  DeclGroupPtrTy ParseDeclarationOrFunctionDefinition(ParsingDeclSpec &DS,
                                                  AttributeList *Attr,
                                                  AccessSpecifier AS = AS_none);
  
  DeclPtrTy ParseFunctionDefinition(ParsingDeclarator &D,
                 const ParsedTemplateInfo &TemplateInfo = ParsedTemplateInfo());
  void ParseKNRParamDeclarations(Declarator &D);
  // EndLoc, if non-NULL, is filled with the location of the last token of
  // the simple-asm.
  OwningExprResult ParseSimpleAsm(SourceLocation *EndLoc = 0);
  OwningExprResult ParseAsmStringLiteral();

  // Objective-C External Declarations
  DeclPtrTy ParseObjCAtDirectives();
  DeclPtrTy ParseObjCAtClassDeclaration(SourceLocation atLoc);
  DeclPtrTy ParseObjCAtInterfaceDeclaration(SourceLocation atLoc,
                                          AttributeList *prefixAttrs = 0);
  void ParseObjCClassInstanceVariables(DeclPtrTy interfaceDecl,
                                       SourceLocation atLoc);
  bool ParseObjCProtocolReferences(llvm::SmallVectorImpl<Action::DeclPtrTy> &P,
                                   llvm::SmallVectorImpl<SourceLocation> &PLocs,
                                   bool WarnOnDeclarations,
                                   SourceLocation &LAngleLoc,
                                   SourceLocation &EndProtoLoc);
  void ParseObjCInterfaceDeclList(DeclPtrTy interfaceDecl,
                                  tok::ObjCKeywordKind contextKey);
  DeclPtrTy ParseObjCAtProtocolDeclaration(SourceLocation atLoc,
                                           AttributeList *prefixAttrs = 0);

  DeclPtrTy ObjCImpDecl;
  llvm::SmallVector<DeclPtrTy, 4> PendingObjCImpDecl;

  DeclPtrTy ParseObjCAtImplementationDeclaration(SourceLocation atLoc);
  DeclPtrTy ParseObjCAtEndDeclaration(SourceLocation atLoc);
  DeclPtrTy ParseObjCAtAliasDeclaration(SourceLocation atLoc);
  DeclPtrTy ParseObjCPropertySynthesize(SourceLocation atLoc);
  DeclPtrTy ParseObjCPropertyDynamic(SourceLocation atLoc);

  IdentifierInfo *ParseObjCSelectorPiece(SourceLocation &MethodLocation);
  // Definitions for Objective-c context sensitive keywords recognition.
  enum ObjCTypeQual {
    objc_in=0, objc_out, objc_inout, objc_oneway, objc_bycopy, objc_byref,
    objc_NumQuals
  };
  IdentifierInfo *ObjCTypeQuals[objc_NumQuals];

  bool isTokIdentifier_in() const;

  TypeTy *ParseObjCTypeName(ObjCDeclSpec &DS);
  void ParseObjCMethodRequirement();
  DeclPtrTy ParseObjCMethodPrototype(DeclPtrTy classOrCat,
            tok::ObjCKeywordKind MethodImplKind = tok::objc_not_keyword);
  DeclPtrTy ParseObjCMethodDecl(SourceLocation mLoc, tok::TokenKind mType,
                                DeclPtrTy classDecl,
            tok::ObjCKeywordKind MethodImplKind = tok::objc_not_keyword);
  void ParseObjCPropertyAttribute(ObjCDeclSpec &DS, DeclPtrTy ClassDecl,
                                  DeclPtrTy *Methods, unsigned NumMethods);

  DeclPtrTy ParseObjCMethodDefinition();

  //===--------------------------------------------------------------------===//
  // C99 6.5: Expressions.

  OwningExprResult ParseExpression();
  OwningExprResult ParseConstantExpression();
  // Expr that doesn't include commas.
  OwningExprResult ParseAssignmentExpression();

  OwningExprResult ParseExpressionWithLeadingAt(SourceLocation AtLoc);

  OwningExprResult ParseExpressionWithLeadingExtension(SourceLocation ExtLoc);

  OwningExprResult ParseRHSOfBinaryExpression(OwningExprResult LHS,
                                              unsigned MinPrec);
  OwningExprResult ParseCastExpression(bool isUnaryExpression,
                                       bool isAddressOfOperand,
                                       bool &NotCastExpr,
                                       TypeTy *TypeOfCast);
  OwningExprResult ParseCastExpression(bool isUnaryExpression,
                                       bool isAddressOfOperand = false,
                                       TypeTy *TypeOfCast = 0);
  OwningExprResult ParsePostfixExpressionSuffix(OwningExprResult LHS);
  OwningExprResult ParseSizeofAlignofExpression();
  OwningExprResult ParseBuiltinPrimaryExpression();

  OwningExprResult ParseExprAfterTypeofSizeofAlignof(const Token &OpTok,
                                                     bool &isCastExpr,
                                                     TypeTy *&CastTy,
                                                     SourceRange &CastRange);

  static const unsigned ExprListSize = 12;
  typedef llvm::SmallVector<ExprTy*, ExprListSize> ExprListTy;
  typedef llvm::SmallVector<SourceLocation, ExprListSize> CommaLocsTy;

  /// ParseExpressionList - Used for C/C++ (argument-)expression-list.
  bool ParseExpressionList(ExprListTy &Exprs, CommaLocsTy &CommaLocs,
                           void (Action::*Completer)(Scope *S, void *Data,
                                                     ExprTy **Args,
                                                     unsigned NumArgs) = 0,
                           void *Data = 0);

  /// ParenParseOption - Control what ParseParenExpression will parse.
  enum ParenParseOption {
    SimpleExpr,      // Only parse '(' expression ')'
    CompoundStmt,    // Also allow '(' compound-statement ')'
    CompoundLiteral, // Also allow '(' type-name ')' '{' ... '}'
    CastExpr         // Also allow '(' type-name ')' <anything>
  };
  OwningExprResult ParseParenExpression(ParenParseOption &ExprType,
                                        bool stopIfCastExpr,
                                        TypeTy *TypeOfCast,
                                        TypeTy *&CastTy,
                                        SourceLocation &RParenLoc);

  OwningExprResult ParseCXXAmbiguousParenExpression(ParenParseOption &ExprType,
                                                    TypeTy *&CastTy,
                                                    SourceLocation LParenLoc,
                                                    SourceLocation &RParenLoc);

  OwningExprResult ParseCompoundLiteralExpression(TypeTy *Ty,
                                                  SourceLocation LParenLoc,
                                                  SourceLocation RParenLoc);

  OwningExprResult ParseStringLiteralExpression();

  //===--------------------------------------------------------------------===//
  // C++ Expressions
  OwningExprResult ParseCXXIdExpression(bool isAddressOfOperand = false);

  bool ParseOptionalCXXScopeSpecifier(CXXScopeSpec &SS,
                                      TypeTy *ObjectType,
                                      bool EnteringContext);

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
  // EndLoc is filled with the location of the last token of the specification.
  bool ParseExceptionSpecification(SourceLocation &EndLoc,
                                   llvm::SmallVector<TypeTy*, 2> &Exceptions,
                                   llvm::SmallVector<SourceRange, 2> &Ranges,
                                   bool &hasAnyExceptionSpec);

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
  // C++ if/switch/while condition expression.
  bool ParseCXXCondition(OwningExprResult &ExprResult, DeclPtrTy &DeclResult);

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
  OwningExprResult ParseInitializerWithPotentialDesignator();

  //===--------------------------------------------------------------------===//
  // clang Expressions

  OwningExprResult ParseBlockLiteralExpression();  // ^{...}

  //===--------------------------------------------------------------------===//
  // Objective-C Expressions

  bool isTokObjCMessageIdentifierReceiver() const {
    if (!Tok.is(tok::identifier))
      return false;

    IdentifierInfo *II = Tok.getIdentifierInfo();
    if (Actions.getTypeName(*II, Tok.getLocation(), CurScope))
      return true;

    return II == Ident_super;
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
  OwningStmtResult ParseLabeledStatement(AttributeList *Attr);
  OwningStmtResult ParseCaseStatement(AttributeList *Attr);
  OwningStmtResult ParseDefaultStatement(AttributeList *Attr);
  OwningStmtResult ParseCompoundStatement(AttributeList *Attr,
                                          bool isStmtExpr = false);
  OwningStmtResult ParseCompoundStatementBody(bool isStmtExpr = false);
  bool ParseParenExprOrCondition(OwningExprResult &ExprResult,
                                 DeclPtrTy &DeclResult);
  OwningStmtResult ParseIfStatement(AttributeList *Attr);
  OwningStmtResult ParseSwitchStatement(AttributeList *Attr);
  OwningStmtResult ParseWhileStatement(AttributeList *Attr);
  OwningStmtResult ParseDoStatement(AttributeList *Attr);
  OwningStmtResult ParseForStatement(AttributeList *Attr);
  OwningStmtResult ParseGotoStatement(AttributeList *Attr);
  OwningStmtResult ParseContinueStatement(AttributeList *Attr);
  OwningStmtResult ParseBreakStatement(AttributeList *Attr);
  OwningStmtResult ParseReturnStatement(AttributeList *Attr);
  OwningStmtResult ParseAsmStatement(bool &msAsm);
  OwningStmtResult FuzzyParseMicrosoftAsmStatement();
  bool ParseAsmOperandsOpt(llvm::SmallVectorImpl<std::string> &Names,
                           llvm::SmallVectorImpl<ExprTy*> &Constraints,
                           llvm::SmallVectorImpl<ExprTy*> &Exprs);

  //===--------------------------------------------------------------------===//
  // C++ 6: Statements and Blocks

  OwningStmtResult ParseCXXTryBlock(AttributeList *Attr);
  OwningStmtResult ParseCXXTryBlockCommon(SourceLocation TryLoc);
  OwningStmtResult ParseCXXCatchBlock();

  //===--------------------------------------------------------------------===//
  // Objective-C Statements

  OwningStmtResult ParseObjCAtStatement(SourceLocation atLoc);
  OwningStmtResult ParseObjCTryStmt(SourceLocation atLoc);
  OwningStmtResult ParseObjCThrowStmt(SourceLocation atLoc);
  OwningStmtResult ParseObjCSynchronizedStmt(SourceLocation atLoc);


  //===--------------------------------------------------------------------===//
  // C99 6.7: Declarations.

  /// A context for parsing declaration specifiers.  TODO: flesh this
  /// out, there are other significant restrictions on specifiers than
  /// would be best implemented in the parser.
  enum DeclSpecContext {
    DSC_normal, // normal context
    DSC_class   // class context, enables 'friend'
  };

  DeclGroupPtrTy ParseDeclaration(unsigned Context, SourceLocation &DeclEnd,
                                  CXX0XAttributeList Attr);
  DeclGroupPtrTy ParseSimpleDeclaration(unsigned Context,
                                        SourceLocation &DeclEnd,
                                        AttributeList *Attr);
  DeclGroupPtrTy ParseDeclGroup(ParsingDeclSpec &DS, unsigned Context,
                                bool AllowFunctionDefinitions,
                                SourceLocation *DeclEnd = 0);
  DeclPtrTy ParseDeclarationAfterDeclarator(Declarator &D,
               const ParsedTemplateInfo &TemplateInfo = ParsedTemplateInfo());
  DeclPtrTy ParseFunctionStatementBody(DeclPtrTy Decl);
  DeclPtrTy ParseFunctionTryBlock(DeclPtrTy Decl);

  bool ParseImplicitInt(DeclSpec &DS, CXXScopeSpec *SS,
                        const ParsedTemplateInfo &TemplateInfo,
                        AccessSpecifier AS);
  void ParseDeclarationSpecifiers(DeclSpec &DS,
                const ParsedTemplateInfo &TemplateInfo = ParsedTemplateInfo(),
                                  AccessSpecifier AS = AS_none,
                                  DeclSpecContext DSC = DSC_normal);
  bool ParseOptionalTypeSpecifier(DeclSpec &DS, bool &isInvalid,
                                  const char *&PrevSpec,
                                  unsigned &DiagID,
               const ParsedTemplateInfo &TemplateInfo = ParsedTemplateInfo());

  void ParseSpecifierQualifierList(DeclSpec &DS);

  void ParseObjCTypeQualifierList(ObjCDeclSpec &DS);

  void ParseEnumSpecifier(SourceLocation TagLoc, DeclSpec &DS,
                          AccessSpecifier AS = AS_none);
  void ParseEnumBody(SourceLocation StartLoc, DeclPtrTy TagDecl);
  void ParseStructUnionBody(SourceLocation StartLoc, unsigned TagType,
                            DeclPtrTy TagDecl);

  struct FieldCallback {
    virtual DeclPtrTy invoke(FieldDeclarator &Field) = 0;
    virtual ~FieldCallback() {}

  private:
    virtual void _anchor();
  };
  struct ObjCPropertyCallback;

  void ParseStructDeclaration(DeclSpec &DS, FieldCallback &Callback);

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

  /// \brief Specifies the context in which type-id/expression
  /// disambiguation will occur.
  enum TentativeCXXTypeIdContext {
    TypeIdInParens,
    TypeIdAsTemplateArgument
  };


  /// isTypeIdInParens - Assumes that a '(' was parsed and now we want to know
  /// whether the parens contain an expression or a type-id.
  /// Returns true for a type-id and false for an expression.
  bool isTypeIdInParens(bool &isAmbiguous) {
    if (getLang().CPlusPlus)
      return isCXXTypeId(TypeIdInParens, isAmbiguous);
    isAmbiguous = false;
    return isTypeSpecifierQualifier();
  }
  bool isTypeIdInParens() {
    bool isAmbiguous;
    return isTypeIdInParens(isAmbiguous);
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

  bool isCXXTypeId(TentativeCXXTypeIdContext Context, bool &isAmbiguous);
  bool isCXXTypeId(TentativeCXXTypeIdContext Context) {
    bool isAmbiguous;
    return isCXXTypeId(Context, isAmbiguous);
  }

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

  TypeResult ParseTypeName(SourceRange *Range = 0);
  void ParseBlockId();
  // EndLoc, if non-NULL, is filled with the location of the last token of
  // the attribute list.
  CXX0XAttributeList ParseCXX0XAttributes(SourceLocation *EndLoc = 0);
  AttributeList *ParseGNUAttributes(SourceLocation *EndLoc = 0);
  AttributeList *ParseMicrosoftDeclSpec(AttributeList* CurrAttr = 0);
  AttributeList *ParseMicrosoftTypeAttributes(AttributeList* CurrAttr = 0);
  void ParseTypeofSpecifier(DeclSpec &DS);
  void ParseDecltypeSpecifier(DeclSpec &DS);
  
  OwningExprResult ParseCXX0XAlignArgument(SourceLocation Start);

  /// DeclaratorScopeObj - RAII object used in Parser::ParseDirectDeclarator to
  /// enter a new C++ declarator scope and exit it when the function is
  /// finished.
  class DeclaratorScopeObj {
    Parser &P;
    CXXScopeSpec &SS;
    bool EnteredScope;
    bool CreatedScope;
  public:
    DeclaratorScopeObj(Parser &p, CXXScopeSpec &ss)
      : P(p), SS(ss), EnteredScope(false), CreatedScope(false) {}

    void EnterDeclaratorScope() {
      assert(!EnteredScope && "Already entered the scope!");
      assert(SS.isSet() && "C++ scope was not set!");

      CreatedScope = true;
      P.EnterScope(0); // Not a decl scope.

      if (P.Actions.ActOnCXXEnterDeclaratorScope(P.CurScope, SS))
        SS.setScopeRep(0);
      
      if (!SS.isInvalid())
        EnteredScope = true;
    }

    ~DeclaratorScopeObj() {
      if (EnteredScope) {
        assert(SS.isSet() && "C++ scope was cleared ?");
        P.Actions.ActOnCXXExitDeclaratorScope(P.CurScope, SS);
      }
      if (CreatedScope)
        P.ExitScope();
    }
  };

  /// ParseDeclarator - Parse and verify a newly-initialized declarator.
  void ParseDeclarator(Declarator &D);
  /// A function that parses a variant of direct-declarator.
  typedef void (Parser::*DirectDeclParseFunction)(Declarator&);
  void ParseDeclaratorInternal(Declarator &D,
                               DirectDeclParseFunction DirectDeclParser);
  void ParseTypeQualifierListOpt(DeclSpec &DS, bool GNUAttributesAllowed = true,
                                 bool CXX0XAttributesAllowed = true);
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

  bool isCXX0XAttributeSpecifier(bool FullLookahead = false, 
                                 tok::TokenKind *After = 0);
  
  DeclPtrTy ParseNamespace(unsigned Context, SourceLocation &DeclEnd);
  DeclPtrTy ParseLinkage(ParsingDeclSpec &DS, unsigned Context);
  DeclPtrTy ParseUsingDirectiveOrDeclaration(unsigned Context,
                                             SourceLocation &DeclEnd,
                                             CXX0XAttributeList Attrs);
  DeclPtrTy ParseUsingDirective(unsigned Context, SourceLocation UsingLoc,
                                SourceLocation &DeclEnd,
                                AttributeList *Attr);
  DeclPtrTy ParseUsingDeclaration(unsigned Context, SourceLocation UsingLoc,
                                  SourceLocation &DeclEnd,
                                  AccessSpecifier AS = AS_none);
  DeclPtrTy ParseStaticAssertDeclaration(SourceLocation &DeclEnd);
  DeclPtrTy ParseNamespaceAlias(SourceLocation NamespaceLoc,
                                SourceLocation AliasLoc, IdentifierInfo *Alias,
                                SourceLocation &DeclEnd);

  //===--------------------------------------------------------------------===//
  // C++ 9: classes [class] and C structs/unions.
  TypeResult ParseClassName(SourceLocation &EndLocation,
                            const CXXScopeSpec *SS = 0,
                            bool DestrExpected = false);
  void ParseClassSpecifier(tok::TokenKind TagTokKind, SourceLocation TagLoc,
                           DeclSpec &DS,
                const ParsedTemplateInfo &TemplateInfo = ParsedTemplateInfo(),
                           AccessSpecifier AS = AS_none);
  void ParseCXXMemberSpecification(SourceLocation StartLoc, unsigned TagType,
                                   DeclPtrTy TagDecl);
  void ParseCXXClassMemberDeclaration(AccessSpecifier AS,
                const ParsedTemplateInfo &TemplateInfo = ParsedTemplateInfo());
  void ParseConstructorInitializer(DeclPtrTy ConstructorDecl);
  MemInitResult ParseMemInitializer(DeclPtrTy ConstructorDecl);
  void HandleMemberFunctionDefaultArgs(Declarator& DeclaratorInfo,
                                       DeclPtrTy ThisDecl);

  //===--------------------------------------------------------------------===//
  // C++ 10: Derived classes [class.derived]
  void ParseBaseClause(DeclPtrTy ClassDecl);
  BaseResult ParseBaseSpecifier(DeclPtrTy ClassDecl);
  AccessSpecifier getAccessSpecifierIfPresent() const;

  bool ParseUnqualifiedIdTemplateId(CXXScopeSpec &SS, 
                                    IdentifierInfo *Name,
                                    SourceLocation NameLoc,
                                    bool EnteringContext,
                                    TypeTy *ObjectType,
                                    UnqualifiedId &Id);
  bool ParseUnqualifiedIdOperator(CXXScopeSpec &SS, bool EnteringContext,
                                  TypeTy *ObjectType,
                                  UnqualifiedId &Result);
  bool ParseUnqualifiedId(CXXScopeSpec &SS, bool EnteringContext,
                          bool AllowDestructorName,
                          bool AllowConstructorName,
                          TypeTy *ObjectType,
                          UnqualifiedId &Result);
    
  //===--------------------------------------------------------------------===//
  // C++ 14: Templates [temp]
  typedef llvm::SmallVector<DeclPtrTy, 4> TemplateParameterList;

  // C++ 14.1: Template Parameters [temp.param]
  DeclPtrTy ParseDeclarationStartingWithTemplate(unsigned Context,
                                                 SourceLocation &DeclEnd,
                                                 AccessSpecifier AS = AS_none);
  DeclPtrTy ParseTemplateDeclarationOrSpecialization(unsigned Context,
                                                     SourceLocation &DeclEnd,
                                                     AccessSpecifier AS);
  DeclPtrTy ParseSingleDeclarationAfterTemplate(
                                       unsigned Context,
                                       const ParsedTemplateInfo &TemplateInfo,
                                       SourceLocation &DeclEnd,
                                       AccessSpecifier AS=AS_none);
  bool ParseTemplateParameters(unsigned Depth,
                               TemplateParameterList &TemplateParams,
                               SourceLocation &LAngleLoc,
                               SourceLocation &RAngleLoc);
  bool ParseTemplateParameterList(unsigned Depth,
                                  TemplateParameterList &TemplateParams);
  bool isStartOfTemplateTypeParameter();
  DeclPtrTy ParseTemplateParameter(unsigned Depth, unsigned Position);
  DeclPtrTy ParseTypeParameter(unsigned Depth, unsigned Position);
  DeclPtrTy ParseTemplateTemplateParameter(unsigned Depth, unsigned Position);
  DeclPtrTy ParseNonTypeTemplateParameter(unsigned Depth, unsigned Position);
  // C++ 14.3: Template arguments [temp.arg]
  typedef llvm::SmallVector<ParsedTemplateArgument, 16> TemplateArgList;

  bool ParseTemplateIdAfterTemplateName(TemplateTy Template,
                                        SourceLocation TemplateNameLoc,
                                        const CXXScopeSpec *SS,
                                        bool ConsumeLastToken,
                                        SourceLocation &LAngleLoc,
                                        TemplateArgList &TemplateArgs,
                                        SourceLocation &RAngleLoc);

  bool AnnotateTemplateIdToken(TemplateTy Template, TemplateNameKind TNK,
                               const CXXScopeSpec *SS,
                               UnqualifiedId &TemplateName,
                               SourceLocation TemplateKWLoc = SourceLocation(),
                               bool AllowTypeAnnotation = true);
  void AnnotateTemplateIdTokenAsType(const CXXScopeSpec *SS = 0);
  bool ParseTemplateArgumentList(TemplateArgList &TemplateArgs);
  ParsedTemplateArgument ParseTemplateTemplateArgument();
  ParsedTemplateArgument ParseTemplateArgument();
  DeclPtrTy ParseExplicitInstantiation(SourceLocation ExternLoc,
                                       SourceLocation TemplateLoc,
                                       SourceLocation &DeclEnd);

  //===--------------------------------------------------------------------===//
  // GNU G++: Type Traits [Type-Traits.html in the GCC manual]
  OwningExprResult ParseUnaryTypeTrait();
};

}  // end namespace clang

#endif
