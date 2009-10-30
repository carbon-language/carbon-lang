//===--- Preprocessor.h - C Language Family Preprocessor --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Preprocessor interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LEX_PREPROCESSOR_H
#define LLVM_CLANG_LEX_PREPROCESSOR_H

#include "clang/Lex/Lexer.h"
#include "clang/Lex/PTHLexer.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/TokenLexer.h"
#include "clang/Lex/PTHManager.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Allocator.h"
#include <vector>

namespace clang {

class SourceManager;
class FileManager;
class FileEntry;
class HeaderSearch;
class PragmaNamespace;
class PragmaHandler;
class CommentHandler;
class ScratchBuffer;
class TargetInfo;
class PPCallbacks;
class DirectoryLookup;

/// Preprocessor - This object engages in a tight little dance with the lexer to
/// efficiently preprocess tokens.  Lexers know only about tokens within a
/// single source file, and don't know anything about preprocessor-level issues
/// like the #include stack, token expansion, etc.
///
class Preprocessor {
  Diagnostic        *Diags;
  LangOptions        Features;
  TargetInfo        &Target;
  FileManager       &FileMgr;
  SourceManager     &SourceMgr;
  ScratchBuffer     *ScratchBuf;
  HeaderSearch      &HeaderInfo;

  /// PTH - An optional PTHManager object used for getting tokens from
  ///  a token cache rather than lexing the original source file.
  llvm::OwningPtr<PTHManager> PTH;

  /// BP - A BumpPtrAllocator object used to quickly allocate and release
  ///  objects internal to the Preprocessor.
  llvm::BumpPtrAllocator BP;

  /// Identifiers for builtin macros and other builtins.
  IdentifierInfo *Ident__LINE__, *Ident__FILE__;   // __LINE__, __FILE__
  IdentifierInfo *Ident__DATE__, *Ident__TIME__;   // __DATE__, __TIME__
  IdentifierInfo *Ident__INCLUDE_LEVEL__;          // __INCLUDE_LEVEL__
  IdentifierInfo *Ident__BASE_FILE__;              // __BASE_FILE__
  IdentifierInfo *Ident__TIMESTAMP__;              // __TIMESTAMP__
  IdentifierInfo *Ident__COUNTER__;                // __COUNTER__
  IdentifierInfo *Ident_Pragma, *Ident__VA_ARGS__; // _Pragma, __VA_ARGS__
  IdentifierInfo *Ident__has_feature;              // __has_feature
  IdentifierInfo *Ident__has_builtin;              // __has_builtin

  SourceLocation DATELoc, TIMELoc;
  unsigned CounterValue;  // Next __COUNTER__ value.

  enum {
    /// MaxIncludeStackDepth - Maximum depth of #includes.
    MaxAllowedIncludeStackDepth = 200
  };

  // State that is set before the preprocessor begins.
  bool KeepComments : 1;
  bool KeepMacroComments : 1;

  // State that changes while the preprocessor runs:
  bool DisableMacroExpansion : 1;  // True if macro expansion is disabled.
  bool InMacroArgs : 1;            // True if parsing fn macro invocation args.

  /// Identifiers - This is mapping/lookup information for all identifiers in
  /// the program, including program keywords.
  IdentifierTable Identifiers;

  /// Selectors - This table contains all the selectors in the program. Unlike
  /// IdentifierTable above, this table *isn't* populated by the preprocessor.
  /// It is declared/instantiated here because it's role/lifetime is
  /// conceptually similar the IdentifierTable. In addition, the current control
  /// flow (in clang::ParseAST()), make it convenient to put here.
  /// FIXME: Make sure the lifetime of Identifiers/Selectors *isn't* tied to
  /// the lifetime fo the preprocessor.
  SelectorTable Selectors;

  /// BuiltinInfo - Information about builtins.
  Builtin::Context BuiltinInfo;

  /// PragmaHandlers - This tracks all of the pragmas that the client registered
  /// with this preprocessor.
  PragmaNamespace *PragmaHandlers;

  /// \brief Tracks all of the comment handlers that the client registered
  /// with this preprocessor.
  std::vector<CommentHandler *> CommentHandlers;

  /// CurLexer - This is the current top of the stack that we're lexing from if
  /// not expanding a macro and we are lexing directly from source code.
  ///  Only one of CurLexer, CurPTHLexer, or CurTokenLexer will be non-null.
  llvm::OwningPtr<Lexer> CurLexer;

  /// CurPTHLexer - This is the current top of stack that we're lexing from if
  ///  not expanding from a macro and we are lexing from a PTH cache.
  ///  Only one of CurLexer, CurPTHLexer, or CurTokenLexer will be non-null.
  llvm::OwningPtr<PTHLexer> CurPTHLexer;

  /// CurPPLexer - This is the current top of the stack what we're lexing from
  ///  if not expanding a macro.  This is an alias for either CurLexer or
  ///  CurPTHLexer.
  PreprocessorLexer* CurPPLexer;

  /// CurLookup - The DirectoryLookup structure used to find the current
  /// FileEntry, if CurLexer is non-null and if applicable.  This allows us to
  /// implement #include_next and find directory-specific properties.
  const DirectoryLookup *CurDirLookup;

  /// CurTokenLexer - This is the current macro we are expanding, if we are
  /// expanding a macro.  One of CurLexer and CurTokenLexer must be null.
  llvm::OwningPtr<TokenLexer> CurTokenLexer;

  /// IncludeMacroStack - This keeps track of the stack of files currently
  /// #included, and macros currently being expanded from, not counting
  /// CurLexer/CurTokenLexer.
  struct IncludeStackInfo {
    Lexer                 *TheLexer;
    PTHLexer              *ThePTHLexer;
    PreprocessorLexer     *ThePPLexer;
    TokenLexer            *TheTokenLexer;
    const DirectoryLookup *TheDirLookup;

    IncludeStackInfo(Lexer *L, PTHLexer* P, PreprocessorLexer* PPL,
                     TokenLexer* TL, const DirectoryLookup *D)
      : TheLexer(L), ThePTHLexer(P), ThePPLexer(PPL), TheTokenLexer(TL),
        TheDirLookup(D) {}
  };
  std::vector<IncludeStackInfo> IncludeMacroStack;

  /// Callbacks - These are actions invoked when some preprocessor activity is
  /// encountered (e.g. a file is #included, etc).
  PPCallbacks *Callbacks;

  /// Macros - For each IdentifierInfo with 'HasMacro' set, we keep a mapping
  /// to the actual definition of the macro.
  llvm::DenseMap<IdentifierInfo*, MacroInfo*> Macros;

  /// MICache - A "freelist" of MacroInfo objects that can be reused for quick
  ///  allocation.
  std::vector<MacroInfo*> MICache;

  // Various statistics we track for performance analysis.
  unsigned NumDirectives, NumIncluded, NumDefined, NumUndefined, NumPragma;
  unsigned NumIf, NumElse, NumEndif;
  unsigned NumEnteredSourceFiles, MaxIncludeStackDepth;
  unsigned NumMacroExpanded, NumFnMacroExpanded, NumBuiltinMacroExpanded;
  unsigned NumFastMacroExpanded, NumTokenPaste, NumFastTokenPaste;
  unsigned NumSkipped;

  /// Predefines - This string is the predefined macros that preprocessor
  /// should use from the command line etc.
  std::string Predefines;

  /// TokenLexerCache - Cache macro expanders to reduce malloc traffic.
  enum { TokenLexerCacheSize = 8 };
  unsigned NumCachedTokenLexers;
  TokenLexer *TokenLexerCache[TokenLexerCacheSize];

private:  // Cached tokens state.
  typedef llvm::SmallVector<Token, 1> CachedTokensTy;

  /// CachedTokens - Cached tokens are stored here when we do backtracking or
  /// lookahead. They are "lexed" by the CachingLex() method.
  CachedTokensTy CachedTokens;

  /// CachedLexPos - The position of the cached token that CachingLex() should
  /// "lex" next. If it points beyond the CachedTokens vector, it means that
  /// a normal Lex() should be invoked.
  CachedTokensTy::size_type CachedLexPos;

  /// BacktrackPositions - Stack of backtrack positions, allowing nested
  /// backtracks. The EnableBacktrackAtThisPos() method pushes a position to
  /// indicate where CachedLexPos should be set when the BackTrack() method is
  /// invoked (at which point the last position is popped).
  std::vector<CachedTokensTy::size_type> BacktrackPositions;

public:
  Preprocessor(Diagnostic &diags, const LangOptions &opts, TargetInfo &target,
               SourceManager &SM, HeaderSearch &Headers,
               IdentifierInfoLookup *IILookup = 0);

  ~Preprocessor();

  Diagnostic &getDiagnostics() const { return *Diags; }
  void setDiagnostics(Diagnostic &D) { Diags = &D; }

  const LangOptions &getLangOptions() const { return Features; }
  TargetInfo &getTargetInfo() const { return Target; }
  FileManager &getFileManager() const { return FileMgr; }
  SourceManager &getSourceManager() const { return SourceMgr; }
  HeaderSearch &getHeaderSearchInfo() const { return HeaderInfo; }

  IdentifierTable &getIdentifierTable() { return Identifiers; }
  SelectorTable &getSelectorTable() { return Selectors; }
  Builtin::Context &getBuiltinInfo() { return BuiltinInfo; }
  llvm::BumpPtrAllocator &getPreprocessorAllocator() { return BP; }

  void setPTHManager(PTHManager* pm);

  PTHManager *getPTHManager() { return PTH.get(); }

  /// SetCommentRetentionState - Control whether or not the preprocessor retains
  /// comments in output.
  void SetCommentRetentionState(bool KeepComments, bool KeepMacroComments) {
    this->KeepComments = KeepComments | KeepMacroComments;
    this->KeepMacroComments = KeepMacroComments;
  }

  bool getCommentRetentionState() const { return KeepComments; }

  /// isCurrentLexer - Return true if we are lexing directly from the specified
  /// lexer.
  bool isCurrentLexer(const PreprocessorLexer *L) const {
    return CurPPLexer == L;
  }

  /// getCurrentLexer - Return the current lexer being lexed from.  Note
  /// that this ignores any potentially active macro expansions and _Pragma
  /// expansions going on at the time.
  PreprocessorLexer *getCurrentLexer() const { return CurPPLexer; }

  /// getCurrentFileLexer - Return the current file lexer being lexed from.  Note
  /// that this ignores any potentially active macro expansions and _Pragma
  /// expansions going on at the time.
  PreprocessorLexer *getCurrentFileLexer() const;

  /// getPPCallbacks/setPPCallbacks - Accessors for preprocessor callbacks.
  /// Note that this class takes ownership of any PPCallbacks object given to
  /// it.
  PPCallbacks *getPPCallbacks() const { return Callbacks; }
  void setPPCallbacks(PPCallbacks *C) {
    if (Callbacks)
      C = new PPChainedCallbacks(C, Callbacks);
    Callbacks = C;
  }

  /// getMacroInfo - Given an identifier, return the MacroInfo it is #defined to
  /// or null if it isn't #define'd.
  MacroInfo *getMacroInfo(IdentifierInfo *II) const {
    return II->hasMacroDefinition() ? Macros.find(II)->second : 0;
  }

  /// setMacroInfo - Specify a macro for this identifier.
  ///
  void setMacroInfo(IdentifierInfo *II, MacroInfo *MI);

  /// macro_iterator/macro_begin/macro_end - This allows you to walk the current
  /// state of the macro table.  This visits every currently-defined macro.
  typedef llvm::DenseMap<IdentifierInfo*,
                         MacroInfo*>::const_iterator macro_iterator;
  macro_iterator macro_begin() const { return Macros.begin(); }
  macro_iterator macro_end() const { return Macros.end(); }



  const std::string &getPredefines() const { return Predefines; }
  /// setPredefines - Set the predefines for this Preprocessor.  These
  /// predefines are automatically injected when parsing the main file.
  void setPredefines(const char *P) { Predefines = P; }
  void setPredefines(const std::string &P) { Predefines = P; }

  /// getIdentifierInfo - Return information about the specified preprocessor
  /// identifier token.  The version of this method that takes two character
  /// pointers is preferred unless the identifier is already available as a
  /// string (this avoids allocation and copying of memory to construct an
  /// std::string).
  IdentifierInfo *getIdentifierInfo(const char *NameStart,
                                    const char *NameEnd) {
    return &Identifiers.get(NameStart, NameEnd);
  }
  IdentifierInfo *getIdentifierInfo(const char *NameStr) {
    return getIdentifierInfo(NameStr, NameStr+strlen(NameStr));
  }

  /// AddPragmaHandler - Add the specified pragma handler to the preprocessor.
  /// If 'Namespace' is non-null, then it is a token required to exist on the
  /// pragma line before the pragma string starts, e.g. "STDC" or "GCC".
  void AddPragmaHandler(const char *Namespace, PragmaHandler *Handler);

  /// RemovePragmaHandler - Remove the specific pragma handler from
  /// the preprocessor. If \arg Namespace is non-null, then it should
  /// be the namespace that \arg Handler was added to. It is an error
  /// to remove a handler that has not been registered.
  void RemovePragmaHandler(const char *Namespace, PragmaHandler *Handler);

  /// \brief Add the specified comment handler to the preprocessor.
  void AddCommentHandler(CommentHandler *Handler);

  /// \brief Remove the specified comment handler.
  ///
  /// It is an error to remove a handler that has not been registered.
  void RemoveCommentHandler(CommentHandler *Handler);

  /// EnterMainSourceFile - Enter the specified FileID as the main source file,
  /// which implicitly adds the builtin defines etc.
  void EnterMainSourceFile();

  /// EnterSourceFile - Add a source file to the top of the include stack and
  /// start lexing tokens from it instead of the current buffer.  If isMainFile
  /// is true, this is the main file for the translation unit.
  void EnterSourceFile(FileID CurFileID, const DirectoryLookup *Dir);

  /// EnterMacro - Add a Macro to the top of the include stack and start lexing
  /// tokens from it instead of the current buffer.  Args specifies the
  /// tokens input to a function-like macro.
  ///
  /// ILEnd specifies the location of the ')' for a function-like macro or the
  /// identifier for an object-like macro.
  void EnterMacro(Token &Identifier, SourceLocation ILEnd, MacroArgs *Args);

  /// EnterTokenStream - Add a "macro" context to the top of the include stack,
  /// which will cause the lexer to start returning the specified tokens.
  ///
  /// If DisableMacroExpansion is true, tokens lexed from the token stream will
  /// not be subject to further macro expansion.  Otherwise, these tokens will
  /// be re-macro-expanded when/if expansion is enabled.
  ///
  /// If OwnsTokens is false, this method assumes that the specified stream of
  /// tokens has a permanent owner somewhere, so they do not need to be copied.
  /// If it is true, it assumes the array of tokens is allocated with new[] and
  /// must be freed.
  ///
  void EnterTokenStream(const Token *Toks, unsigned NumToks,
                        bool DisableMacroExpansion, bool OwnsTokens);

  /// RemoveTopOfLexerStack - Pop the current lexer/macro exp off the top of the
  /// lexer stack.  This should only be used in situations where the current
  /// state of the top-of-stack lexer is known.
  void RemoveTopOfLexerStack();

  /// EnableBacktrackAtThisPos - From the point that this method is called, and
  /// until CommitBacktrackedTokens() or Backtrack() is called, the Preprocessor
  /// keeps track of the lexed tokens so that a subsequent Backtrack() call will
  /// make the Preprocessor re-lex the same tokens.
  ///
  /// Nested backtracks are allowed, meaning that EnableBacktrackAtThisPos can
  /// be called multiple times and CommitBacktrackedTokens/Backtrack calls will
  /// be combined with the EnableBacktrackAtThisPos calls in reverse order.
  ///
  /// NOTE: *DO NOT* forget to call either CommitBacktrackedTokens or Backtrack
  /// at some point after EnableBacktrackAtThisPos. If you don't, caching of
  /// tokens will continue indefinitely.
  ///
  void EnableBacktrackAtThisPos();

  /// CommitBacktrackedTokens - Disable the last EnableBacktrackAtThisPos call.
  void CommitBacktrackedTokens();

  /// Backtrack - Make Preprocessor re-lex the tokens that were lexed since
  /// EnableBacktrackAtThisPos() was previously called.
  void Backtrack();

  /// isBacktrackEnabled - True if EnableBacktrackAtThisPos() was called and
  /// caching of tokens is on.
  bool isBacktrackEnabled() const { return !BacktrackPositions.empty(); }

  /// Lex - To lex a token from the preprocessor, just pull a token from the
  /// current lexer or macro object.
  void Lex(Token &Result) {
    if (CurLexer)
      CurLexer->Lex(Result);
    else if (CurPTHLexer)
      CurPTHLexer->Lex(Result);
    else if (CurTokenLexer)
      CurTokenLexer->Lex(Result);
    else
      CachingLex(Result);
  }

  /// LexNonComment - Lex a token.  If it's a comment, keep lexing until we get
  /// something not a comment.  This is useful in -E -C mode where comments
  /// would foul up preprocessor directive handling.
  void LexNonComment(Token &Result) {
    do
      Lex(Result);
    while (Result.getKind() == tok::comment);
  }

  /// LexUnexpandedToken - This is just like Lex, but this disables macro
  /// expansion of identifier tokens.
  void LexUnexpandedToken(Token &Result) {
    // Disable macro expansion.
    bool OldVal = DisableMacroExpansion;
    DisableMacroExpansion = true;
    // Lex the token.
    Lex(Result);

    // Reenable it.
    DisableMacroExpansion = OldVal;
  }

  /// LookAhead - This peeks ahead N tokens and returns that token without
  /// consuming any tokens.  LookAhead(0) returns the next token that would be
  /// returned by Lex(), LookAhead(1) returns the token after it, etc.  This
  /// returns normal tokens after phase 5.  As such, it is equivalent to using
  /// 'Lex', not 'LexUnexpandedToken'.
  const Token &LookAhead(unsigned N) {
    if (CachedLexPos + N < CachedTokens.size())
      return CachedTokens[CachedLexPos+N];
    else
      return PeekAhead(N+1);
  }

  /// RevertCachedTokens - When backtracking is enabled and tokens are cached,
  /// this allows to revert a specific number of tokens.
  /// Note that the number of tokens being reverted should be up to the last
  /// backtrack position, not more.
  void RevertCachedTokens(unsigned N) {
    assert(isBacktrackEnabled() &&
           "Should only be called when tokens are cached for backtracking");
    assert(signed(CachedLexPos) - signed(N) >= signed(BacktrackPositions.back())
         && "Should revert tokens up to the last backtrack position, not more");
    assert(signed(CachedLexPos) - signed(N) >= 0 &&
           "Corrupted backtrack positions ?");
    CachedLexPos -= N;
  }

  /// EnterToken - Enters a token in the token stream to be lexed next. If
  /// BackTrack() is called afterwards, the token will remain at the insertion
  /// point.
  void EnterToken(const Token &Tok) {
    EnterCachingLexMode();
    CachedTokens.insert(CachedTokens.begin()+CachedLexPos, Tok);
  }

  /// AnnotateCachedTokens - We notify the Preprocessor that if it is caching
  /// tokens (because backtrack is enabled) it should replace the most recent
  /// cached tokens with the given annotation token. This function has no effect
  /// if backtracking is not enabled.
  ///
  /// Note that the use of this function is just for optimization; so that the
  /// cached tokens doesn't get re-parsed and re-resolved after a backtrack is
  /// invoked.
  void AnnotateCachedTokens(const Token &Tok) {
    assert(Tok.isAnnotation() && "Expected annotation token");
    if (CachedLexPos != 0 && isBacktrackEnabled())
      AnnotatePreviousCachedTokens(Tok);
  }

  /// \brief Replace the last token with an annotation token.
  ///
  /// Like AnnotateCachedTokens(), this routine replaces an
  /// already-parsed (and resolved) token with an annotation
  /// token. However, this routine only replaces the last token with
  /// the annotation token; it does not affect any other cached
  /// tokens. This function has no effect if backtracking is not
  /// enabled.
  void ReplaceLastTokenWithAnnotation(const Token &Tok) {
    assert(Tok.isAnnotation() && "Expected annotation token");
    if (CachedLexPos != 0 && isBacktrackEnabled())
      CachedTokens[CachedLexPos-1] = Tok;
  }

  /// Diag - Forwarding function for diagnostics.  This emits a diagnostic at
  /// the specified Token's location, translating the token's start
  /// position in the current buffer into a SourcePosition object for rendering.
  DiagnosticBuilder Diag(SourceLocation Loc, unsigned DiagID) {
    return Diags->Report(FullSourceLoc(Loc, getSourceManager()), DiagID);
  }

  DiagnosticBuilder Diag(const Token &Tok, unsigned DiagID) {
    return Diags->Report(FullSourceLoc(Tok.getLocation(), getSourceManager()),
                         DiagID);
  }

  /// getSpelling() - Return the 'spelling' of the Tok token.  The spelling of a
  /// token is the characters used to represent the token in the source file
  /// after trigraph expansion and escaped-newline folding.  In particular, this
  /// wants to get the true, uncanonicalized, spelling of things like digraphs
  /// UCNs, etc.
  std::string getSpelling(const Token &Tok) const;

  /// getSpelling - This method is used to get the spelling of a token into a
  /// preallocated buffer, instead of as an std::string.  The caller is required
  /// to allocate enough space for the token, which is guaranteed to be at least
  /// Tok.getLength() bytes long.  The length of the actual result is returned.
  ///
  /// Note that this method may do two possible things: it may either fill in
  /// the buffer specified with characters, or it may *change the input pointer*
  /// to point to a constant buffer with the data already in it (avoiding a
  /// copy).  The caller is not allowed to modify the returned buffer pointer
  /// if an internal buffer is returned.
  unsigned getSpelling(const Token &Tok, const char *&Buffer) const;

  /// getSpellingOfSingleCharacterNumericConstant - Tok is a numeric constant
  /// with length 1, return the character.
  char getSpellingOfSingleCharacterNumericConstant(const Token &Tok) const {
    assert(Tok.is(tok::numeric_constant) &&
           Tok.getLength() == 1 && "Called on unsupported token");
    assert(!Tok.needsCleaning() && "Token can't need cleaning with length 1");

    // If the token is carrying a literal data pointer, just use it.
    if (const char *D = Tok.getLiteralData())
      return *D;

    // Otherwise, fall back on getCharacterData, which is slower, but always
    // works.
    return *SourceMgr.getCharacterData(Tok.getLocation());
  }

  /// CreateString - Plop the specified string into a scratch buffer and set the
  /// specified token's location and length to it.  If specified, the source
  /// location provides a location of the instantiation point of the token.
  void CreateString(const char *Buf, unsigned Len,
                    Token &Tok, SourceLocation SourceLoc = SourceLocation());

  /// \brief Computes the source location just past the end of the
  /// token at this source location.
  ///
  /// This routine can be used to produce a source location that
  /// points just past the end of the token referenced by \p Loc, and
  /// is generally used when a diagnostic needs to point just after a
  /// token where it expected something different that it received. If
  /// the returned source location would not be meaningful (e.g., if
  /// it points into a macro), this routine returns an invalid
  /// source location.
  SourceLocation getLocForEndOfToken(SourceLocation Loc);

  /// DumpToken - Print the token to stderr, used for debugging.
  ///
  void DumpToken(const Token &Tok, bool DumpFlags = false) const;
  void DumpLocation(SourceLocation Loc) const;
  void DumpMacro(const MacroInfo &MI) const;

  /// AdvanceToTokenCharacter - Given a location that specifies the start of a
  /// token, return a new location that specifies a character within the token.
  SourceLocation AdvanceToTokenCharacter(SourceLocation TokStart,unsigned Char);

  /// IncrementPasteCounter - Increment the counters for the number of token
  /// paste operations performed.  If fast was specified, this is a 'fast paste'
  /// case we handled.
  ///
  void IncrementPasteCounter(bool isFast) {
    if (isFast)
      ++NumFastTokenPaste;
    else
      ++NumTokenPaste;
  }

  void PrintStats();

  /// HandleMicrosoftCommentPaste - When the macro expander pastes together a
  /// comment (/##/) in microsoft mode, this method handles updating the current
  /// state, returning the token on the next source line.
  void HandleMicrosoftCommentPaste(Token &Tok);

  //===--------------------------------------------------------------------===//
  // Preprocessor callback methods.  These are invoked by a lexer as various
  // directives and events are found.

  /// LookUpIdentifierInfo - Given a tok::identifier token, look up the
  /// identifier information for the token and install it into the token.
  IdentifierInfo *LookUpIdentifierInfo(Token &Identifier,
                                       const char *BufPtr = 0);

  /// HandleIdentifier - This callback is invoked when the lexer reads an
  /// identifier and has filled in the tokens IdentifierInfo member.  This
  /// callback potentially macro expands it or turns it into a named token (like
  /// 'for').
  void HandleIdentifier(Token &Identifier);


  /// HandleEndOfFile - This callback is invoked when the lexer hits the end of
  /// the current file.  This either returns the EOF token and returns true, or
  /// pops a level off the include stack and returns false, at which point the
  /// client should call lex again.
  bool HandleEndOfFile(Token &Result, bool isEndOfMacro = false);

  /// HandleEndOfTokenLexer - This callback is invoked when the current
  /// TokenLexer hits the end of its token stream.
  bool HandleEndOfTokenLexer(Token &Result);

  /// HandleDirective - This callback is invoked when the lexer sees a # token
  /// at the start of a line.  This consumes the directive, modifies the
  /// lexer/preprocessor state, and advances the lexer(s) so that the next token
  /// read is the correct one.
  void HandleDirective(Token &Result);

  /// CheckEndOfDirective - Ensure that the next token is a tok::eom token.  If
  /// not, emit a diagnostic and consume up until the eom.  If EnableMacros is
  /// true, then we consider macros that expand to zero tokens as being ok.
  void CheckEndOfDirective(const char *Directive, bool EnableMacros = false);

  /// DiscardUntilEndOfDirective - Read and discard all tokens remaining on the
  /// current line until the tok::eom token is found.
  void DiscardUntilEndOfDirective();

  /// SawDateOrTime - This returns true if the preprocessor has seen a use of
  /// __DATE__ or __TIME__ in the file so far.
  bool SawDateOrTime() const {
    return DATELoc != SourceLocation() || TIMELoc != SourceLocation();
  }
  unsigned getCounterValue() const { return CounterValue; }
  void setCounterValue(unsigned V) { CounterValue = V; }

  /// AllocateMacroInfo - Allocate a new MacroInfo object with the provide
  ///  SourceLocation.
  MacroInfo* AllocateMacroInfo(SourceLocation L);

  /// GetIncludeFilenameSpelling - Turn the specified lexer token into a fully
  /// checked and spelled filename, e.g. as an operand of #include. This returns
  /// true if the input filename was in <>'s or false if it were in ""'s.  The
  /// caller is expected to provide a buffer that is large enough to hold the
  /// spelling of the filename, but is also expected to handle the case when
  /// this method decides to use a different buffer.
  bool GetIncludeFilenameSpelling(SourceLocation Loc,
                                  const char *&BufStart, const char *&BufEnd);

  /// LookupFile - Given a "foo" or <foo> reference, look up the indicated file,
  /// return null on failure.  isAngled indicates whether the file reference is
  /// for system #include's or not (i.e. using <> instead of "").
  const FileEntry *LookupFile(const char *FilenameStart,const char *FilenameEnd,
                              bool isAngled, const DirectoryLookup *FromDir,
                              const DirectoryLookup *&CurDir);

  /// GetCurLookup - The DirectoryLookup structure used to find the current
  /// FileEntry, if CurLexer is non-null and if applicable.  This allows us to
  /// implement #include_next and find directory-specific properties.
  const DirectoryLookup *GetCurDirLookup() { return CurDirLookup; }

  /// isInPrimaryFile - Return true if we're in the top-level file, not in a
  /// #include.
  bool isInPrimaryFile() const;

  /// ConcatenateIncludeName - Handle cases where the #include name is expanded
  /// from a macro as multiple tokens, which need to be glued together.  This
  /// occurs for code like:
  ///    #define FOO <a/b.h>
  ///    #include FOO
  /// because in this case, "<a/b.h>" is returned as 7 tokens, not one.
  ///
  /// This code concatenates and consumes tokens up to the '>' token.  It returns
  /// false if the > was found, otherwise it returns true if it finds and consumes
  /// the EOM marker.
  bool ConcatenateIncludeName(llvm::SmallVector<char, 128> &FilenameBuffer);

private:

  void PushIncludeMacroStack() {
    IncludeMacroStack.push_back(IncludeStackInfo(CurLexer.take(),
                                                 CurPTHLexer.take(),
                                                 CurPPLexer,
                                                 CurTokenLexer.take(),
                                                 CurDirLookup));
    CurPPLexer = 0;
  }

  void PopIncludeMacroStack() {
    CurLexer.reset(IncludeMacroStack.back().TheLexer);
    CurPTHLexer.reset(IncludeMacroStack.back().ThePTHLexer);
    CurPPLexer = IncludeMacroStack.back().ThePPLexer;
    CurTokenLexer.reset(IncludeMacroStack.back().TheTokenLexer);
    CurDirLookup  = IncludeMacroStack.back().TheDirLookup;
    IncludeMacroStack.pop_back();
  }

  /// ReleaseMacroInfo - Release the specified MacroInfo.  This memory will
  ///  be reused for allocating new MacroInfo objects.
  void ReleaseMacroInfo(MacroInfo* MI);

  /// ReadMacroName - Lex and validate a macro name, which occurs after a
  /// #define or #undef.  This emits a diagnostic, sets the token kind to eom,
  /// and discards the rest of the macro line if the macro name is invalid.
  void ReadMacroName(Token &MacroNameTok, char isDefineUndef = 0);

  /// ReadMacroDefinitionArgList - The ( starting an argument list of a macro
  /// definition has just been read.  Lex the rest of the arguments and the
  /// closing ), updating MI with what we learn.  Return true if an error occurs
  /// parsing the arg list.
  bool ReadMacroDefinitionArgList(MacroInfo *MI);

  /// SkipExcludedConditionalBlock - We just read a #if or related directive and
  /// decided that the subsequent tokens are in the #if'd out portion of the
  /// file.  Lex the rest of the file, until we see an #endif.  If
  /// FoundNonSkipPortion is true, then we have already emitted code for part of
  /// this #if directive, so #else/#elif blocks should never be entered. If
  /// FoundElse is false, then #else directives are ok, if not, then we have
  /// already seen one so a #else directive is a duplicate.  When this returns,
  /// the caller can lex the first valid token.
  void SkipExcludedConditionalBlock(SourceLocation IfTokenLoc,
                                    bool FoundNonSkipPortion, bool FoundElse);

  /// PTHSkipExcludedConditionalBlock - A fast PTH version of
  ///  SkipExcludedConditionalBlock.
  void PTHSkipExcludedConditionalBlock();

  /// EvaluateDirectiveExpression - Evaluate an integer constant expression that
  /// may occur after a #if or #elif directive and return it as a bool.  If the
  /// expression is equivalent to "!defined(X)" return X in IfNDefMacro.
  bool EvaluateDirectiveExpression(IdentifierInfo *&IfNDefMacro);

  /// RegisterBuiltinPragmas - Install the standard preprocessor pragmas:
  /// #pragma GCC poison/system_header/dependency and #pragma once.
  void RegisterBuiltinPragmas();

  /// RegisterBuiltinMacros - Register builtin macros, such as __LINE__ with the
  /// identifier table.
  void RegisterBuiltinMacros();

  /// HandleMacroExpandedIdentifier - If an identifier token is read that is to
  /// be expanded as a macro, handle it and return the next token as 'Tok'.  If
  /// the macro should not be expanded return true, otherwise return false.
  bool HandleMacroExpandedIdentifier(Token &Tok, MacroInfo *MI);

  /// isNextPPTokenLParen - Determine whether the next preprocessor token to be
  /// lexed is a '('.  If so, consume the token and return true, if not, this
  /// method should have no observable side-effect on the lexed tokens.
  bool isNextPPTokenLParen();

  /// ReadFunctionLikeMacroArgs - After reading "MACRO(", this method is
  /// invoked to read all of the formal arguments specified for the macro
  /// invocation.  This returns null on error.
  MacroArgs *ReadFunctionLikeMacroArgs(Token &MacroName, MacroInfo *MI,
                                       SourceLocation &InstantiationEnd);

  /// ExpandBuiltinMacro - If an identifier token is read that is to be expanded
  /// as a builtin macro, handle it and return the next token as 'Tok'.
  void ExpandBuiltinMacro(Token &Tok);

  /// Handle_Pragma - Read a _Pragma directive, slice it up, process it, then
  /// return the first token after the directive.  The _Pragma token has just
  /// been read into 'Tok'.
  void Handle_Pragma(Token &Tok);

  /// EnterSourceFileWithLexer - Add a lexer to the top of the include stack and
  /// start lexing tokens from it instead of the current buffer.
  void EnterSourceFileWithLexer(Lexer *TheLexer, const DirectoryLookup *Dir);

  /// EnterSourceFileWithPTH - Add a lexer to the top of the include stack and
  /// start getting tokens from it using the PTH cache.
  void EnterSourceFileWithPTH(PTHLexer *PL, const DirectoryLookup *Dir);

  /// IsFileLexer - Returns true if we are lexing from a file and not a
  ///  pragma or a macro.
  static bool IsFileLexer(const Lexer* L, const PreprocessorLexer* P) {
    return L ? !L->isPragmaLexer() : P != 0;
  }

  static bool IsFileLexer(const IncludeStackInfo& I) {
    return IsFileLexer(I.TheLexer, I.ThePPLexer);
  }

  bool IsFileLexer() const {
    return IsFileLexer(CurLexer.get(), CurPPLexer);
  }

  //===--------------------------------------------------------------------===//
  // Caching stuff.
  void CachingLex(Token &Result);
  bool InCachingLexMode() const { return CurPPLexer == 0 && CurTokenLexer == 0;}
  void EnterCachingLexMode();
  void ExitCachingLexMode() {
    if (InCachingLexMode())
      RemoveTopOfLexerStack();
  }
  const Token &PeekAhead(unsigned N);
  void AnnotatePreviousCachedTokens(const Token &Tok);

  //===--------------------------------------------------------------------===//
  /// Handle*Directive - implement the various preprocessor directives.  These
  /// should side-effect the current preprocessor object so that the next call
  /// to Lex() will return the appropriate token next.
  void HandleLineDirective(Token &Tok);
  void HandleDigitDirective(Token &Tok);
  void HandleUserDiagnosticDirective(Token &Tok, bool isWarning);
  void HandleIdentSCCSDirective(Token &Tok);

  // File inclusion.
  void HandleIncludeDirective(Token &Tok,
                              const DirectoryLookup *LookupFrom = 0,
                              bool isImport = false);
  void HandleIncludeNextDirective(Token &Tok);
  void HandleIncludeMacrosDirective(Token &Tok);
  void HandleImportDirective(Token &Tok);

  // Macro handling.
  void HandleDefineDirective(Token &Tok);
  void HandleUndefDirective(Token &Tok);
  // HandleAssertDirective(Token &Tok);
  // HandleUnassertDirective(Token &Tok);

  // Conditional Inclusion.
  void HandleIfdefDirective(Token &Tok, bool isIfndef,
                            bool ReadAnyTokensBeforeDirective);
  void HandleIfDirective(Token &Tok, bool ReadAnyTokensBeforeDirective);
  void HandleEndifDirective(Token &Tok);
  void HandleElseDirective(Token &Tok);
  void HandleElifDirective(Token &Tok);

  // Pragmas.
  void HandlePragmaDirective();
public:
  void HandlePragmaOnce(Token &OnceTok);
  void HandlePragmaMark();
  void HandlePragmaPoison(Token &PoisonTok);
  void HandlePragmaSystemHeader(Token &SysHeaderTok);
  void HandlePragmaDependency(Token &DependencyTok);
  void HandlePragmaComment(Token &CommentTok);
  void HandleComment(SourceRange Comment);
};

/// PreprocessorFactory - A generic factory interface for lazily creating
///  Preprocessor objects on-demand when they are needed.
class PreprocessorFactory {
public:
  virtual ~PreprocessorFactory();
  virtual Preprocessor* CreatePreprocessor() = 0;
};

/// \brief Abstract base class that describes a handler that will receive
/// source ranges for each of the comments encountered in the source file.
class CommentHandler {
public:
  virtual ~CommentHandler();

  virtual void HandleComment(Preprocessor &PP, SourceRange Comment) = 0;
};

}  // end namespace clang

#endif
