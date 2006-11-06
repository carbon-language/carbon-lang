//===--- Preprocessor.h - C Language Family Preprocessor --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Preprocessor interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LEX_PREPROCESSOR_H
#define LLVM_CLANG_LEX_PREPROCESSOR_H

#include "clang/Lex/DirectoryLookup.h"
#include "clang/Lex/IdentifierTable.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/MacroExpander.h"
#include "clang/Basic/SourceLocation.h"

namespace llvm {
namespace clang {
  
class SourceManager;
class FileManager;
class FileEntry;
class HeaderSearch;
class PragmaNamespace;
class PragmaHandler;
class ScratchBuffer;
class TargetInfo;

/// Preprocessor - This object forms engages in a tight little dance to
/// efficiently preprocess tokens.  Lexers know only about tokens within a
/// single source file, and don't know anything about preprocessor-level issues
/// like the #include stack, token expansion, etc.
///
class Preprocessor {
  Diagnostic        &Diags;
  const LangOptions &Features;
  TargetInfo        &Target;
  FileManager       &FileMgr;
  SourceManager     &SourceMgr;
  ScratchBuffer     *ScratchBuf;
  HeaderSearch      &HeaderInfo;
    
  /// Identifiers for builtin macros and other builtins.
  IdentifierInfo *Ident__LINE__, *Ident__FILE__;   // __LINE__, __FILE__
  IdentifierInfo *Ident__DATE__, *Ident__TIME__;   // __DATE__, __TIME__
  IdentifierInfo *Ident__INCLUDE_LEVEL__;          // __INCLUDE_LEVEL__
  IdentifierInfo *Ident__BASE_FILE__;              // __BASE_FILE__
  IdentifierInfo *Ident__TIMESTAMP__;              // __TIMESTAMP__
  IdentifierInfo *Ident_Pragma, *Ident__VA_ARGS__; // _Pragma, __VA_ARGS__
  
  SourceLocation DATELoc, TIMELoc;
public:
  enum FileChangeReason {
    EnterFile, ExitFile, SystemHeaderPragma, RenameFile
  };
  typedef void (*FileChangeHandler_t)(SourceLocation Loc, 
                                      FileChangeReason Reason,
                                      DirectoryLookup::DirType FileType);
  typedef void (*IdentHandler_t)(SourceLocation Loc, const std::string &str);
  
private:
  /// FileChangeHandler - This callback is invoked whenever a source file is
  /// entered or exited.  The SourceLocation indicates the new location, and
  /// EnteringFile indicates whether this is because we are entering a new
  /// #include'd file (when true) or whether we're exiting one because we ran
  /// off the end (when false).
  FileChangeHandler_t FileChangeHandler;
  
  /// IdentHandler - This is called whenever a #ident or #sccs directive is
  /// found.
  IdentHandler_t IdentHandler;
  
  enum {
    /// MaxIncludeStackDepth - Maximum depth of #includes.
    MaxAllowedIncludeStackDepth = 200
  };

  // State that changes while the preprocessor runs:
  bool DisableMacroExpansion;    // True if macro expansion is disabled.
  bool InMacroArgs;              // True if parsing fn macro invocation args.

  /// Identifiers - This is mapping/lookup information for all identifiers in
  /// the program, including program keywords.
  IdentifierTable Identifiers;
  
  /// PragmaHandlers - This tracks all of the pragmas that the client registered
  /// with this preprocessor.
  PragmaNamespace *PragmaHandlers;
  
  /// CurLexer - This is the current top of the stack that we're lexing from if
  /// not expanding a macro.  One of CurLexer and CurMacroExpander must be null.
  Lexer *CurLexer;
  
  /// CurLookup - The DirectoryLookup structure used to find the current
  /// FileEntry, if CurLexer is non-null and if applicable.  This allows us to
  /// implement #include_next and find directory-specific properties.
  const DirectoryLookup *CurDirLookup;

  /// CurMacroExpander - This is the current macro we are expanding, if we are
  /// expanding a macro.  One of CurLexer and CurMacroExpander must be null.
  MacroExpander *CurMacroExpander;
  
  /// IncludeMacroStack - This keeps track of the stack of files currently
  /// #included, and macros currently being expanded from, not counting
  /// CurLexer/CurMacroExpander.
  struct IncludeStackInfo {
    Lexer *TheLexer;
    const DirectoryLookup *TheDirLookup;
    MacroExpander *TheMacroExpander;
    IncludeStackInfo(Lexer *L, const DirectoryLookup *D, MacroExpander *M)
      : TheLexer(L), TheDirLookup(D), TheMacroExpander(M) {
    }
  };
  std::vector<IncludeStackInfo> IncludeMacroStack;
  
  
  // Various statistics we track for performance analysis.
  unsigned NumDirectives, NumIncluded, NumDefined, NumUndefined, NumPragma;
  unsigned NumIf, NumElse, NumEndif;
  unsigned NumEnteredSourceFiles, MaxIncludeStackDepth;
  unsigned NumMacroExpanded, NumFnMacroExpanded, NumBuiltinMacroExpanded;
  unsigned NumFastMacroExpanded, NumTokenPaste, NumFastTokenPaste;
  unsigned NumSkipped;
public:
  Preprocessor(Diagnostic &diags, const LangOptions &opts, TargetInfo &target,
               FileManager &FM, SourceManager &SM, HeaderSearch &Headers);
  ~Preprocessor();

  Diagnostic &getDiagnostics() const { return Diags; }
  const LangOptions &getLangOptions() const { return Features; }
  TargetInfo &getTargetInfo() const { return Target; }
  FileManager &getFileManager() const { return FileMgr; }
  SourceManager &getSourceManager() const { return SourceMgr; }
  HeaderSearch &getHeaderSearchInfo() const { return HeaderInfo; }

  IdentifierTable &getIdentifierTable() { return Identifiers; }

  /// isCurrentLexer - Return true if we are lexing directly from the specified
  /// lexer.
  bool isCurrentLexer(const Lexer *L) const {
    return CurLexer == L;
  }
  
  /// isInPrimaryFile - Return true if we're in the top-level file, not in a
  /// #include.
  bool isInPrimaryFile() const;
  
  /// getCurrentLexer - Return the current file lexer being lexed from.  Note
  /// that this ignores any potentially active macro expansions and _Pragma
  /// expansions going on at the time.
  Lexer *getCurrentFileLexer() const;
  
  
  /// setFileChangeHandler - Set the callback invoked whenever a source file is
  /// entered or exited.  The SourceLocation indicates the new location, and
  /// EnteringFile indicates whether this is because we are entering a new
  /// #include'd file (when true) or whether we're exiting one because we ran
  /// off the end (when false).
  void setFileChangeHandler(FileChangeHandler_t Handler) {
    FileChangeHandler = Handler;
  }

  /// setIdentHandler - Set the callback invoked whenever a #ident/#sccs
  /// directive is found.
  void setIdentHandler(IdentHandler_t Handler) {
    IdentHandler = Handler;
  }
  
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

  /// EnterSourceFile - Add a source file to the top of the include stack and
  /// start lexing tokens from it instead of the current buffer.  If isMainFile
  /// is true, this is the main file for the translation unit.
  void EnterSourceFile(unsigned CurFileID, const DirectoryLookup *Dir,
                       bool isMainFile = false);

  /// EnterMacro - Add a Macro to the top of the include stack and start lexing
  /// tokens from it instead of the current buffer.  Args specifies the
  /// tokens input to a function-like macro.
  void EnterMacro(LexerToken &Identifier, MacroArgs *Args);
  
  /// EnterTokenStream - Add a "macro" context to the top of the include stack,
  /// which will cause the lexer to start returning the specified tokens.  Note
  /// that these tokens will be re-macro-expanded when/if expansion is enabled.
  /// This method assumes that the specified stream of tokens has a permanent
  /// owner somewhere, so they do not need to be copied.
  void EnterTokenStream(const LexerToken *Toks, unsigned NumToks);
  
  /// RemoveTopOfLexerStack - Pop the current lexer/macro exp off the top of the
  /// lexer stack.  This should only be used in situations where the current
  /// state of the top-of-stack lexer is known.
  void RemoveTopOfLexerStack();
  
  /// Lex - To lex a token from the preprocessor, just pull a token from the
  /// current lexer or macro object.
  void Lex(LexerToken &Result) {
    if (CurLexer)
      CurLexer->Lex(Result);
    else
      CurMacroExpander->Lex(Result);
  }
  
  /// LexNonComment - Lex a token.  If it's a comment, keep lexing until we get
  /// something not a comment.  This is useful in -E -C mode where comments
  /// would foul up preprocessor directive handling.
  void LexNonComment(LexerToken &Result) {
    do
      Lex(Result);
    while (Result.getKind() == tok::comment);
  }
  
  /// LexUnexpandedToken - This is just like Lex, but this disables macro
  /// expansion of identifier tokens.
  void LexUnexpandedToken(LexerToken &Result) {
    // Disable macro expansion.
    bool OldVal = DisableMacroExpansion;
    DisableMacroExpansion = true;
    // Lex the token.
    Lex(Result);
    
    // Reenable it.
    DisableMacroExpansion = OldVal;
  }
  
  /// Diag - Forwarding function for diagnostics.  This emits a diagnostic at
  /// the specified LexerToken's location, translating the token's start
  /// position in the current buffer into a SourcePosition object for rendering.
  void Diag(SourceLocation Loc, unsigned DiagID,
            const std::string &Msg = std::string());  
  void Diag(const LexerToken &Tok, unsigned DiagID,
            const std::string &Msg = std::string()) {
    Diag(Tok.getLocation(), DiagID, Msg);
  }
  
  /// getSpelling() - Return the 'spelling' of the Tok token.  The spelling of a
  /// token is the characters used to represent the token in the source file
  /// after trigraph expansion and escaped-newline folding.  In particular, this
  /// wants to get the true, uncanonicalized, spelling of things like digraphs
  /// UCNs, etc.
  std::string getSpelling(const LexerToken &Tok) const;
  
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
  unsigned getSpelling(const LexerToken &Tok, const char *&Buffer) const;
  
  
  /// CreateString - Plop the specified string into a scratch buffer and return
  /// a location for it.  If specified, the source location provides a source
  /// location for the token.
  SourceLocation CreateString(const char *Buf, unsigned Len,
                              SourceLocation SourceLoc = SourceLocation());
  
  /// DumpToken - Print the token to stderr, used for debugging.
  ///
  void DumpToken(const LexerToken &Tok, bool DumpFlags = false) const;
  void DumpMacro(const MacroInfo &MI) const;
  
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

  //===--------------------------------------------------------------------===//
  // Preprocessor callback methods.  These are invoked by a lexer as various
  // directives and events are found.

  /// LookUpIdentifierInfo - Given a tok::identifier token, look up the
  /// identifier information for the token and install it into the token.
  IdentifierInfo *LookUpIdentifierInfo(LexerToken &Identifier,
                                       const char *BufPtr = 0);
  
  /// HandleIdentifier - This callback is invoked when the lexer reads an
  /// identifier and has filled in the tokens IdentifierInfo member.  This
  /// callback potentially macro expands it or turns it into a named token (like
  /// 'for').
  void HandleIdentifier(LexerToken &Identifier);

  
  /// HandleEndOfFile - This callback is invoked when the lexer hits the end of
  /// the current file.  This either returns the EOF token and returns true, or
  /// pops a level off the include stack and returns false, at which point the
  /// client should call lex again.
  bool HandleEndOfFile(LexerToken &Result, bool isEndOfMacro = false);
  
  /// HandleEndOfMacro - This callback is invoked when the lexer hits the end of
  /// the current macro line.  It returns true if Result is filled in with a
  /// token, or false if Lex should be called again.
  bool HandleEndOfMacro(LexerToken &Result);
  
  /// HandleDirective - This callback is invoked when the lexer sees a # token
  /// at the start of a line.  This consumes the directive, modifies the 
  /// lexer/preprocessor state, and advances the lexer(s) so that the next token
  /// read is the correct one.
  void HandleDirective(LexerToken &Result);

  /// CheckEndOfDirective - Ensure that the next token is a tok::eom token.  If
  /// not, emit a diagnostic and consume up until the eom.
  void CheckEndOfDirective(const char *Directive);
private:

  /// DiscardUntilEndOfDirective - Read and discard all tokens remaining on the
  /// current line until the tok::eom token is found.
  void DiscardUntilEndOfDirective();

  /// ReadMacroName - Lex and validate a macro name, which occurs after a
  /// #define or #undef.  This emits a diagnostic, sets the token kind to eom,
  /// and discards the rest of the macro line if the macro name is invalid.
  void ReadMacroName(LexerToken &MacroNameTok, char isDefineUndef = 0);
  
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
  IdentifierInfo *RegisterBuiltinMacro(const char *Name);
  
  /// HandleMacroExpandedIdentifier - If an identifier token is read that is to
  /// be expanded as a macro, handle it and return the next token as 'Tok'.  If
  /// the macro should not be expanded return true, otherwise return false.
  bool HandleMacroExpandedIdentifier(LexerToken &Tok, MacroInfo *MI);
  
  /// isNextPPTokenLParen - Determine whether the next preprocessor token to be
  /// lexed is a '('.  If so, consume the token and return true, if not, this
  /// method should have no observable side-effect on the lexed tokens.
  bool isNextPPTokenLParen();
  
  /// ReadFunctionLikeMacroArgs - After reading "MACRO(", this method is
  /// invoked to read all of the formal arguments specified for the macro
  /// invocation.  This returns null on error.
  MacroArgs *ReadFunctionLikeMacroArgs(LexerToken &MacroName, MacroInfo *MI);

  /// ExpandBuiltinMacro - If an identifier token is read that is to be expanded
  /// as a builtin macro, handle it and return the next token as 'Tok'.
  void ExpandBuiltinMacro(LexerToken &Tok);
  
  /// Handle_Pragma - Read a _Pragma directive, slice it up, process it, then
  /// return the first token after the directive.  The _Pragma token has just
  /// been read into 'Tok'.
  void Handle_Pragma(LexerToken &Tok);
  
  
  /// EnterSourceFileWithLexer - Add a lexer to the top of the include stack and
  /// start lexing tokens from it instead of the current buffer.
  void EnterSourceFileWithLexer(Lexer *TheLexer, const DirectoryLookup *Dir);
  
  /// GetIncludeFilenameSpelling - Turn the specified lexer token into a fully
  /// checked and spelled filename, e.g. as an operand of #include. This returns
  /// true if the input filename was in <>'s or false if it were in ""'s.  The
  /// caller is expected to provide a buffer that is large enough to hold the
  /// spelling of the filename, but is also expected to handle the case when
  /// this method decides to use a different buffer.
  bool GetIncludeFilenameSpelling(const LexerToken &FNTok,
                                  const char *&BufStart, const char *&BufEnd);
  
  /// LookupFile - Given a "foo" or <foo> reference, look up the indicated file,
  /// return null on failure.  isAngled indicates whether the file reference is
  /// for system #include's or not (i.e. using <> instead of "").
  const FileEntry *LookupFile(const char *FilenameStart,const char *FilenameEnd,
                              bool isAngled, const DirectoryLookup *FromDir,
                              const DirectoryLookup *&CurDir);
    
  //===--------------------------------------------------------------------===//
  /// Handle*Directive - implement the various preprocessor directives.  These
  /// should side-effect the current preprocessor object so that the next call
  /// to Lex() will return the appropriate token next.
  
  void HandleUserDiagnosticDirective(LexerToken &Tok, bool isWarning);
  void HandleIdentSCCSDirective(LexerToken &Tok);
  
  // File inclusion.
  void HandleIncludeDirective(LexerToken &Tok,
                              const DirectoryLookup *LookupFrom = 0,
                              bool isImport = false);
  void HandleIncludeNextDirective(LexerToken &Tok);
  void HandleImportDirective(LexerToken &Tok);
  
  // Macro handling.
  void HandleDefineDirective(LexerToken &Tok, bool isTargetSpecific);
  void HandleUndefDirective(LexerToken &Tok);
  void HandleDefineOtherTargetDirective(LexerToken &Tok);
  // HandleAssertDirective(LexerToken &Tok);
  // HandleUnassertDirective(LexerToken &Tok);
  
  // Conditional Inclusion.
  void HandleIfdefDirective(LexerToken &Tok, bool isIfndef,
                            bool ReadAnyTokensBeforeDirective);
  void HandleIfDirective(LexerToken &Tok, bool ReadAnyTokensBeforeDirective);
  void HandleEndifDirective(LexerToken &Tok);
  void HandleElseDirective(LexerToken &Tok);
  void HandleElifDirective(LexerToken &Tok);
  
  // Pragmas.
  void HandlePragmaDirective();
public:
  void HandlePragmaOnce(LexerToken &OnceTok);
  void HandlePragmaPoison(LexerToken &PoisonTok);
  void HandlePragmaSystemHeader(LexerToken &SysHeaderTok);
  void HandlePragmaDependency(LexerToken &DependencyTok);
};

}  // end namespace clang
}  // end namespace llvm

#endif
