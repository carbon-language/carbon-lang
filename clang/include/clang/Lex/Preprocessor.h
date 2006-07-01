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

#ifndef LLVM_CLANG_PREPROCESSOR_H
#define LLVM_CLANG_PREPROCESSOR_H

#include "clang/Lex/Lexer.h"
#include "clang/Lex/MacroExpander.h"
#include "clang/Lex/IdentifierTable.h"
#include "clang/Basic/SourceLocation.h"

namespace llvm {
namespace clang {
  
class Lexer;
class LexerToken;
class SourceManager;
class FileManager;
class DirectoryEntry;
class FileEntry;
class PragmaNamespace;
class PragmaHandler;
class ScratchBuffer;

/// DirectoryLookup - This class is used to specify the search order for
/// directories in #include directives.
class DirectoryLookup {
public:
  enum DirType {
    NormalHeaderDir,
    SystemHeaderDir,
    ExternCSystemHeaderDir
  };
private:  
  /// Dir - This is the actual directory that we're referring to.
  ///
  const DirectoryEntry *Dir;
  
  /// DirCharacteristic - The type of directory this is, one of the DirType enum
  /// values.
  DirType DirCharacteristic : 2;
  
  /// UserSupplied - True if this is a user-supplied directory.
  ///
  bool UserSupplied;
public:
  DirectoryLookup(const DirectoryEntry *dir, DirType DT, bool isUser)
    : Dir(dir), DirCharacteristic(DT), UserSupplied(isUser) {}
    
  /// getDir - Return the directory that this entry refers to.
  ///
  const DirectoryEntry *getDir() const { return Dir; }
  
  /// DirCharacteristic - The type of directory this is, one of the DirType enum
  /// values.
  DirType getDirCharacteristic() const { return DirCharacteristic; }
  
  /// isUserSupplied - True if this is a user-supplied directory.
  ///
  bool isUserSupplied() const { return UserSupplied; }
};

/// Preprocessor - This object forms engages in a tight little dance to
/// efficiently preprocess tokens.  Lexers know only about tokens within a
/// single source file, and don't know anything about preprocessor-level issues
/// like the #include stack, token expansion, etc.
///
class Preprocessor {
  Diagnostic &Diags;
  const LangOptions &Features;
  FileManager   &FileMgr;
  SourceManager &SourceMgr;
  ScratchBuffer *ScratchBuf;
  
  // #include search path information.  Requests for #include "x" search the
  /// directory of the #including file first, then each directory in SearchDirs
  /// consequtively. Requests for <x> search the current dir first, then each
  /// directory in SearchDirs, starting at SystemDirIdx, consequtively.  If
  /// NoCurDirSearch is true, then the check for the file in the current
  /// directory is supressed.
  std::vector<DirectoryLookup> SearchDirs;
  unsigned SystemDirIdx;
  bool NoCurDirSearch;
  
  /// Identifiers for builtin macros.
  IdentifierTokenInfo *Ident__LINE__, *Ident__FILE__; // __LINE__, __FILE__
  IdentifierTokenInfo *Ident__DATE__, *Ident__TIME__; // __DATE__, __TIME__
  IdentifierTokenInfo *Ident__INCLUDE_LEVEL__;        // __INCLUDE_LEVEL__
  IdentifierTokenInfo *Ident__BASE_FILE__;            // __BASE_FILE__
  IdentifierTokenInfo *Ident__TIMESTAMP__;            // __TIMESTAMP__
  
  SourceLocation DATELoc, TIMELoc;
public:
  enum FileChangeReason {
    EnterFile, ExitFile, SystemHeaderPragma, RenameFile
  };
private:
  /// FileChangeHandler - This callback is invoked whenever a source file is
  /// entered or exited.  The SourceLocation indicates the new location, and
  /// EnteringFile indicates whether this is because we are entering a new
  /// #include'd file (when true) or whether we're exiting one because we ran
  /// off the end (when false).
  void (*FileChangeHandler)(SourceLocation Loc, FileChangeReason Reason,
                            DirectoryLookup::DirType FileType);
  
  enum {
    /// MaxIncludeStackDepth - Maximum depth of #includes.
    MaxAllowedIncludeStackDepth = 200
  };

  // State that changes while the preprocessor runs:
  bool DisableMacroExpansion;    // True if macro expansion is disabled.
  bool SkippingContents;         // True if in a #if 0 block.

  /// IdentifierInfo - This is mapping/lookup information for all identifiers in
  /// the program, including program keywords.
  IdentifierTable IdentifierInfo;
  
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
  
  /// IncludeStack - This keeps track of the stack of files currently #included,
  /// not counting CurLexer.
  struct IncludeStackInfo {
    Lexer *TheLexer;
    const DirectoryLookup *TheDirLookup;
    IncludeStackInfo(Lexer *L, const DirectoryLookup *D)
      : TheLexer(L), TheDirLookup(D) {
    }
  };
  std::vector<IncludeStackInfo> IncludeStack;
  
  /// CurMacroExpander - This is the current macro we are expanding, if we are
  /// expanding a macro.  One of CurLexer and CurMacroExpander must be null.
  MacroExpander *CurMacroExpander;
  
  /// MacroStack - This keeps track of the macros that are recursively being
  /// expanded.
  std::vector<MacroExpander*> MacroStack;
  
  
  /// PreFileInfo - The preprocessor keeps track of this information for each
  /// file that is #included.
  struct PerFileInfo {
    /// isImport - True if this is a #import'd or #pragma once file.
    bool isImport : 1;
    
    /// DirInfo - Keep track of whether this is a system header, and if so,
    /// whether it is C++ clean or not.  This can be set by the include paths or
    /// by #pragma gcc system_header.
    DirectoryLookup::DirType DirInfo : 2;
    
    /// NumIncludes - This is the number of times the file has been included
    /// already.
    unsigned short NumIncludes;
    
    PerFileInfo() : isImport(false), DirInfo(DirectoryLookup::NormalHeaderDir),
                    NumIncludes(0) {}
  };
  
  /// FileInfo - This contains all of the preprocessor-specific data about files
  /// that are included.  The vector is indexed by the FileEntry's UID.
  ///
  std::vector<PerFileInfo> FileInfo;
  
  // Various statistics we track for performance analysis.
  unsigned NumDirectives, NumIncluded, NumDefined, NumUndefined, NumPragma;
  unsigned NumIf, NumElse, NumEndif;
  unsigned NumEnteredSourceFiles, MaxIncludeStackDepth;
  unsigned NumMacroExpanded, NumFastMacroExpanded, MaxMacroStackDepth;
  unsigned NumSkipped;
public:
  Preprocessor(Diagnostic &diags, const LangOptions &opts, FileManager &FM,
               SourceManager &SM);
  ~Preprocessor();

  Diagnostic &getDiagnostics() const { return Diags; }
  const LangOptions &getLangOptions() const { return Features; }
  FileManager &getFileManager() const { return FileMgr; }
  SourceManager &getSourceManager() const { return SourceMgr; }

  IdentifierTable &getIdentifierTable() { return IdentifierInfo; }

  /// isSkipping - Return true if we're lexing a '#if 0' block.  This causes
  /// lexer errors/warnings to get ignored.
  bool isSkipping() const { return SkippingContents; }
  
  /// isCurrentLexer - Return true if we are lexing directly from the specified
  /// lexer.
  bool isCurrentLexer(const Lexer *L) const {
    return CurLexer == L;
  }
  
  /// SetSearchPaths - Interface for setting the file search paths.
  ///
  void SetSearchPaths(const std::vector<DirectoryLookup> &dirs,
                      unsigned systemDirIdx, bool noCurDirSearch) {
    SearchDirs = dirs;
    SystemDirIdx = systemDirIdx;
    NoCurDirSearch = noCurDirSearch;
  }
  
  /// setFileChangeHandler - Set the callback invoked whenever a source file is
  /// entered or exited.  The SourceLocation indicates the new location, and
  /// EnteringFile indicates whether this is because we are entering a new
  /// #include'd file (when true) or whether we're exiting one because we ran
  /// off the end (when false).
  void setFileChangeHandler(void (*Handler)(SourceLocation, FileChangeReason,
                                            DirectoryLookup::DirType)) {
    FileChangeHandler = Handler;
  }
  
  
  /// getIdentifierInfo - Return information about the specified preprocessor
  /// identifier token.  The version of this method that takes two character
  /// pointers is preferred unless the identifier is already available as a
  /// string (this avoids allocation and copying of memory to construct an
  /// std::string).
  IdentifierTokenInfo *getIdentifierInfo(const char *NameStart,
                                         const char *NameEnd) {
    // If we are in a "#if 0" block, don't bother lookup up identifiers.
    if (SkippingContents) return 0;
    return &IdentifierInfo.get(NameStart, NameEnd);
  }
  IdentifierTokenInfo *getIdentifierInfo(const char *NameStr) {
    return getIdentifierInfo(NameStr, NameStr+strlen(NameStr));
  }
  
  /// AddKeyword - This method is used to associate a token ID with specific
  /// identifiers because they are language keywords.  This causes the lexer to
  /// automatically map matching identifiers to specialized token codes.
  ///
  /// The C90/C99/CPP flags are set to 0 if the token should be enabled in the
  /// specified langauge, set to 1 if it is an extension in the specified
  /// language, and set to 2 if disabled in the specified language.
  void AddKeyword(const std::string &Keyword, tok::TokenKind TokenCode,
                  int C90, int C99, int CPP) {
    int Flags = Features.CPlusPlus ? CPP : (Features.C99 ? C99 : C90);
    
    // Don't add this keyword if disabled in this language or if an extension
    // and extensions are disabled.
    if (Flags+Features.NoExtensions >= 2) return;
    
    const char *Str = &Keyword[0];
    IdentifierTokenInfo &Info = *getIdentifierInfo(Str, Str+Keyword.size());
    Info.setTokenID(TokenCode);
    Info.setIsExtensionToken(Flags == 1);
  }
  
  /// AddKeywords - Add all keywords to the symbol table.
  ///
  void AddKeywords();
  
  
  /// AddPragmaHandler - Add the specified pragma handler to the preprocessor.
  /// If 'Namespace' is non-null, then it is a token required to exist on the
  /// pragma line before the pragma string starts, e.g. "STDC" or "GCC".
  void AddPragmaHandler(const char *Namespace, PragmaHandler *Handler);

  /// LookupFile - Given a "foo" or <foo> reference, look up the indicated file,
  /// return null on failure.  isAngled indicates whether the file reference is
  /// a <> reference.  If successful, this returns 'UsedDir', the
  /// DirectoryLookup member the file was found in, or null if not applicable.
  /// If CurDir is non-null, the file was found in the specified directory
  /// search location.  This is used to implement #include_next.
  const FileEntry *LookupFile(const std::string &Filename, bool isAngled,
                              const DirectoryLookup *FromDir,
                              const DirectoryLookup *&CurDir);
  
  /// EnterSourceFile - Add a source file to the top of the include stack and
  /// start lexing tokens from it instead of the current buffer.
  void EnterSourceFile(unsigned CurFileID, const DirectoryLookup *Dir);

  /// EnterMacro - Add a Macro to the top of the include stack and start lexing
  /// tokens from it instead of the current buffer.
  void EnterMacro(LexerToken &Identifier);
  
  
  /// Lex - To lex a token from the preprocessor, just pull a token from the
  /// current lexer or macro object.
  void Lex(LexerToken &Result) {
    if (CurLexer)
      CurLexer->Lex(Result);
    else
      CurMacroExpander->Lex(Result);
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
  void Diag(const LexerToken &Tok, unsigned DiagID, const std::string &Msg="");  
  void Diag(SourceLocation Loc, unsigned DiagID, const std::string &Msg="");  
  
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
  unsigned getSpelling(const LexerToken &Tok, char *Buffer) const;
  
  /// DumpToken - Print the token to stderr, used for debugging.
  ///
  void DumpToken(const LexerToken &Tok, bool DumpFlags = false) const;
  void DumpMacro(const MacroInfo &MI) const;
  
  void PrintStats();

  //===--------------------------------------------------------------------===//
  // Preprocessor callback methods.  These are invoked by a lexer as various
  // directives and events are found.

  /// HandleIdentifier - This callback is invoked when the lexer reads an
  /// identifier and has filled in the tokens IdentifierInfo member.  This
  /// callback potentially macro expands it or turns it into a named token (like
  /// 'for').
  void HandleIdentifier(LexerToken &Identifier);

  
  /// HandleEndOfFile - This callback is invoked when the lexer hits the end of
  /// the current file.  This either returns the EOF token or pops a level off
  /// the include stack and keeps going.
  void HandleEndOfFile(LexerToken &Result, bool isEndOfMacro = false);
  
  /// HandleEndOfMacro - This callback is invoked when the lexer hits the end of
  /// the current macro line.
  void HandleEndOfMacro(LexerToken &Result);
  
  /// HandleDirective - This callback is invoked when the lexer sees a # token
  /// at the start of a line.  This consumes the directive, modifies the 
  /// lexer/preprocessor state, and advances the lexer(s) so that the next token
  /// read is the correct one.
  void HandleDirective(LexerToken &Result);

  /// CheckEndOfDirective - Ensure that the next token is a tok::eom token.  If
  /// not, emit a diagnostic and consume up until the eom.
  void CheckEndOfDirective(const char *Directive);
private:
  /// getFileInfo - Return the PerFileInfo structure for the specified
  /// FileEntry.
  PerFileInfo &getFileInfo(const FileEntry *FE);

  /// DiscardUntilEndOfDirective - Read and discard all tokens remaining on the
  /// current line until the tok::eom token is found.
  void DiscardUntilEndOfDirective();

  /// ReadMacroName - Lex and validate a macro name, which occurs after a
  /// #define or #undef.  This emits a diagnostic, sets the token kind to eom,
  /// and discards the rest of the macro line if the macro name is invalid.
  void ReadMacroName(LexerToken &MacroNameTok);
  
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
  /// may occur after a #if or #elif directive.  Sets Result to the result of
  /// the expression.
  bool EvaluateDirectiveExpression();
  /// EvaluateValue/EvaluateDirectiveSubExpr - Used to implement
  /// EvaluateDirectiveExpression, see PPExpressions.cpp.
  bool EvaluateValue(int &Result, LexerToken &PeekTok);
  bool EvaluateDirectiveSubExpr(int &LHS, unsigned MinPrec,
                                LexerToken &PeekTok);
  
  /// RegisterBuiltinPragmas - Install the standard preprocessor pragmas:
  /// #pragma GCC poison/system_header/dependency and #pragma once.
  void RegisterBuiltinPragmas();
  
  /// RegisterBuiltinMacros - Register builtin macros, such as __LINE__ with the
  /// identifier table.
  void RegisterBuiltinMacros();
  IdentifierTokenInfo *RegisterBuiltinMacro(const char *Name);
  
  /// HandleMacroExpandedIdentifier - If an identifier token is read that is to
  /// be expanded as a macro, handle it and return the next token as 'Tok'.
  void HandleMacroExpandedIdentifier(LexerToken &Tok, MacroInfo *MI);

  /// ExpandBuiltinMacro - If an identifier token is read that is to be expanded
  /// as a builtin macro, handle it and return the next token as 'Tok'.
  void ExpandBuiltinMacro(LexerToken &Tok, MacroInfo *MI);
  
  //===--------------------------------------------------------------------===//
  /// Handle*Directive - implement the various preprocessor directives.  These
  /// should side-effect the current preprocessor object so that the next call
  /// to Lex() will return the appropriate token next.
  
  void HandleUserDiagnosticDirective(LexerToken &Result, bool isWarning);
  
  // File inclusion.
  void HandleIncludeDirective(LexerToken &Tok,
                              const DirectoryLookup *LookupFrom = 0,
                              bool isImport = false);
  void HandleIncludeNextDirective(LexerToken &Tok);
  void HandleImportDirective(LexerToken &Tok);
  
  // Macro handling.
  void HandleDefineDirective(LexerToken &Tok);
  void HandleUndefDirective(LexerToken &Tok);
  // HandleAssertDirective(LexerToken &Tok);
  // HandleUnassertDirective(LexerToken &Tok);
  
  // Conditional Inclusion.
  void HandleIfdefDirective(LexerToken &Tok, bool isIfndef);
  void HandleIfDirective(LexerToken &Tok);
  void HandleEndifDirective(LexerToken &Tok);
  void HandleElseDirective(LexerToken &Tok);
  void HandleElifDirective(LexerToken &Tok);
  
  // Pragmas.
  void HandlePragmaDirective(LexerToken &Result);
public:
  void HandlePragmaOnce(LexerToken &OnceTok);
  void HandlePragmaPoison(LexerToken &PoisonTok);
  void HandlePragmaSystemHeader(LexerToken &SysHeaderTok);
  void HandlePragmaDependency(LexerToken &DependencyTok);
};

}  // end namespace clang
}  // end namespace llvm

#endif
