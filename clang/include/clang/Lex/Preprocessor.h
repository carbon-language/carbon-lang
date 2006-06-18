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
  
  // #include search path information.  Requests for #include "x" search the
  /// directory of the #including file first, then each directory in SearchDirs
  /// consequtively. Requests for <x> search the current dir first, then each
  /// directory in SearchDirs, starting at SystemDirIdx, consequtively.  If
  /// NoCurDirSearch is true, then the check for the file in the current
  /// directory is supressed.
  std::vector<DirectoryLookup> SearchDirs;
  unsigned SystemDirIdx;
  bool NoCurDirSearch;
  
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
  
  /// CurLexer - This is the current top of the stack that we're lexing from if
  /// not expanding a macro.  One of CurLexer and CurMacroExpander must be null.
  ///
  Lexer *CurLexer;
  
  /// CurDirLookup - The next DirectoryLookup structure to search for a file if
  /// CurLexer is non-null.  This allows us to implement #include_next.
  const DirectoryLookup *CurNextDirLookup;
  
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
    // isImport - True if this is a #import'd or #pragma once file.
    bool isImport;
    
    // NumIncludes - This is the number of times the file has been included
    // already.
    unsigned short NumIncludes;
    
    PerFileInfo() : isImport(false), NumIncludes(0) {}
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
  IdentifierTokenInfo *getIdentifierInfo(const std::string &Name) {
    // If we are in a "#if 0" block, don't bother lookup up identifiers.
    if (SkippingContents) return 0;
    return &IdentifierInfo.get(Name);
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
    
    IdentifierTokenInfo &Info = *getIdentifierInfo(Keyword);
    Info.setTokenID(TokenCode);
    Info.setIsExtensionToken(Flags == 1);
  }
  
  /// AddKeywords - Add all keywords to the symbol table.
  ///
  void AddKeywords();

  /// LookupFile - Given a "foo" or <foo> reference, look up the indicated file,
  /// return null on failure.  isSystem indicates whether the file reference is
  /// for system #include's or not.  If successful, this returns 'UsedDir', the
  /// DirectoryLookup member the file was found in, or null if not applicable.
  /// If FromDir is non-null, the directory search should start with the entry
  /// after the indicated lookup.  This is used to implement #include_next.
  const FileEntry *LookupFile(const std::string &Filename, bool isSystem,
                              const DirectoryLookup *FromDir,
                              const DirectoryLookup *&NextDir);
  
  /// EnterSourceFile - Add a source file to the top of the include stack and
  /// start lexing tokens from it instead of the current buffer.
  void EnterSourceFile(unsigned CurFileID, const DirectoryLookup *Dir);

  /// EnterMacro - Add a Macro to the top of the include stack and start lexing
  /// tokens from it instead of the current buffer.  Return true on failure.
  bool EnterMacro(LexerToken &Identifier);
  
  
  /// Lex - To lex a token from the preprocessor, just pull a token from the
  /// current lexer or macro object.
  bool Lex(LexerToken &Result) {
    if (CurLexer)
      return CurLexer->Lex(Result);
    else
      return CurMacroExpander->Lex(Result);
  }
  
  /// LexUnexpandedToken - This is just like Lex, but this disables macro
  /// expansion of identifier tokens.
  bool LexUnexpandedToken(LexerToken &Result) {
    // Disable macro expansion.
    bool OldVal = DisableMacroExpansion;
    DisableMacroExpansion = true;
    // Lex the token.
    bool ResVal = Lex(Result);
    
    // Reenable it.
    DisableMacroExpansion = OldVal;
    return ResVal;
  }
  
  /// Diag - Forwarding function for diagnostics.  This emits a diagnostic at
  /// the specified LexerToken's location, translating the token's start
  /// position in the current buffer into a SourcePosition object for rendering.
  bool Diag(const LexerToken &Tok, unsigned DiagID, const std::string &Msg="");  
  bool Diag(SourceLocation Loc, unsigned DiagID, const std::string &Msg="");  
  
  void PrintStats();

  //===--------------------------------------------------------------------===//
  // Preprocessor callback methods.  These are invoked by a lexer as various
  // directives and events are found.

  /// HandleIdentifier - This callback is invoked when the lexer reads an
  /// identifier and has filled in the tokens IdentifierInfo member.  This
  /// callback potentially macro expands it or turns it into a named token (like
  /// 'for').
  bool HandleIdentifier(LexerToken &Identifier);

  /// HandleEndOfFile - This callback is invoked when the lexer hits the end of
  /// the current file.  This either returns the EOF token or pops a level off
  /// the include stack and keeps going.
  bool HandleEndOfFile(LexerToken &Result);
  
  /// HandleEndOfMacro - This callback is invoked when the lexer hits the end of
  /// the current macro.  This either returns the EOF token or pops a level off
  /// the include stack and keeps going.
  bool HandleEndOfMacro(LexerToken &Result);
  
  /// HandleDirective - This callback is invoked when the lexer sees a # token
  /// at the start of a line.  This consumes the directive, modifies the 
  /// lexer/preprocessor state, and advances the lexer(s) so that the next token
  /// read is the correct one.
  bool HandleDirective(LexerToken &Result);

private:
  /// getFileInfo - Return the PerFileInfo structure for the specified
  /// FileEntry.
  PerFileInfo &getFileInfo(const FileEntry *FE);

  /// DiscardUntilEndOfDirective - Read and discard all tokens remaining on the
  /// current line until the tok::eom token is found.
  bool DiscardUntilEndOfDirective();

  /// ReadMacroName - Lex and validate a macro name, which occurs after a
  /// #define or #undef.  This emits a diagnostic, sets the token kind to eom,
  /// and discards the rest of the macro line if the macro name is invalid.
  bool ReadMacroName(LexerToken &MacroNameTok);
  
  /// CheckEndOfDirective - Ensure that the next token is a tok::eom token.  If
  /// not, emit a diagnostic and consume up until the eom.
  bool CheckEndOfDirective(const char *Directive);
  
  /// SkipExcludedConditionalBlock - We just read a #if or related directive and
  /// decided that the subsequent tokens are in the #if'd out portion of the
  /// file.  Lex the rest of the file, until we see an #endif.  If
  /// FoundNonSkipPortion is true, then we have already emitted code for part of
  /// this #if directive, so #else/#elif blocks should never be entered. If
  /// FoundElse is false, then #else directives are ok, if not, then we have
  /// already seen one so a #else directive is a duplicate.  When this returns,
  /// the caller can lex the first valid token.
  bool SkipExcludedConditionalBlock(const char *IfTokenLoc,
                                    bool FoundNonSkipPortion, bool FoundElse);
  
  /// EvaluateDirectiveExpression - Evaluate an integer constant expression that
  /// may occur after a #if or #elif directive.  Sets Result to the result of
  /// the expression.  Returns false normally, true if lexing must be aborted.
  bool EvaluateDirectiveExpression(bool &Result);
  /// EvaluateValue - Used to implement EvaluateDirectiveExpression,
  /// see PPExpressions.cpp.
  bool EvaluateValue(int &Result, LexerToken &PeekTok, bool &StopParse);
  /// EvaluateDirectiveSubExpr - Used to implement EvaluateDirectiveExpression,
  /// see PPExpressions.cpp.
  bool EvaluateDirectiveSubExpr(int &LHS, unsigned MinPrec,
                                LexerToken &PeekTok, bool &StopParse);
  
  //===--------------------------------------------------------------------===//
  /// Handle*Directive - implement the various preprocessor directives.  These
  /// should side-effect the current preprocessor object so that the next call
  /// to Lex() will return the appropriate token next.  If a fatal error occurs
  /// return true, otherwise return false.
  
  bool HandleUserDiagnosticDirective(LexerToken &Result, bool isWarning);
  
  // File inclusion.
  bool HandleIncludeDirective(LexerToken &Result,
                              const DirectoryLookup *LookupFrom = 0,
                              bool isImport = false);
  bool HandleIncludeNextDirective(LexerToken &Result);
  bool HandleImportDirective(LexerToken &Result);
  
  // Macro handling.
  bool HandleDefineDirective(LexerToken &Result);
  bool HandleUndefDirective(LexerToken &Result);
  
  // Conditional Inclusion.
  bool HandleIfdefDirective(LexerToken &Result, bool isIfndef);
  bool HandleIfDirective(LexerToken &Result);
  bool HandleEndifDirective(LexerToken &Result);
  bool HandleElseDirective(LexerToken &Result);
  bool HandleElifDirective(LexerToken &Result);
};

}  // end namespace clang
}  // end namespace llvm

#endif
