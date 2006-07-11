//===--- Preprocess.cpp - C Language Family Preprocessor Implementation ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the Preprocessor interface.
//
//===----------------------------------------------------------------------===//
//
// Options to support:
//   -H       - Print the name of each header file used.
//   -C -CC   - Do not discard comments for cpp.
//   -d[MDNI] - Dump various things.
//   -fworking-directory - #line's with preprocessor's working dir.
//   -fpreprocessed
//   -dependency-file,-M,-MM,-MF,-MG,-MP,-MT,-MQ,-MD,-MMD
//   -W*
//   -w
//
// Messages to emit:
//   "Multiple include guards may be useful for:\n"
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/Pragma.h"
#include "clang/Lex/ScratchBuffer.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include <iostream>
using namespace llvm;
using namespace clang;

//===----------------------------------------------------------------------===//

Preprocessor::Preprocessor(Diagnostic &diags, const LangOptions &opts, 
                           FileManager &FM, SourceManager &SM) 
  : Diags(diags), Features(opts), FileMgr(FM), SourceMgr(SM),
    SystemDirIdx(0), NoCurDirSearch(false),
    CurLexer(0), CurDirLookup(0), CurMacroExpander(0) {
  ScratchBuf = new ScratchBuffer(SourceMgr);
      
  // Clear stats.
  NumDirectives = NumIncluded = NumDefined = NumUndefined = NumPragma = 0;
  NumIf = NumElse = NumEndif = 0;
  NumEnteredSourceFiles = 0;
  NumMacroExpanded = NumFnMacroExpanded = NumBuiltinMacroExpanded = 0;
  NumFastMacroExpanded = 0;
  MaxIncludeStackDepth = 0; NumMultiIncludeFileOptzn = 0;
  NumSkipped = 0;
    
  // Macro expansion is enabled.
  DisableMacroExpansion = false;
  InMacroFormalArgs = false;

  // There is no file-change handler yet.
  FileChangeHandler = 0;
  IdentHandler = 0;
  
  // "Poison" __VA_ARGS__, which can only appear in the expansion of a macro.
  // This gets unpoisoned where it is allowed.
  (Ident__VA_ARGS__ = getIdentifierInfo("__VA_ARGS__"))->setIsPoisoned();
  
  // Initialize the pragma handlers.
  PragmaHandlers = new PragmaNamespace(0);
  RegisterBuiltinPragmas();
  
  // Initialize builtin macros like __LINE__ and friends.
  RegisterBuiltinMacros();
}

Preprocessor::~Preprocessor() {
  // Free any active lexers.
  delete CurLexer;
  
  while (!IncludeMacroStack.empty()) {
    delete IncludeMacroStack.back().TheLexer;
    delete IncludeMacroStack.back().TheMacroExpander;
    IncludeMacroStack.pop_back();
  }
  
  // Release pragma information.
  delete PragmaHandlers;

  // Delete the scratch buffer info.
  delete ScratchBuf;
}

/// getFileInfo - Return the PerFileInfo structure for the specified
/// FileEntry.
Preprocessor::PerFileInfo &Preprocessor::getFileInfo(const FileEntry *FE) {
  if (FE->getUID() >= FileInfo.size())
    FileInfo.resize(FE->getUID()+1);
  return FileInfo[FE->getUID()];
}  


/// AddKeywords - Add all keywords to the symbol table.
///
void Preprocessor::AddKeywords() {
  enum {
    C90Shift = 0,
    EXTC90   = 1 << C90Shift,
    NOTC90   = 2 << C90Shift,
    C99Shift = 2,
    EXTC99   = 1 << C99Shift,
    NOTC99   = 2 << C99Shift,
    CPPShift = 4,
    EXTCPP   = 1 << CPPShift,
    NOTCPP   = 2 << CPPShift,
    Mask     = 3
  };

  // Add keywords and tokens for the current language.
#define KEYWORD(NAME, FLAGS) \
  AddKeyword(#NAME+1, tok::kw##NAME,     \
             (FLAGS >> C90Shift) & Mask, \
             (FLAGS >> C99Shift) & Mask, \
             (FLAGS >> CPPShift) & Mask);
#define ALIAS(NAME, TOK) \
  AddKeyword(NAME, tok::kw_ ## TOK, 0, 0, 0);
#include "clang/Basic/TokenKinds.def"
}

/// Diag - Forwarding function for diagnostics.  This emits a diagnostic at
/// the specified LexerToken's location, translating the token's start
/// position in the current buffer into a SourcePosition object for rendering.
void Preprocessor::Diag(SourceLocation Loc, unsigned DiagID, 
                        const std::string &Msg) {
  // If we are in a '#if 0' block, don't emit any diagnostics for notes,
  // warnings or extensions.
  if (isSkipping() && Diagnostic::isNoteWarningOrExtension(DiagID))
    return;
  
  Diags.Report(Loc, DiagID, Msg);
}

void Preprocessor::DumpToken(const LexerToken &Tok, bool DumpFlags) const {
  std::cerr << tok::getTokenName(Tok.getKind()) << " '"
            << getSpelling(Tok) << "'";
  
  if (!DumpFlags) return;
  std::cerr << "\t";
  if (Tok.isAtStartOfLine())
    std::cerr << " [StartOfLine]";
  if (Tok.hasLeadingSpace())
    std::cerr << " [LeadingSpace]";
  if (Tok.needsCleaning()) {
    const char *Start = SourceMgr.getCharacterData(Tok.getLocation());
    std::cerr << " [UnClean='" << std::string(Start, Start+Tok.getLength())
              << "']";
  }
}

void Preprocessor::DumpMacro(const MacroInfo &MI) const {
  std::cerr << "MACRO: ";
  for (unsigned i = 0, e = MI.getNumTokens(); i != e; ++i) {
    DumpToken(MI.getReplacementToken(i));
    std::cerr << "  ";
  }
  std::cerr << "\n";
}

void Preprocessor::PrintStats() {
  std::cerr << "\n*** Preprocessor Stats:\n";
  std::cerr << FileInfo.size() << " files tracked.\n";
  unsigned NumOnceOnlyFiles = 0, MaxNumIncludes = 0, NumSingleIncludedFiles = 0;
  for (unsigned i = 0, e = FileInfo.size(); i != e; ++i) {
    NumOnceOnlyFiles += FileInfo[i].isImport;
    if (MaxNumIncludes < FileInfo[i].NumIncludes)
      MaxNumIncludes = FileInfo[i].NumIncludes;
    NumSingleIncludedFiles += FileInfo[i].NumIncludes == 1;
  }
  std::cerr << "  " << NumOnceOnlyFiles << " #import/#pragma once files.\n";
  std::cerr << "  " << NumSingleIncludedFiles << " included exactly once.\n";
  std::cerr << "  " << MaxNumIncludes << " max times a file is included.\n";
  
  std::cerr << NumDirectives << " directives found:\n";
  std::cerr << "  " << NumDefined << " #define.\n";
  std::cerr << "  " << NumUndefined << " #undef.\n";
  std::cerr << "  " << NumIncluded << " #include/#include_next/#import.\n";
  std::cerr << "    " << NumMultiIncludeFileOptzn << " #includes skipped due to"
                      << " the multi-include optimization.\n";
  std::cerr << "    " << NumEnteredSourceFiles << " source files entered.\n";
  std::cerr << "    " << MaxIncludeStackDepth << " max include stack depth\n";
  std::cerr << "  " << NumIf << " #if/#ifndef/#ifdef.\n";
  std::cerr << "  " << NumElse << " #else/#elif.\n";
  std::cerr << "  " << NumEndif << " #endif.\n";
  std::cerr << "  " << NumPragma << " #pragma.\n";
  std::cerr << NumSkipped << " #if/#ifndef#ifdef regions skipped\n";

  std::cerr << NumMacroExpanded << "/" << NumFnMacroExpanded << "/"
            << NumBuiltinMacroExpanded << " obj/fn/builtin macros expanded, "
            << NumFastMacroExpanded << " on the fast path.\n";
}

//===----------------------------------------------------------------------===//
// Token Spelling
//===----------------------------------------------------------------------===//


/// getSpelling() - Return the 'spelling' of this token.  The spelling of a
/// token are the characters used to represent the token in the source file
/// after trigraph expansion and escaped-newline folding.  In particular, this
/// wants to get the true, uncanonicalized, spelling of things like digraphs
/// UCNs, etc.
std::string Preprocessor::getSpelling(const LexerToken &Tok) const {
  assert((int)Tok.getLength() >= 0 && "Token character range is bogus!");
  
  // If this token contains nothing interesting, return it directly.
  const char *TokStart = SourceMgr.getCharacterData(Tok.getLocation());
  if (!Tok.needsCleaning())
    return std::string(TokStart, TokStart+Tok.getLength());
  
  std::string Result;
  Result.reserve(Tok.getLength());
  
  // Otherwise, hard case, relex the characters into the string.
  for (const char *Ptr = TokStart, *End = TokStart+Tok.getLength();
       Ptr != End; ) {
    unsigned CharSize;
    Result.push_back(Lexer::getCharAndSizeNoWarn(Ptr, CharSize, Features));
    Ptr += CharSize;
  }
  assert(Result.size() != unsigned(Tok.getLength()) &&
         "NeedsCleaning flag set on something that didn't need cleaning!");
  return Result;
}

/// getSpelling - This method is used to get the spelling of a token into a
/// preallocated buffer, instead of as an std::string.  The caller is required
/// to allocate enough space for the token, which is guaranteed to be at least
/// Tok.getLength() bytes long.  The actual length of the token is returned.
///
/// Note that this method may do two possible things: it may either fill in
/// the buffer specified with characters, or it may *change the input pointer*
/// to point to a constant buffer with the data already in it (avoiding a
/// copy).  The caller is not allowed to modify the returned buffer pointer
/// if an internal buffer is returned.
unsigned Preprocessor::getSpelling(const LexerToken &Tok,
                                   const char *&Buffer) const {
  assert((int)Tok.getLength() >= 0 && "Token character range is bogus!");
  
  // If this token is an identifier, just return the string from the identifier
  // table, which is very quick.
  if (const IdentifierInfo *II = Tok.getIdentifierInfo()) {
    Buffer = II->getName();
    return Tok.getLength();
  }
  
  // Otherwise, compute the start of the token in the input lexer buffer.
  const char *TokStart = SourceMgr.getCharacterData(Tok.getLocation());

  // If this token contains nothing interesting, return it directly.
  if (!Tok.needsCleaning()) {
    Buffer = TokStart;
    return Tok.getLength();
  }
  // Otherwise, hard case, relex the characters into the string.
  char *OutBuf = const_cast<char*>(Buffer);
  for (const char *Ptr = TokStart, *End = TokStart+Tok.getLength();
       Ptr != End; ) {
    unsigned CharSize;
    *OutBuf++ = Lexer::getCharAndSizeNoWarn(Ptr, CharSize, Features);
    Ptr += CharSize;
  }
  assert(unsigned(OutBuf-Buffer) != Tok.getLength() &&
         "NeedsCleaning flag set on something that didn't need cleaning!");
  
  return OutBuf-Buffer;
}

//===----------------------------------------------------------------------===//
// Source File Location Methods.
//===----------------------------------------------------------------------===//


/// LookupFile - Given a "foo" or <foo> reference, look up the indicated file,
/// return null on failure.  isAngled indicates whether the file reference is
/// for system #include's or not (i.e. using <> instead of "").
const FileEntry *Preprocessor::LookupFile(const std::string &Filename, 
                                          bool isAngled,
                                          const DirectoryLookup *FromDir,
                                          const DirectoryLookup *&CurDir) {
  assert(CurLexer && "Cannot enter a #include inside a macro expansion!");
  CurDir = 0;
  
  // If 'Filename' is absolute, check to see if it exists and no searching.
  // FIXME: Portability.  This should be a sys::Path interface, this doesn't
  // handle things like C:\foo.txt right, nor win32 \\network\device\blah.
  if (Filename[0] == '/') {
    // If this was an #include_next "/absolute/file", fail.
    if (FromDir) return 0;

    // Otherwise, just return the file.
    return FileMgr.getFile(Filename);
  }
  
  // Step #0, unless disabled, check to see if the file is in the #includer's
  // directory.  This search is not done for <> headers.
  if (!isAngled && !FromDir && !NoCurDirSearch) {
    unsigned TheFileID = getCurrentFileLexer()->getCurFileID();
    const FileEntry *CurFE = SourceMgr.getFileEntryForFileID(TheFileID);
    if (CurFE) {
      // Concatenate the requested file onto the directory.
      // FIXME: Portability.  Should be in sys::Path.
      if (const FileEntry *FE = 
            FileMgr.getFile(CurFE->getDir()->getName()+"/"+Filename)) {
        if (CurDirLookup)
          CurDir = CurDirLookup;
        else
          CurDir = 0;
        
        // This file is a system header or C++ unfriendly if the old file is.
        getFileInfo(FE).DirInfo = getFileInfo(CurFE).DirInfo;
        return FE;
      }
    }
  }
  
  // If this is a system #include, ignore the user #include locs.
  unsigned i = isAngled ? SystemDirIdx : 0;

  // If this is a #include_next request, start searching after the directory the
  // file was found in.
  if (FromDir)
    i = FromDir-&SearchDirs[0];
  
  // Check each directory in sequence to see if it contains this file.
  for (; i != SearchDirs.size(); ++i) {
    // Concatenate the requested file onto the directory.
    // FIXME: Portability.  Adding file to dir should be in sys::Path.
    std::string SearchDir = SearchDirs[i].getDir()->getName()+"/"+Filename;
    if (const FileEntry *FE = FileMgr.getFile(SearchDir)) {
      CurDir = &SearchDirs[i];
      
      // This file is a system header or C++ unfriendly if the dir is.
      getFileInfo(FE).DirInfo = CurDir->getDirCharacteristic();
      return FE;
    }
  }
  
  // Otherwise, didn't find it.
  return 0;
}

/// isInPrimaryFile - Return true if we're in the top-level file, not in a
/// #include.
bool Preprocessor::isInPrimaryFile() const {
  if (CurLexer && !CurLexer->Is_PragmaLexer)
    return CurLexer->isMainFile();
  
  // If there are any stacked lexers, we're in a #include.
  for (unsigned i = 0, e = IncludeMacroStack.size(); i != e; ++i)
    if (IncludeMacroStack[i].TheLexer &&
        !IncludeMacroStack[i].TheLexer->Is_PragmaLexer)
      return IncludeMacroStack[i].TheLexer->isMainFile();
  return false;
}

/// getCurrentLexer - Return the current file lexer being lexed from.  Note
/// that this ignores any potentially active macro expansions and _Pragma
/// expansions going on at the time.
Lexer *Preprocessor::getCurrentFileLexer() const {
  if (CurLexer && !CurLexer->Is_PragmaLexer) return CurLexer;
  
  // Look for a stacked lexer.
  for (unsigned i = IncludeMacroStack.size(); i != 0; --i) {
    Lexer *L = IncludeMacroStack[i-1].TheLexer;
    if (L && !L->Is_PragmaLexer) // Ignore macro & _Pragma expansions.
      return L;
  }
  return 0;
}


/// EnterSourceFile - Add a source file to the top of the include stack and
/// start lexing tokens from it instead of the current buffer.  Return true
/// on failure.
void Preprocessor::EnterSourceFile(unsigned FileID,
                                   const DirectoryLookup *CurDir,
                                   bool isMainFile) {
  assert(CurMacroExpander == 0 && "Cannot #include a file inside a macro!");
  ++NumEnteredSourceFiles;
  
  if (MaxIncludeStackDepth < IncludeMacroStack.size())
    MaxIncludeStackDepth = IncludeMacroStack.size();

  const SourceBuffer *Buffer = SourceMgr.getBuffer(FileID);
  Lexer *TheLexer = new Lexer(Buffer, FileID, *this);
  if (isMainFile) TheLexer->setIsMainFile();
  EnterSourceFileWithLexer(TheLexer, CurDir);
}  
  
/// EnterSourceFile - Add a source file to the top of the include stack and
/// start lexing tokens from it instead of the current buffer.
void Preprocessor::EnterSourceFileWithLexer(Lexer *TheLexer, 
                                            const DirectoryLookup *CurDir) {
    
  // Add the current lexer to the include stack.
  if (CurLexer || CurMacroExpander)
    IncludeMacroStack.push_back(IncludeStackInfo(CurLexer, CurDirLookup,
                                                 CurMacroExpander));
  
  CurLexer = TheLexer;
  CurDirLookup = CurDir;
  CurMacroExpander = 0;
  
  // Notify the client, if desired, that we are in a new source file.
  if (FileChangeHandler && !CurLexer->Is_PragmaLexer) {
    DirectoryLookup::DirType FileType = DirectoryLookup::NormalHeaderDir;
    
    // Get the file entry for the current file.
    if (const FileEntry *FE = 
          SourceMgr.getFileEntryForFileID(CurLexer->getCurFileID()))
      FileType = getFileInfo(FE).DirInfo;
    
    FileChangeHandler(SourceLocation(CurLexer->getCurFileID(), 0),
                      EnterFile, FileType);
  }
}



/// EnterMacro - Add a Macro to the top of the include stack and start lexing
/// tokens from it instead of the current buffer.
void Preprocessor::EnterMacro(LexerToken &Tok, MacroFormalArgs *Formals) {
  IdentifierInfo *Identifier = Tok.getIdentifierInfo();
  MacroInfo &MI = *Identifier->getMacroInfo();
  IncludeMacroStack.push_back(IncludeStackInfo(CurLexer, CurDirLookup,
                                               CurMacroExpander));
  CurLexer     = 0;
  CurDirLookup = 0;
  
  // Mark the macro as currently disabled, so that it is not recursively
  // expanded.
  MI.DisableMacro();
  CurMacroExpander = new MacroExpander(Tok, Formals, *this);
}

//===----------------------------------------------------------------------===//
// Macro Expansion Handling.
//===----------------------------------------------------------------------===//

/// RegisterBuiltinMacro - Register the specified identifier in the identifier
/// table and mark it as a builtin macro to be expanded.
IdentifierInfo *Preprocessor::RegisterBuiltinMacro(const char *Name) {
  // Get the identifier.
  IdentifierInfo *Id = getIdentifierInfo(Name);
  
  // Mark it as being a macro that is builtin.
  MacroInfo *MI = new MacroInfo(SourceLocation());
  MI->setIsBuiltinMacro();
  Id->setMacroInfo(MI);
  return Id;
}


/// RegisterBuiltinMacros - Register builtin macros, such as __LINE__ with the
/// identifier table.
void Preprocessor::RegisterBuiltinMacros() {
  Ident__LINE__ = RegisterBuiltinMacro("__LINE__");
  Ident__FILE__ = RegisterBuiltinMacro("__FILE__");
  Ident__DATE__ = RegisterBuiltinMacro("__DATE__");
  Ident__TIME__ = RegisterBuiltinMacro("__TIME__");
  Ident_Pragma  = RegisterBuiltinMacro("_Pragma");
  
  // GCC Extensions.
  Ident__BASE_FILE__     = RegisterBuiltinMacro("__BASE_FILE__");
  Ident__INCLUDE_LEVEL__ = RegisterBuiltinMacro("__INCLUDE_LEVEL__");
  Ident__TIMESTAMP__     = RegisterBuiltinMacro("__TIMESTAMP__");
}

/// isTrivialSingleTokenExpansion - Return true if MI, which has a single token
/// in its expansion, currently expands to that token literally.
static bool isTrivialSingleTokenExpansion(const MacroInfo *MI,
                                          const IdentifierInfo *MacroIdent) {
  IdentifierInfo *II = MI->getReplacementToken(0).getIdentifierInfo();

  // If the token isn't an identifier, it's always literally expanded.
  if (II == 0) return true;
  
  // If the identifier is a macro, and if that macro is enabled, it may be
  // expanded so it's not a trivial expansion.
  if (II->getMacroInfo() && II->getMacroInfo()->isEnabled() &&
      // Fast expanding "#define X X" is ok, because X would be disabled.
      II != MacroIdent)
    return false;
  
  // If this is an object-like macro invocation, it is safe to trivially expand
  // it.
  if (MI->isObjectLike()) return true;

  // If this is a function-like macro invocation, it's safe to trivially expand
  // as long as the identifier is not a macro argument.
  for (MacroInfo::arg_iterator I = MI->arg_begin(), E = MI->arg_end();
       I != E; ++I)
    if (*I == II)
      return false;   // Identifier is a macro argument.
  return true;
}


/// isNextPPTokenLParen - Determine whether the next preprocessor token to be
/// lexed is a '('.  If so, consume the token and return true, if not, this
/// method should have no observable side-effect on the lexed tokens.
bool Preprocessor::isNextPPTokenLParen() {
  // Do some quick tests for rejection cases.
  unsigned Val;
  if (CurLexer)
    Val = CurLexer->isNextPPTokenLParen();
  else
    Val = CurMacroExpander->isNextTokenLParen();
  
  if (Val == 2) {
    // If we ran off the end of the lexer or macro expander, walk the include
    // stack, looking for whatever will return the next token.
    for (unsigned i = IncludeMacroStack.size(); Val == 2 && i != 0; --i) {
      IncludeStackInfo &Entry = IncludeMacroStack[i-1];
      if (Entry.TheLexer)
        Val = Entry.TheLexer->isNextPPTokenLParen();
      else
        Val = Entry.TheMacroExpander->isNextTokenLParen();
    }
  }

  // Okay, if we know that the token is a '(', lex it and return.  Otherwise we
  // have found something that isn't a '(' or we found the end of the
  // translation unit.  In either case, return false.
  if (Val != 1)
    return false;
  
  LexerToken Tok;
  LexUnexpandedToken(Tok);
  assert(Tok.getKind() == tok::l_paren && "Error computing l-paren-ness?");
  return true;
}

/// HandleMacroExpandedIdentifier - If an identifier token is read that is to be
/// expanded as a macro, handle it and return the next token as 'Identifier'.
bool Preprocessor::HandleMacroExpandedIdentifier(LexerToken &Identifier, 
                                                 MacroInfo *MI) {
  
  // If this is a builtin macro, like __LINE__ or _Pragma, handle it specially.
  if (MI->isBuiltinMacro()) {
    ExpandBuiltinMacro(Identifier);
    return false;
  }
  
  /// FormalArgs - If this is a function-like macro expansion, this contains,
  /// for each macro argument, the list of tokens that were provided to the
  /// invocation.
  MacroFormalArgs *FormalArgs = 0;
  
  // If this is a function-like macro, read the arguments.
  if (MI->isFunctionLike()) {
    // C99 6.10.3p10: If the preprocessing token immediately after the the macro
    // name isn't a '(', this macro should not be expanded.
    if (!isNextPPTokenLParen())
      return true;
    
    // Remember that we are now parsing the arguments to a macro invocation.
    // Preprocessor directives used inside macro arguments are not portable, and
    // this enables the warning.
    InMacroFormalArgs = true;
    FormalArgs = ReadFunctionLikeMacroFormalArgs(Identifier, MI);
    
    // Finished parsing args.
    InMacroFormalArgs = false;
    
    // If there was an error parsing the arguments, bail out.
    if (FormalArgs == 0) return false;
    
    ++NumFnMacroExpanded;
  } else {
    ++NumMacroExpanded;
  }
  
  // Notice that this macro has been used.
  MI->setIsUsed(true);
  
  // If we started lexing a macro, enter the macro expansion body.
  
  // If this macro expands to no tokens, don't bother to push it onto the
  // expansion stack, only to take it right back off.
  if (MI->getNumTokens() == 0) {
    // No need for formal arg info.
    delete FormalArgs;
    
    // Ignore this macro use, just return the next token in the current
    // buffer.
    bool HadLeadingSpace = Identifier.hasLeadingSpace();
    bool IsAtStartOfLine = Identifier.isAtStartOfLine();
    
    Lex(Identifier);
    
    // If the identifier isn't on some OTHER line, inherit the leading
    // whitespace/first-on-a-line property of this token.  This handles
    // stuff like "! XX," -> "! ," and "   XX," -> "    ,", when XX is
    // empty.
    if (!Identifier.isAtStartOfLine()) {
      if (IsAtStartOfLine) Identifier.SetFlag(LexerToken::StartOfLine);
      if (HadLeadingSpace) Identifier.SetFlag(LexerToken::LeadingSpace);
    }
    ++NumFastMacroExpanded;
    return false;
    
  } else if (MI->getNumTokens() == 1 &&
             isTrivialSingleTokenExpansion(MI, Identifier.getIdentifierInfo())){
    // Otherwise, if this macro expands into a single trivially-expanded
    // token: expand it now.  This handles common cases like 
    // "#define VAL 42".
    
    // Propagate the isAtStartOfLine/hasLeadingSpace markers of the macro
    // identifier to the expanded token.
    bool isAtStartOfLine = Identifier.isAtStartOfLine();
    bool hasLeadingSpace = Identifier.hasLeadingSpace();
    
    // Remember where the token is instantiated.
    SourceLocation InstantiateLoc = Identifier.getLocation();
    
    // Replace the result token.
    Identifier = MI->getReplacementToken(0);
    
    // Restore the StartOfLine/LeadingSpace markers.
    Identifier.SetFlagValue(LexerToken::StartOfLine , isAtStartOfLine);
    Identifier.SetFlagValue(LexerToken::LeadingSpace, hasLeadingSpace);
    
    // Update the tokens location to include both its logical and physical
    // locations.
    SourceLocation Loc =
      SourceMgr.getInstantiationLoc(Identifier.getLocation(), InstantiateLoc);
    Identifier.SetLocation(Loc);
    
    // Since this is not an identifier token, it can't be macro expanded, so
    // we're done.
    ++NumFastMacroExpanded;
    return false;
  }
  
  // Start expanding the macro.
  EnterMacro(Identifier, FormalArgs);
  
  // Now that the macro is at the top of the include stack, ask the
  // preprocessor to read the next token from it.
  Lex(Identifier);
  return false;
}

/// ReadFunctionLikeMacroFormalArgs - After reading "MACRO(", this method is
/// invoked to read all of the formal arguments specified for the macro
/// invocation.  This returns null on error.
MacroFormalArgs *Preprocessor::
ReadFunctionLikeMacroFormalArgs(LexerToken &MacroName, MacroInfo *MI) {
  // Use an auto_ptr here so that the MacroFormalArgs object is deleted on
  // all error paths.
  std::auto_ptr<MacroFormalArgs> Args(new MacroFormalArgs(MI));
  
  // The number of fixed arguments to parse.
  unsigned NumFixedArgsLeft = MI->getNumArgs();
  bool isVariadic = MI->isVariadic();
  
  // If this is a C99-style varargs macro invocation, add an extra expected
  // argument, which will catch all of the varargs formals in one argument.
  if (MI->isC99Varargs())
    ++NumFixedArgsLeft;
  
  // Outer loop, while there are more arguments, keep reading them.
  LexerToken Tok;
  Tok.SetKind(tok::comma);
  --NumFixedArgsLeft;  // Start reading the first arg.
  
  while (Tok.getKind() == tok::comma) {
    // ArgTokens - Build up a list of tokens that make up this argument.
    std::vector<LexerToken> ArgTokens;
    // C99 6.10.3p11: Keep track of the number of l_parens we have seen.
    unsigned NumParens = 0;

    while (1) {
      // Read arguments as unexpanded tokens.  This avoids issues, e.g., where
      // an argument value in a macro could expand to ',' or '(' or ')'.
      LexUnexpandedToken(Tok);
      
      if (Tok.getKind() == tok::eof) {
        Diag(MacroName, diag::err_unterm_macro_invoc);
        // Do not lose the EOF.  Return it to the client.
        MacroName = Tok;
        return 0;
      } else if (Tok.getKind() == tok::r_paren) {
        // If we found the ) token, the macro arg list is done.
        if (NumParens-- == 0)
          break;
      } else if (Tok.getKind() == tok::l_paren) {
        ++NumParens;
      } else if (Tok.getKind() == tok::comma && NumParens == 0) {
        // Comma ends this argument if there are more fixed arguments expected.
        if (NumFixedArgsLeft)
          break;
        
        // If this is not a variadic macro, too many formals were specified.
        if (!isVariadic) {
          // Emit the diagnostic at the macro name in case there is a missing ).
          // Emitting it at the , could be far away from the macro name.
          Diag(MacroName, diag::err_too_many_formals_in_macro_invoc);
          return 0;
        }
        // Otherwise, continue to add the tokens to this variable argument.
      }
  
      ArgTokens.push_back(Tok);
    }

    // Empty arguments are standard in C99 and supported as an extension in
    // other modes.
    if (ArgTokens.empty() && !Features.C99)
      Diag(Tok, diag::ext_empty_fnmacro_arg);
    
    // Remember the tokens that make up this argument.  This destroys ArgTokens.
    Args->addArgument(ArgTokens);
    --NumFixedArgsLeft;
  };
  
  // Okay, we either found the r_paren.  Check to see if we parsed too few
  // arguments.
  unsigned NumFormals = Args->getNumArguments();
  unsigned MinArgsExpected = MI->getNumArgs();
  
  // C99 expects us to pass at least one vararg arg (but as an extension, we
  // don't require this).  GNU-style varargs already include the 'rest' name in
  // the count.
  MinArgsExpected += MI->isC99Varargs();
  
  if (NumFormals < MinArgsExpected) {
    // There are several cases where too few arguments is ok, handle them now.
    if (NumFormals+1 == MinArgsExpected && MI->isVariadic()) {
      // Varargs where the named vararg parameter is missing: ok as extension.
      // #define A(x, ...)
      // A("blah")
      Diag(Tok, diag::ext_missing_varargs_arg);
    } else if (MI->getNumArgs() == 1) {
      // #define A(x)
      //   A()
      // is ok because it is an empty argument.  Add it explicitly.
      std::vector<LexerToken> ArgTokens;
      Args->addArgument(ArgTokens);
      
      // Empty arguments are standard in C99 and supported as an extension in
      // other modes.
      if (ArgTokens.empty() && !Features.C99)
        Diag(Tok, diag::ext_empty_fnmacro_arg);
    } else {
      // Otherwise, emit the error.
      Diag(Tok, diag::err_too_few_formals_in_macro_invoc);
      return 0;
    }
  }
  
  return Args.release();
}

/// ComputeDATE_TIME - Compute the current time, enter it into the specified
/// scratch buffer, then return DATELoc/TIMELoc locations with the position of
/// the identifier tokens inserted.
static void ComputeDATE_TIME(SourceLocation &DATELoc, SourceLocation &TIMELoc,
                             ScratchBuffer *ScratchBuf) {
  time_t TT = time(0);
  struct tm *TM = localtime(&TT);
  
  static const char * const Months[] = {
    "Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"
  };
  
  char TmpBuffer[100];
  sprintf(TmpBuffer, "\"%s %2d %4d\"", Months[TM->tm_mon], TM->tm_mday, 
          TM->tm_year+1900);
  DATELoc = ScratchBuf->getToken(TmpBuffer, strlen(TmpBuffer));

  sprintf(TmpBuffer, "\"%02d:%02d:%02d\"", TM->tm_hour, TM->tm_min, TM->tm_sec);
  TIMELoc = ScratchBuf->getToken(TmpBuffer, strlen(TmpBuffer));
}

/// ExpandBuiltinMacro - If an identifier token is read that is to be expanded
/// as a builtin macro, handle it and return the next token as 'Tok'.
void Preprocessor::ExpandBuiltinMacro(LexerToken &Tok) {
  // Figure out which token this is.
  IdentifierInfo *II = Tok.getIdentifierInfo();
  assert(II && "Can't be a macro without id info!");
  
  // If this is an _Pragma directive, expand it, invoke the pragma handler, then
  // lex the token after it.
  if (II == Ident_Pragma)
    return Handle_Pragma(Tok);
  
  ++NumBuiltinMacroExpanded;

  char TmpBuffer[100];

  // Set up the return result.
  Tok.SetIdentifierInfo(0);
  Tok.ClearFlag(LexerToken::NeedsCleaning);
  
  if (II == Ident__LINE__) {
    // __LINE__ expands to a simple numeric value.
    sprintf(TmpBuffer, "%u", SourceMgr.getLineNumber(Tok.getLocation()));
    unsigned Length = strlen(TmpBuffer);
    Tok.SetKind(tok::numeric_constant);
    Tok.SetLength(Length);
    Tok.SetLocation(ScratchBuf->getToken(TmpBuffer, Length, Tok.getLocation()));
  } else if (II == Ident__FILE__ || II == Ident__BASE_FILE__) {
    SourceLocation Loc = Tok.getLocation();
    if (II == Ident__BASE_FILE__) {
      Diag(Tok, diag::ext_pp_base_file);
      SourceLocation NextLoc = SourceMgr.getIncludeLoc(Loc.getFileID());
      while (NextLoc.getFileID() != 0) {
        Loc = NextLoc;
        NextLoc = SourceMgr.getIncludeLoc(Loc.getFileID());
      }
    }
    
    // Escape this filename.  Turn '\' -> '\\' '"' -> '\"'
    std::string FN = SourceMgr.getSourceName(Loc);
    FN = Lexer::Stringify(FN);
    Tok.SetKind(tok::string_literal);
    Tok.SetLength(FN.size());
    Tok.SetLocation(ScratchBuf->getToken(&FN[0], FN.size(), Tok.getLocation()));
  } else if (II == Ident__DATE__) {
    if (!DATELoc.isValid())
      ComputeDATE_TIME(DATELoc, TIMELoc, ScratchBuf);
    Tok.SetKind(tok::string_literal);
    Tok.SetLength(strlen("\"Mmm dd yyyy\""));
    Tok.SetLocation(SourceMgr.getInstantiationLoc(DATELoc, Tok.getLocation()));
  } else if (II == Ident__TIME__) {
    if (!TIMELoc.isValid())
      ComputeDATE_TIME(DATELoc, TIMELoc, ScratchBuf);
    Tok.SetKind(tok::string_literal);
    Tok.SetLength(strlen("\"hh:mm:ss\""));
    Tok.SetLocation(SourceMgr.getInstantiationLoc(TIMELoc, Tok.getLocation()));
  } else if (II == Ident__INCLUDE_LEVEL__) {
    Diag(Tok, diag::ext_pp_include_level);

    // Compute the include depth of this token.
    unsigned Depth = 0;
    SourceLocation Loc = SourceMgr.getIncludeLoc(Tok.getLocation().getFileID());
    for (; Loc.getFileID() != 0; ++Depth)
      Loc = SourceMgr.getIncludeLoc(Loc.getFileID());
    
    // __INCLUDE_LEVEL__ expands to a simple numeric value.
    sprintf(TmpBuffer, "%u", Depth);
    unsigned Length = strlen(TmpBuffer);
    Tok.SetKind(tok::numeric_constant);
    Tok.SetLength(Length);
    Tok.SetLocation(ScratchBuf->getToken(TmpBuffer, Length, Tok.getLocation()));
  } else if (II == Ident__TIMESTAMP__) {
    // MSVC, ICC, GCC, VisualAge C++ extension.  The generated string should be
    // of the form "Ddd Mmm dd hh::mm::ss yyyy", which is returned by asctime.
    Diag(Tok, diag::ext_pp_timestamp);

    // Get the file that we are lexing out of.  If we're currently lexing from
    // a macro, dig into the include stack.
    const FileEntry *CurFile = 0;
    Lexer *TheLexer = getCurrentFileLexer();
    
    if (TheLexer)
      CurFile = SourceMgr.getFileEntryForFileID(TheLexer->getCurFileID());
    
    // If this file is older than the file it depends on, emit a diagnostic.
    const char *Result;
    if (CurFile) {
      time_t TT = CurFile->getModificationTime();
      struct tm *TM = localtime(&TT);
      Result = asctime(TM);
    } else {
      Result = "??? ??? ?? ??:??:?? ????\n";
    }
    TmpBuffer[0] = '"';
    strcpy(TmpBuffer+1, Result);
    unsigned Len = strlen(TmpBuffer);
    TmpBuffer[Len-1] = '"';  // Replace the newline with a quote.
    Tok.SetKind(tok::string_literal);
    Tok.SetLength(Len);
    Tok.SetLocation(ScratchBuf->getToken(TmpBuffer, Len, Tok.getLocation()));
  } else {
    assert(0 && "Unknown identifier!");
  }  
}

namespace {
struct UnusedIdentifierReporter : public IdentifierVisitor {
  Preprocessor &PP;
  UnusedIdentifierReporter(Preprocessor &pp) : PP(pp) {}

  void VisitIdentifier(IdentifierInfo &II) const {
    if (II.getMacroInfo() && !II.getMacroInfo()->isUsed())
      PP.Diag(II.getMacroInfo()->getDefinitionLoc(), diag::pp_macro_not_used);
  }
};
}

//===----------------------------------------------------------------------===//
// Lexer Event Handling.
//===----------------------------------------------------------------------===//

/// LookUpIdentifierInfo - Given a tok::identifier token, look up the
/// identifier information for the token and install it into the token.
IdentifierInfo *Preprocessor::LookUpIdentifierInfo(LexerToken &Identifier,
                                                   const char *BufPtr) {
  assert(Identifier.getKind() == tok::identifier && "Not an identifier!");
  assert(Identifier.getIdentifierInfo() == 0 && "Identinfo already exists!");
  
  // Look up this token, see if it is a macro, or if it is a language keyword.
  IdentifierInfo *II;
  if (BufPtr && !Identifier.needsCleaning()) {
    // No cleaning needed, just use the characters from the lexed buffer.
    II = getIdentifierInfo(BufPtr, BufPtr+Identifier.getLength());
  } else {
    // Cleaning needed, alloca a buffer, clean into it, then use the buffer.
    const char *TmpBuf = (char*)alloca(Identifier.getLength());
    unsigned Size = getSpelling(Identifier, TmpBuf);
    II = getIdentifierInfo(TmpBuf, TmpBuf+Size);
  }
  Identifier.SetIdentifierInfo(II);
  return II;
}


/// HandleIdentifier - This callback is invoked when the lexer reads an
/// identifier.  This callback looks up the identifier in the map and/or
/// potentially macro expands it or turns it into a named token (like 'for').
void Preprocessor::HandleIdentifier(LexerToken &Identifier) {
  if (Identifier.getIdentifierInfo() == 0) {
    // If we are skipping tokens (because we are in a #if 0 block), there will
    // be no identifier info, just return the token.
    assert(isSkipping() && "Token isn't an identifier?");
    return;
  }
  IdentifierInfo &II = *Identifier.getIdentifierInfo();

  // If this identifier was poisoned, and if it was not produced from a macro
  // expansion, emit an error.
  if (II.isPoisoned() && CurLexer) {
    if (&II != Ident__VA_ARGS__)   // We warn about __VA_ARGS__ with poisoning.
      Diag(Identifier, diag::err_pp_used_poisoned_id);
    else
      Diag(Identifier, diag::ext_pp_bad_vaargs_use);
  }
  
  // If this is a macro to be expanded, do it.
  if (MacroInfo *MI = II.getMacroInfo())
    if (MI->isEnabled() && !DisableMacroExpansion)
      if (!HandleMacroExpandedIdentifier(Identifier, MI))
        return;

  // Change the kind of this identifier to the appropriate token kind, e.g.
  // turning "for" into a keyword.
  Identifier.SetKind(II.getTokenID());
    
  // If this is an extension token, diagnose its use.
  if (II.isExtensionToken()) Diag(Identifier, diag::ext_token_used);
}

/// HandleEndOfFile - This callback is invoked when the lexer hits the end of
/// the current file.  This either returns the EOF token or pops a level off
/// the include stack and keeps going.
void Preprocessor::HandleEndOfFile(LexerToken &Result, bool isEndOfMacro) {
  assert(!CurMacroExpander &&
         "Ending a file when currently in a macro!");
  
  // If we are in a #if 0 block skipping tokens, and we see the end of the file,
  // this is an error condition.  Just return the EOF token up to
  // SkipExcludedConditionalBlock.  The code that enabled skipping will issue
  // errors for the unterminated #if's on the conditional stack if it is
  // interested.
  if (isSkipping()) {
    Result.StartToken();
    CurLexer->BufferPtr = CurLexer->BufferEnd;
    CurLexer->FormTokenWithChars(Result, CurLexer->BufferEnd);
    Result.SetKind(tok::eof);
    return;
  }
  
  // See if this file had a controlling macro.
  if (CurLexer) {  // Not ending a macro, ignore it.
    if (const IdentifierInfo *ControllingMacro = 
          CurLexer->MIOpt.GetControllingMacroAtEndOfFile()) {
      // Okay, this has a controlling macro, remember in PerFileInfo.
      if (const FileEntry *FE = 
          SourceMgr.getFileEntryForFileID(CurLexer->getCurFileID()))
        getFileInfo(FE).ControllingMacro = ControllingMacro;
    }
  }
  
  // If this is a #include'd file, pop it off the include stack and continue
  // lexing the #includer file.
  if (!IncludeMacroStack.empty()) {
    // We're done with the #included file.
    delete CurLexer;
    CurLexer         = IncludeMacroStack.back().TheLexer;
    CurDirLookup     = IncludeMacroStack.back().TheDirLookup;
    CurMacroExpander = IncludeMacroStack.back().TheMacroExpander;
    IncludeMacroStack.pop_back();

    // Notify the client, if desired, that we are in a new source file.
    if (FileChangeHandler && !isEndOfMacro && CurLexer) {
      DirectoryLookup::DirType FileType = DirectoryLookup::NormalHeaderDir;
      
      // Get the file entry for the current file.
      if (const FileEntry *FE = 
            SourceMgr.getFileEntryForFileID(CurLexer->getCurFileID()))
        FileType = getFileInfo(FE).DirInfo;

      FileChangeHandler(CurLexer->getSourceLocation(CurLexer->BufferPtr),
                        ExitFile, FileType);
    }
    
    return Lex(Result);
  }
  
  Result.StartToken();
  CurLexer->BufferPtr = CurLexer->BufferEnd;
  CurLexer->FormTokenWithChars(Result, CurLexer->BufferEnd);
  Result.SetKind(tok::eof);
  
  // We're done with the #included file.
  delete CurLexer;
  CurLexer = 0;

  // This is the end of the top-level file.  If the diag::pp_macro_not_used
  // diagnostic is enabled, walk all of the identifiers, looking for macros that
  // have not been used.
  if (Diags.getDiagnosticLevel(diag::pp_macro_not_used) != Diagnostic::Ignored)
    Identifiers.VisitIdentifiers(UnusedIdentifierReporter(*this));
}

/// HandleEndOfMacro - This callback is invoked when the lexer hits the end of
/// the current macro expansion.
void Preprocessor::HandleEndOfMacro(LexerToken &Result) {
  assert(CurMacroExpander && !CurLexer &&
         "Ending a macro when currently in a #include file!");

  // Mark macro not ignored now that it is no longer being expanded.
  CurMacroExpander->getMacro().EnableMacro();
  delete CurMacroExpander;

  // Handle this like a #include file being popped off the stack.
  CurMacroExpander = 0;
  return HandleEndOfFile(Result, true);
}


//===----------------------------------------------------------------------===//
// Utility Methods for Preprocessor Directive Handling.
//===----------------------------------------------------------------------===//

/// DiscardUntilEndOfDirective - Read and discard all tokens remaining on the
/// current line until the tok::eom token is found.
void Preprocessor::DiscardUntilEndOfDirective() {
  LexerToken Tmp;
  do {
    LexUnexpandedToken(Tmp);
  } while (Tmp.getKind() != tok::eom);
}

/// ReadMacroName - Lex and validate a macro name, which occurs after a
/// #define or #undef.  This sets the token kind to eom and discards the rest
/// of the macro line if the macro name is invalid.  isDefineUndef is 1 if
/// this is due to a a #define, 2 if #undef directive, 0 if it is something
/// else (e.g. #ifdef).
void Preprocessor::ReadMacroName(LexerToken &MacroNameTok, char isDefineUndef) {
  // Read the token, don't allow macro expansion on it.
  LexUnexpandedToken(MacroNameTok);
  
  // Missing macro name?
  if (MacroNameTok.getKind() == tok::eom)
    return Diag(MacroNameTok, diag::err_pp_missing_macro_name);
  
  IdentifierInfo *II = MacroNameTok.getIdentifierInfo();
  if (II == 0) {
    Diag(MacroNameTok, diag::err_pp_macro_not_identifier);
    // Fall through on error.
  } else if (0) {
    // FIXME: C++.  Error if defining a C++ named operator.
    
  } else if (isDefineUndef && II->getName()[0] == 'd' &&      // defined
             !strcmp(II->getName()+1, "efined")) {
    // Error if defining "defined": C99 6.10.8.4.
    Diag(MacroNameTok, diag::err_defined_macro_name);
  } else if (isDefineUndef && II->getMacroInfo() &&
             II->getMacroInfo()->isBuiltinMacro()) {
    // Error if defining "__LINE__" and other builtins: C99 6.10.8.4.
    if (isDefineUndef == 1)
      Diag(MacroNameTok, diag::pp_redef_builtin_macro);
    else
      Diag(MacroNameTok, diag::pp_undef_builtin_macro);
  } else {
    // Okay, we got a good identifier node.  Return it.
    return;
  }
  
  // Invalid macro name, read and discard the rest of the line.  Then set the
  // token kind to tok::eom.
  MacroNameTok.SetKind(tok::eom);
  return DiscardUntilEndOfDirective();
}

/// CheckEndOfDirective - Ensure that the next token is a tok::eom token.  If
/// not, emit a diagnostic and consume up until the eom.
void Preprocessor::CheckEndOfDirective(const char *DirType) {
  LexerToken Tmp;
  Lex(Tmp);
  // There should be no tokens after the directive, but we allow them as an
  // extension.
  if (Tmp.getKind() != tok::eom) {
    Diag(Tmp, diag::ext_pp_extra_tokens_at_eol, DirType);
    DiscardUntilEndOfDirective();
  }
}



/// SkipExcludedConditionalBlock - We just read a #if or related directive and
/// decided that the subsequent tokens are in the #if'd out portion of the
/// file.  Lex the rest of the file, until we see an #endif.  If
/// FoundNonSkipPortion is true, then we have already emitted code for part of
/// this #if directive, so #else/#elif blocks should never be entered. If ElseOk
/// is true, then #else directives are ok, if not, then we have already seen one
/// so a #else directive is a duplicate.  When this returns, the caller can lex
/// the first valid token.
void Preprocessor::SkipExcludedConditionalBlock(SourceLocation IfTokenLoc,
                                                bool FoundNonSkipPortion,
                                                bool FoundElse) {
  ++NumSkipped;
  assert(CurMacroExpander == 0 && CurLexer &&
         "Lexing a macro, not a file?");

  CurLexer->pushConditionalLevel(IfTokenLoc, /*isSkipping*/false,
                                 FoundNonSkipPortion, FoundElse);
  
  // Enter raw mode to disable identifier lookup (and thus macro expansion),
  // disabling warnings, etc.
  CurLexer->LexingRawMode = true;
  LexerToken Tok;
  while (1) {
    CurLexer->Lex(Tok);
    
    // If this is the end of the buffer, we have an error.
    if (Tok.getKind() == tok::eof) {
      // Emit errors for each unterminated conditional on the stack, including
      // the current one.
      while (!CurLexer->ConditionalStack.empty()) {
        Diag(CurLexer->ConditionalStack.back().IfLoc,
             diag::err_pp_unterminated_conditional);
        CurLexer->ConditionalStack.pop_back();
      }  
      
      // Just return and let the caller lex after this #include.
      break;
    }
    
    // If this token is not a preprocessor directive, just skip it.
    if (Tok.getKind() != tok::hash || !Tok.isAtStartOfLine())
      continue;
      
    // We just parsed a # character at the start of a line, so we're in
    // directive mode.  Tell the lexer this so any newlines we see will be
    // converted into an EOM token (this terminates the macro).
    CurLexer->ParsingPreprocessorDirective = true;
    
    // Read the next token, the directive flavor.
    LexUnexpandedToken(Tok);
    
    // If this isn't an identifier directive (e.g. is "# 1\n" or "#\n", or
    // something bogus), skip it.
    if (Tok.getKind() != tok::identifier) {
      CurLexer->ParsingPreprocessorDirective = false;
      continue;
    }

    // If the first letter isn't i or e, it isn't intesting to us.  We know that
    // this is safe in the face of spelling differences, because there is no way
    // to spell an i/e in a strange way that is another letter.  Skipping this
    // allows us to avoid looking up the identifier info for #define/#undef and
    // other common directives.
    const char *RawCharData = SourceMgr.getCharacterData(Tok.getLocation());
    char FirstChar = RawCharData[0];
    if (FirstChar >= 'a' && FirstChar <= 'z' && 
        FirstChar != 'i' && FirstChar != 'e') {
      CurLexer->ParsingPreprocessorDirective = false;
      continue;
    }
    
    // Get the identifier name without trigraphs or embedded newlines.  Note
    // that we can't use Tok.getIdentifierInfo() because its lookup is disabled
    // when skipping.
    // TODO: could do this with zero copies in the no-clean case by using
    // strncmp below.
    char Directive[20];
    unsigned IdLen;
    if (!Tok.needsCleaning() && Tok.getLength() < 20) {
      IdLen = Tok.getLength();
      memcpy(Directive, RawCharData, IdLen);
      Directive[IdLen] = 0;
    } else {
      std::string DirectiveStr = getSpelling(Tok);
      IdLen = DirectiveStr.size();
      if (IdLen >= 20) {
        CurLexer->ParsingPreprocessorDirective = false;
        continue;
      }
      memcpy(Directive, &DirectiveStr[0], IdLen);
      Directive[IdLen] = 0;
    }
    
    if (FirstChar == 'i' && Directive[1] == 'f') {
      if ((IdLen == 2) ||   // "if"
          (IdLen == 5 && !strcmp(Directive+2, "def")) ||   // "ifdef"
          (IdLen == 6 && !strcmp(Directive+2, "ndef"))) {  // "ifndef"
        // We know the entire #if/#ifdef/#ifndef block will be skipped, don't
        // bother parsing the condition.
        DiscardUntilEndOfDirective();
        CurLexer->pushConditionalLevel(Tok.getLocation(), /*wasskipping*/true,
                                       /*foundnonskip*/false,
                                       /*fnddelse*/false);
      }
    } else if (FirstChar == 'e') {
      if (IdLen == 5 && !strcmp(Directive+1, "ndif")) {  // "endif"
        CheckEndOfDirective("#endif");
        PPConditionalInfo CondInfo;
        CondInfo.WasSkipping = true; // Silence bogus warning.
        bool InCond = CurLexer->popConditionalLevel(CondInfo);
        assert(!InCond && "Can't be skipping if not in a conditional!");
        
        // If we popped the outermost skipping block, we're done skipping!
        if (!CondInfo.WasSkipping)
          break;
      } else if (IdLen == 4 && !strcmp(Directive+1, "lse")) { // "else".
        // #else directive in a skipping conditional.  If not in some other
        // skipping conditional, and if #else hasn't already been seen, enter it
        // as a non-skipping conditional.
        CheckEndOfDirective("#else");
        PPConditionalInfo &CondInfo = CurLexer->peekConditionalLevel();
        
        // If this is a #else with a #else before it, report the error.
        if (CondInfo.FoundElse) Diag(Tok, diag::pp_err_else_after_else);
        
        // Note that we've seen a #else in this conditional.
        CondInfo.FoundElse = true;
        
        // If the conditional is at the top level, and the #if block wasn't
        // entered, enter the #else block now.
        if (!CondInfo.WasSkipping && !CondInfo.FoundNonSkip) {
          CondInfo.FoundNonSkip = true;
          break;
        }
      } else if (IdLen == 4 && !strcmp(Directive+1, "lif")) {  // "elif".
        PPConditionalInfo &CondInfo = CurLexer->peekConditionalLevel();

        bool ShouldEnter;
        // If this is in a skipping block or if we're already handled this #if
        // block, don't bother parsing the condition.
        if (CondInfo.WasSkipping || CondInfo.FoundNonSkip) {
          DiscardUntilEndOfDirective();
          ShouldEnter = false;
        } else {
          // Restore the value of LexingRawMode so that identifiers are
          // looked up, etc, inside the #elif expression.
          assert(CurLexer->LexingRawMode && "We have to be skipping here!");
          CurLexer->LexingRawMode = false;
          IdentifierInfo *IfNDefMacro = 0;
          ShouldEnter = EvaluateDirectiveExpression(IfNDefMacro);
          CurLexer->LexingRawMode = true;
        }
        
        // If this is a #elif with a #else before it, report the error.
        if (CondInfo.FoundElse) Diag(Tok, diag::pp_err_elif_after_else);
        
        // If this condition is true, enter it!
        if (ShouldEnter) {
          CondInfo.FoundNonSkip = true;
          break;
        }
      }
    }
    
    CurLexer->ParsingPreprocessorDirective = false;
  }

  // Finally, if we are out of the conditional (saw an #endif or ran off the end
  // of the file, just stop skipping and return to lexing whatever came after
  // the #if block.
  CurLexer->LexingRawMode = false;
}

//===----------------------------------------------------------------------===//
// Preprocessor Directive Handling.
//===----------------------------------------------------------------------===//

/// HandleDirective - This callback is invoked when the lexer sees a # token
/// at the start of a line.  This consumes the directive, modifies the 
/// lexer/preprocessor state, and advances the lexer(s) so that the next token
/// read is the correct one.
void Preprocessor::HandleDirective(LexerToken &Result) {
  // FIXME: Traditional: # with whitespace before it not recognized by K&R?
  
  // We just parsed a # character at the start of a line, so we're in directive
  // mode.  Tell the lexer this so any newlines we see will be converted into an
  // EOM token (which terminates the directive).
  CurLexer->ParsingPreprocessorDirective = true;
  
  ++NumDirectives;
  
  // We are about to read a token.  For the multiple-include optimization FA to
  // work, we have to remember if we had read any tokens *before* this 
  // pp-directive.
  bool ReadAnyTokensBeforeDirective = CurLexer->MIOpt.getHasReadAnyTokensVal();
  
  // Read the next token, the directive flavor.  This isn't expanded due to
  // C99 6.10.3p8.
  LexUnexpandedToken(Result);
  
  // C99 6.10.3p11: Is this preprocessor directive in macro invocation?  e.g.:
  //   #define A(x) #x
  //   A(abc
  //     #warning blah
  //   def)
  // If so, the user is relying on non-portable behavior, emit a diagnostic.
  if (InMacroFormalArgs)
    Diag(Result, diag::ext_embedded_directive);
  
  switch (Result.getKind()) {
  default: break;
  case tok::eom:
    return;   // null directive.

#if 0
  case tok::numeric_constant:
    // FIXME: implement # 7 line numbers!
    break;
#endif
  case tok::kw_else:
    return HandleElseDirective(Result);
  case tok::kw_if:
    return HandleIfDirective(Result, ReadAnyTokensBeforeDirective);
  case tok::identifier:
    // Get the identifier name without trigraphs or embedded newlines.
    const char *Directive = Result.getIdentifierInfo()->getName();
    bool isExtension = false;
    switch (Result.getIdentifierInfo()->getNameLength()) {
    case 4:
      if (Directive[0] == 'l' && !strcmp(Directive, "line"))
        ;  // FIXME: implement #line
      if (Directive[0] == 'e' && !strcmp(Directive, "elif"))
        return HandleElifDirective(Result);
      if (Directive[0] == 's' && !strcmp(Directive, "sccs"))
        return HandleIdentSCCSDirective(Result);
      break;
    case 5:
      if (Directive[0] == 'e' && !strcmp(Directive, "endif"))
        return HandleEndifDirective(Result);
      if (Directive[0] == 'i' && !strcmp(Directive, "ifdef"))
        return HandleIfdefDirective(Result, false, true/*not valid for miopt*/);
      if (Directive[0] == 'u' && !strcmp(Directive, "undef"))
        return HandleUndefDirective(Result);
      if (Directive[0] == 'e' && !strcmp(Directive, "error"))
        return HandleUserDiagnosticDirective(Result, false);
      if (Directive[0] == 'i' && !strcmp(Directive, "ident"))
        return HandleIdentSCCSDirective(Result);
      break;
    case 6:
      if (Directive[0] == 'd' && !strcmp(Directive, "define"))
        return HandleDefineDirective(Result);
      if (Directive[0] == 'i' && !strcmp(Directive, "ifndef"))
        return HandleIfdefDirective(Result, true, ReadAnyTokensBeforeDirective);
      if (Directive[0] == 'i' && !strcmp(Directive, "import"))
        return HandleImportDirective(Result);
      if (Directive[0] == 'p' && !strcmp(Directive, "pragma"))
        return HandlePragmaDirective();
      if (Directive[0] == 'a' && !strcmp(Directive, "assert"))
        isExtension = true;  // FIXME: implement #assert
      break;
    case 7:
      if (Directive[0] == 'i' && !strcmp(Directive, "include"))
        return HandleIncludeDirective(Result);            // Handle #include.
      if (Directive[0] == 'w' && !strcmp(Directive, "warning")) {
        Diag(Result, diag::ext_pp_warning_directive);
        return HandleUserDiagnosticDirective(Result, true);
      }
      break;
    case 8:
      if (Directive[0] == 'u' && !strcmp(Directive, "unassert")) {
        isExtension = true;  // FIXME: implement #unassert
      }
      break;
    case 12:
      if (Directive[0] == 'i' && !strcmp(Directive, "include_next"))
        return HandleIncludeNextDirective(Result);      // Handle #include_next.
      break;
    }
    break;
  }
  
  // If we reached here, the preprocessing token is not valid!
  Diag(Result, diag::err_pp_invalid_directive);
  
  // Read the rest of the PP line.
  DiscardUntilEndOfDirective();
  
  // Okay, we're done parsing the directive.
}

void Preprocessor::HandleUserDiagnosticDirective(LexerToken &Tok, 
                                                 bool isWarning) {
  // Read the rest of the line raw.  We do this because we don't want macros
  // to be expanded and we don't require that the tokens be valid preprocessing
  // tokens.  For example, this is allowed: "#warning `   'foo".  GCC does
  // collapse multiple consequtive white space between tokens, but this isn't
  // specified by the standard.
  std::string Message = CurLexer->ReadToEndOfLine();

  unsigned DiagID = isWarning ? diag::pp_hash_warning : diag::err_pp_hash_error;
  return Diag(Tok, DiagID, Message);
}

/// HandleIdentSCCSDirective - Handle a #ident/#sccs directive.
///
void Preprocessor::HandleIdentSCCSDirective(LexerToken &Tok) {
  // Yes, this directive is an extension.
  Diag(Tok, diag::ext_pp_ident_directive);
  
  // Read the string argument.
  LexerToken StrTok;
  Lex(StrTok);
  
  // If the token kind isn't a string, it's a malformed directive.
  if (StrTok.getKind() != tok::string_literal)
    return Diag(StrTok, diag::err_pp_malformed_ident);
  
  // Verify that there is nothing after the string, other than EOM.
  CheckEndOfDirective("#ident");

  if (IdentHandler)
    IdentHandler(Tok.getLocation(), getSpelling(StrTok));
}

//===----------------------------------------------------------------------===//
// Preprocessor Include Directive Handling.
//===----------------------------------------------------------------------===//

/// HandleIncludeDirective - The "#include" tokens have just been read, read the
/// file to be included from the lexer, then include it!  This is a common
/// routine with functionality shared between #include, #include_next and
/// #import.
void Preprocessor::HandleIncludeDirective(LexerToken &IncludeTok,
                                          const DirectoryLookup *LookupFrom,
                                          bool isImport) {
  ++NumIncluded;

  LexerToken FilenameTok;
  std::string Filename = CurLexer->LexIncludeFilename(FilenameTok);
  
  // If the token kind is EOM, the error has already been diagnosed.
  if (FilenameTok.getKind() == tok::eom)
    return;
  
  // Verify that there is nothing after the filename, other than EOM.  Use the
  // preprocessor to lex this in case lexing the filename entered a macro.
  CheckEndOfDirective("#include");

  // Check that we don't have infinite #include recursion.
  if (IncludeMacroStack.size() == MaxAllowedIncludeStackDepth-1)
    return Diag(FilenameTok, diag::err_pp_include_too_deep);
  
  // Find out whether the filename is <x> or "x".
  bool isAngled = Filename[0] == '<';
  
  // Remove the quotes.
  Filename = std::string(Filename.begin()+1, Filename.end()-1);
  
  // Search include directories.
  const DirectoryLookup *CurDir;
  const FileEntry *File = LookupFile(Filename, isAngled, LookupFrom, CurDir);
  if (File == 0)
    return Diag(FilenameTok, diag::err_pp_file_not_found);
  
  // Get information about this file.
  PerFileInfo &FileInfo = getFileInfo(File);
  
  // If this is a #import directive, check that we have not already imported
  // this header.
  if (isImport) {
    // If this has already been imported, don't import it again.
    FileInfo.isImport = true;
    
    // Has this already been #import'ed or #include'd?
    if (FileInfo.NumIncludes) return;
  } else {
    // Otherwise, if this is a #include of a file that was previously #import'd
    // or if this is the second #include of a #pragma once file, ignore it.
    if (FileInfo.isImport)
      return;
  }
  
  // Next, check to see if the file is wrapped with #ifndef guards.  If so, and
  // if the macro that guards it is defined, we know the #include has no effect.
  if (FileInfo.ControllingMacro && FileInfo.ControllingMacro->getMacroInfo()) {
    ++NumMultiIncludeFileOptzn;
    return;
  }
  

  // Look up the file, create a File ID for it.
  unsigned FileID = SourceMgr.createFileID(File, FilenameTok.getLocation());
  if (FileID == 0)
    return Diag(FilenameTok, diag::err_pp_file_not_found);

  // Finally, if all is good, enter the new file!
  EnterSourceFile(FileID, CurDir);

  // Increment the number of times this file has been included.
  ++FileInfo.NumIncludes;
}

/// HandleIncludeNextDirective - Implements #include_next.
///
void Preprocessor::HandleIncludeNextDirective(LexerToken &IncludeNextTok) {
  Diag(IncludeNextTok, diag::ext_pp_include_next_directive);
  
  // #include_next is like #include, except that we start searching after
  // the current found directory.  If we can't do this, issue a
  // diagnostic.
  const DirectoryLookup *Lookup = CurDirLookup;
  if (isInPrimaryFile()) {
    Lookup = 0;
    Diag(IncludeNextTok, diag::pp_include_next_in_primary);
  } else if (Lookup == 0) {
    Diag(IncludeNextTok, diag::pp_include_next_absolute_path);
  } else {
    // Start looking up in the next directory.
    ++Lookup;
  }
  
  return HandleIncludeDirective(IncludeNextTok, Lookup);
}

/// HandleImportDirective - Implements #import.
///
void Preprocessor::HandleImportDirective(LexerToken &ImportTok) {
  Diag(ImportTok, diag::ext_pp_import_directive);
  
  return HandleIncludeDirective(ImportTok, 0, true);
}

//===----------------------------------------------------------------------===//
// Preprocessor Macro Directive Handling.
//===----------------------------------------------------------------------===//

/// ReadMacroDefinitionArgList - The ( starting an argument list of a macro
/// definition has just been read.  Lex the rest of the arguments and the
/// closing ), updating MI with what we learn.  Return true if an error occurs
/// parsing the arg list.
bool Preprocessor::ReadMacroDefinitionArgList(MacroInfo *MI) {
  LexerToken Tok;
  while (1) {
    LexUnexpandedToken(Tok);
    switch (Tok.getKind()) {
    case tok::r_paren:
      // Found the end of the argument list.
      if (MI->arg_begin() == MI->arg_end()) return false;  // #define FOO()
      // Otherwise we have #define FOO(A,)
      Diag(Tok, diag::err_pp_expected_ident_in_arg_list);
      return true;
    case tok::ellipsis:  // #define X(... -> C99 varargs
      // Warn if use of C99 feature in non-C99 mode.
      if (!Features.C99) Diag(Tok, diag::ext_variadic_macro);

      // Lex the token after the identifier.
      LexUnexpandedToken(Tok);
      if (Tok.getKind() != tok::r_paren) {
        Diag(Tok, diag::err_pp_missing_rparen_in_macro_def);
        return true;
      }
      MI->setIsC99Varargs();
      return false;
    case tok::eom:  // #define X(
      Diag(Tok, diag::err_pp_missing_rparen_in_macro_def);
      return true;
    default:        // #define X(1
      Diag(Tok, diag::err_pp_invalid_tok_in_arg_list);
      return true;
    case tok::identifier:
      IdentifierInfo *II = Tok.getIdentifierInfo();

      // If this is already used as an argument, it is used multiple times (e.g.
      // #define X(A,A.
      if (II->isMacroArg()) {  // C99 6.10.3p6
        Diag(Tok, diag::err_pp_duplicate_name_in_arg_list, II->getName());
        return true;
      }
        
      // Add the argument to the macro info.
      MI->addArgument(II);
      // Remember it is an argument now.
      II->setIsMacroArg(true);
      
      // Lex the token after the identifier.
      LexUnexpandedToken(Tok);
      
      switch (Tok.getKind()) {
      default:          // #define X(A B
        Diag(Tok, diag::err_pp_expected_comma_in_arg_list);
        return true;
      case tok::r_paren: // #define X(A)
        return false;
      case tok::comma:  // #define X(A,
        break;
      case tok::ellipsis:  // #define X(A... -> GCC extension
        // Diagnose extension.
        Diag(Tok, diag::ext_named_variadic_macro);
        
        // Lex the token after the identifier.
        LexUnexpandedToken(Tok);
        if (Tok.getKind() != tok::r_paren) {
          Diag(Tok, diag::err_pp_missing_rparen_in_macro_def);
          return true;
        }
          
        MI->setIsGNUVarargs();
        return false;
      }
    }
  }
}

/// HandleDefineDirective - Implements #define.  This consumes the entire macro
/// line then lets the caller lex the next real token.
///
void Preprocessor::HandleDefineDirective(LexerToken &DefineTok) {
  ++NumDefined;

  LexerToken MacroNameTok;
  ReadMacroName(MacroNameTok, 1);
  
  // Error reading macro name?  If so, diagnostic already issued.
  if (MacroNameTok.getKind() == tok::eom)
    return;
  
  MacroInfo *MI = new MacroInfo(MacroNameTok.getLocation());
  
  LexerToken Tok;
  LexUnexpandedToken(Tok);
  
  // FIXME: Enable __VA_ARGS__.

  // If this is a function-like macro definition, parse the argument list,
  // marking each of the identifiers as being used as macro arguments.  Also,
  // check other constraints on the first token of the macro body.
  if (Tok.getKind() == tok::eom) {
    // If there is no body to this macro, we have no special handling here.
  } else if (Tok.getKind() == tok::l_paren && !Tok.hasLeadingSpace()) {
    // This is a function-like macro definition.  Read the argument list.
    MI->setIsFunctionLike();
    if (ReadMacroDefinitionArgList(MI)) {
      // Clear the "isMacroArg" flags from all the macro arguments parsed.
      MI->SetIdentifierIsMacroArgFlags(false);
      // Forget about MI.
      delete MI;
      // Throw away the rest of the line.
      if (CurLexer->ParsingPreprocessorDirective)
        DiscardUntilEndOfDirective();
      return;
    }

    // Read the first token after the arg list for down below.
    LexUnexpandedToken(Tok);
  } else if (!Tok.hasLeadingSpace()) {
    // C99 requires whitespace between the macro definition and the body.  Emit
    // a diagnostic for something like "#define X+".
    if (Features.C99) {
      Diag(Tok, diag::ext_c99_whitespace_required_after_macro_name);
    } else {
      // FIXME: C90/C++ do not get this diagnostic, but it does get a similar
      // one in some cases!
    }
  } else {
    // This is a normal token with leading space.  Clear the leading space
    // marker on the first token to get proper expansion.
    Tok.ClearFlag(LexerToken::LeadingSpace);
  }
  
  // Read the rest of the macro body.
  while (Tok.getKind() != tok::eom) {
    MI->AddTokenToBody(Tok);

    // Check C99 6.10.3.2p1: ensure that # operators are followed by macro
    // parameters in function-like macro expansions.
    if (Tok.getKind() != tok::hash || MI->isObjectLike()) {
      // Get the next token of the macro.
      LexUnexpandedToken(Tok);
      continue;
    }
    
    // Get the next token of the macro.
    LexUnexpandedToken(Tok);
   
    // Not a macro arg identifier?
    if (!Tok.getIdentifierInfo() || !Tok.getIdentifierInfo()->isMacroArg()) {
      Diag(Tok, diag::err_pp_stringize_not_parameter);
      // Clear the "isMacroArg" flags from all the macro arguments.
      MI->SetIdentifierIsMacroArgFlags(false);
      delete MI;
      return;
    }
    
    // Things look ok, add the param name token to the macro.
    MI->AddTokenToBody(Tok);

    // Get the next token of the macro.
    LexUnexpandedToken(Tok);
  }

  // Clear the "isMacroArg" flags from all the macro arguments.
  MI->SetIdentifierIsMacroArgFlags(false);
  
  // Check that there is no paste (##) operator at the begining or end of the
  // replacement list.
  unsigned NumTokens = MI->getNumTokens();
  if (NumTokens != 0) {
    if (MI->getReplacementToken(0).getKind() == tok::hashhash) {
      Diag(MI->getReplacementToken(0), diag::err_paste_at_start);
      delete MI;
      return;
    }
    if (MI->getReplacementToken(NumTokens-1).getKind() == tok::hashhash) {
      Diag(MI->getReplacementToken(NumTokens-1), diag::err_paste_at_end);
      delete MI;
      return;
    }
  }
  
  // If this is the primary source file, remember that this macro hasn't been
  // used yet.
  if (isInPrimaryFile())
    MI->setIsUsed(false);
  
  // Finally, if this identifier already had a macro defined for it, verify that
  // the macro bodies are identical and free the old definition.
  if (MacroInfo *OtherMI = MacroNameTok.getIdentifierInfo()->getMacroInfo()) {
    if (!OtherMI->isUsed())
      Diag(OtherMI->getDefinitionLoc(), diag::pp_macro_not_used);

    // Macros must be identical.  This means all tokes and whitespace separation
    // must be the same.  C99 6.10.3.2.
    if (!MI->isIdenticalTo(*OtherMI, *this)) {
      Diag(MI->getDefinitionLoc(), diag::ext_pp_macro_redef,
           MacroNameTok.getIdentifierInfo()->getName());
      Diag(OtherMI->getDefinitionLoc(), diag::ext_pp_macro_redef2);
    }
    delete OtherMI;
  }
  
  MacroNameTok.getIdentifierInfo()->setMacroInfo(MI);
}


/// HandleUndefDirective - Implements #undef.
///
void Preprocessor::HandleUndefDirective(LexerToken &UndefTok) {
  ++NumUndefined;

  LexerToken MacroNameTok;
  ReadMacroName(MacroNameTok, 2);
  
  // Error reading macro name?  If so, diagnostic already issued.
  if (MacroNameTok.getKind() == tok::eom)
    return;
  
  // Check to see if this is the last token on the #undef line.
  CheckEndOfDirective("#undef");
  
  // Okay, we finally have a valid identifier to undef.
  MacroInfo *MI = MacroNameTok.getIdentifierInfo()->getMacroInfo();
  
  // If the macro is not defined, this is a noop undef, just return.
  if (MI == 0) return;

  if (!MI->isUsed())
    Diag(MI->getDefinitionLoc(), diag::pp_macro_not_used);
  
  // Free macro definition.
  delete MI;
  MacroNameTok.getIdentifierInfo()->setMacroInfo(0);
}


//===----------------------------------------------------------------------===//
// Preprocessor Conditional Directive Handling.
//===----------------------------------------------------------------------===//

/// HandleIfdefDirective - Implements the #ifdef/#ifndef directive.  isIfndef is
/// true when this is a #ifndef directive.  ReadAnyTokensBeforeDirective is true
/// if any tokens have been returned or pp-directives activated before this
/// #ifndef has been lexed.
///
void Preprocessor::HandleIfdefDirective(LexerToken &Result, bool isIfndef,
                                        bool ReadAnyTokensBeforeDirective) {
  ++NumIf;
  LexerToken DirectiveTok = Result;

  LexerToken MacroNameTok;
  ReadMacroName(MacroNameTok);
  
  // Error reading macro name?  If so, diagnostic already issued.
  if (MacroNameTok.getKind() == tok::eom)
    return;
  
  // Check to see if this is the last token on the #if[n]def line.
  CheckEndOfDirective(isIfndef ? "#ifndef" : "#ifdef");
  
  // If the start of a top-level #ifdef, inform MIOpt.
  if (!ReadAnyTokensBeforeDirective &&
      CurLexer->getConditionalStackDepth() == 0) {
    assert(isIfndef && "#ifdef shouldn't reach here");
    CurLexer->MIOpt.EnterTopLevelIFNDEF(MacroNameTok.getIdentifierInfo());
  }
  
  MacroInfo *MI = MacroNameTok.getIdentifierInfo()->getMacroInfo();

  // If there is a macro, mark it used.
  if (MI) MI->setIsUsed(true);
  
  // Should we include the stuff contained by this directive?
  if (!MI == isIfndef) {
    // Yes, remember that we are inside a conditional, then lex the next token.
    CurLexer->pushConditionalLevel(DirectiveTok.getLocation(), /*wasskip*/false,
                                   /*foundnonskip*/true, /*foundelse*/false);
  } else {
    // No, skip the contents of this block and return the first token after it.
    SkipExcludedConditionalBlock(DirectiveTok.getLocation(),
                                 /*Foundnonskip*/false, 
                                 /*FoundElse*/false);
  }
}

/// HandleIfDirective - Implements the #if directive.
///
void Preprocessor::HandleIfDirective(LexerToken &IfToken,
                                     bool ReadAnyTokensBeforeDirective) {
  ++NumIf;
  
  // Parse and evaluation the conditional expression.
  IdentifierInfo *IfNDefMacro = 0;
  bool ConditionalTrue = EvaluateDirectiveExpression(IfNDefMacro);
  
  // Should we include the stuff contained by this directive?
  if (ConditionalTrue) {
    // If this condition is equivalent to #ifndef X, and if this is the first
    // directive seen, handle it for the multiple-include optimization.
    if (!ReadAnyTokensBeforeDirective &&
        CurLexer->getConditionalStackDepth() == 0 && IfNDefMacro)
      CurLexer->MIOpt.EnterTopLevelIFNDEF(IfNDefMacro);
    
    // Yes, remember that we are inside a conditional, then lex the next token.
    CurLexer->pushConditionalLevel(IfToken.getLocation(), /*wasskip*/false,
                                   /*foundnonskip*/true, /*foundelse*/false);
  } else {
    // No, skip the contents of this block and return the first token after it.
    SkipExcludedConditionalBlock(IfToken.getLocation(), /*Foundnonskip*/false, 
                                 /*FoundElse*/false);
  }
}

/// HandleEndifDirective - Implements the #endif directive.
///
void Preprocessor::HandleEndifDirective(LexerToken &EndifToken) {
  ++NumEndif;
  
  // Check that this is the whole directive.
  CheckEndOfDirective("#endif");
  
  PPConditionalInfo CondInfo;
  if (CurLexer->popConditionalLevel(CondInfo)) {
    // No conditionals on the stack: this is an #endif without an #if.
    return Diag(EndifToken, diag::err_pp_endif_without_if);
  }
  
  // If this the end of a top-level #endif, inform MIOpt.
  if (CurLexer->getConditionalStackDepth() == 0)
    CurLexer->MIOpt.ExitTopLevelConditional();
  
  assert(!CondInfo.WasSkipping && !isSkipping() &&
         "This code should only be reachable in the non-skipping case!");
}


void Preprocessor::HandleElseDirective(LexerToken &Result) {
  ++NumElse;
  
  // #else directive in a non-skipping conditional... start skipping.
  CheckEndOfDirective("#else");
  
  PPConditionalInfo CI;
  if (CurLexer->popConditionalLevel(CI))
    return Diag(Result, diag::pp_err_else_without_if);
  
  // If this is a top-level #else, inform the MIOpt.
  if (CurLexer->getConditionalStackDepth() == 0)
    CurLexer->MIOpt.FoundTopLevelElse();

  // If this is a #else with a #else before it, report the error.
  if (CI.FoundElse) Diag(Result, diag::pp_err_else_after_else);
  
  // Finally, skip the rest of the contents of this block and return the first
  // token after it.
  return SkipExcludedConditionalBlock(CI.IfLoc, /*Foundnonskip*/true,
                                      /*FoundElse*/true);
}

void Preprocessor::HandleElifDirective(LexerToken &ElifToken) {
  ++NumElse;
  
  // #elif directive in a non-skipping conditional... start skipping.
  // We don't care what the condition is, because we will always skip it (since
  // the block immediately before it was included).
  DiscardUntilEndOfDirective();

  PPConditionalInfo CI;
  if (CurLexer->popConditionalLevel(CI))
    return Diag(ElifToken, diag::pp_err_elif_without_if);
  
  // If this is a top-level #elif, inform the MIOpt.
  if (CurLexer->getConditionalStackDepth() == 0)
    CurLexer->MIOpt.FoundTopLevelElse();
  
  // If this is a #elif with a #else before it, report the error.
  if (CI.FoundElse) Diag(ElifToken, diag::pp_err_elif_after_else);

  // Finally, skip the rest of the contents of this block and return the first
  // token after it.
  return SkipExcludedConditionalBlock(CI.IfLoc, /*Foundnonskip*/true,
                                      /*FoundElse*/CI.FoundElse);
}

