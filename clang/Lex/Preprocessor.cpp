//===--- Preprocess.cpp - C Language Family Preprocessor Implementation ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the Preprocessor interface.
//
//===----------------------------------------------------------------------===//
//
// Options to support:
//   -H       - Print the name of each header file used.
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
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Pragma.h"
#include "clang/Lex/ScratchBuffer.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Streams.h"
#include <ctime>
using namespace clang;

//===----------------------------------------------------------------------===//

Preprocessor::Preprocessor(Diagnostic &diags, const LangOptions &opts,
                           TargetInfo &target, SourceManager &SM, 
                           HeaderSearch &Headers) 
  : Diags(diags), Features(opts), Target(target), FileMgr(Headers.getFileMgr()),
    SourceMgr(SM), HeaderInfo(Headers), Identifiers(opts),
    CurLexer(0), CurDirLookup(0), CurMacroExpander(0), Callbacks(0) {
  ScratchBuf = new ScratchBuffer(SourceMgr);

  // Clear stats.
  NumDirectives = NumDefined = NumUndefined = NumPragma = 0;
  NumIf = NumElse = NumEndif = 0;
  NumEnteredSourceFiles = 0;
  NumMacroExpanded = NumFnMacroExpanded = NumBuiltinMacroExpanded = 0;
  NumFastMacroExpanded = NumTokenPaste = NumFastTokenPaste = 0;
  MaxIncludeStackDepth = 0; 
  NumSkipped = 0;

  // Default to discarding comments.
  KeepComments = false;
  KeepMacroComments = false;
  
  // Macro expansion is enabled.
  DisableMacroExpansion = false;
  InMacroArgs = false;
  NumCachedMacroExpanders = 0;

  // "Poison" __VA_ARGS__, which can only appear in the expansion of a macro.
  // This gets unpoisoned where it is allowed.
  (Ident__VA_ARGS__ = getIdentifierInfo("__VA_ARGS__"))->setIsPoisoned();
  
  Predefines = 0;
  
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

  // Free any macro definitions.
  for (llvm::DenseMap<IdentifierInfo*, MacroInfo*>::iterator I =
       Macros.begin(), E = Macros.end(); I != E; ++I) {
    // Free the macro definition.
    delete I->second;
    I->second = 0;
    I->first->setHasMacroDefinition(false);
  }
  
  // Free any cached macro expanders.
  for (unsigned i = 0, e = NumCachedMacroExpanders; i != e; ++i)
    delete MacroExpanderCache[i];
  
  // Release pragma information.
  delete PragmaHandlers;

  // Delete the scratch buffer info.
  delete ScratchBuf;
}

PPCallbacks::~PPCallbacks() {
}

/// Diag - Forwarding function for diagnostics.  This emits a diagnostic at
/// the specified Token's location, translating the token's start
/// position in the current buffer into a SourcePosition object for rendering.
void Preprocessor::Diag(SourceLocation Loc, unsigned DiagID) {
  Diags.Report(getFullLoc(Loc), DiagID);
}

void Preprocessor::Diag(SourceLocation Loc, unsigned DiagID, 
                        const std::string &Msg) {
  Diags.Report(getFullLoc(Loc), DiagID, &Msg, 1);
}

void Preprocessor::DumpToken(const Token &Tok, bool DumpFlags) const {
  llvm::cerr << tok::getTokenName(Tok.getKind()) << " '"
             << getSpelling(Tok) << "'";
  
  if (!DumpFlags) return;
  
  llvm::cerr << "\t";
  if (Tok.isAtStartOfLine())
    llvm::cerr << " [StartOfLine]";
  if (Tok.hasLeadingSpace())
    llvm::cerr << " [LeadingSpace]";
  if (Tok.isExpandDisabled())
    llvm::cerr << " [ExpandDisabled]";
  if (Tok.needsCleaning()) {
    const char *Start = SourceMgr.getCharacterData(Tok.getLocation());
    llvm::cerr << " [UnClean='" << std::string(Start, Start+Tok.getLength())
               << "']";
  }
  
  llvm::cerr << "\tLoc=<";
  DumpLocation(Tok.getLocation());
  llvm::cerr << ">";
}

void Preprocessor::DumpLocation(SourceLocation Loc) const {
  SourceLocation LogLoc = SourceMgr.getLogicalLoc(Loc);
  llvm::cerr << SourceMgr.getSourceName(LogLoc) << ':'
             << SourceMgr.getLineNumber(LogLoc) << ':'
             << SourceMgr.getLineNumber(LogLoc);
  
  SourceLocation PhysLoc = SourceMgr.getPhysicalLoc(Loc);
  if (PhysLoc != LogLoc) {
    llvm::cerr << " <PhysLoc=";
    DumpLocation(PhysLoc);
    llvm::cerr << ">";
  }
}

void Preprocessor::DumpMacro(const MacroInfo &MI) const {
  llvm::cerr << "MACRO: ";
  for (unsigned i = 0, e = MI.getNumTokens(); i != e; ++i) {
    DumpToken(MI.getReplacementToken(i));
    llvm::cerr << "  ";
  }
  llvm::cerr << "\n";
}

void Preprocessor::PrintStats() {
  llvm::cerr << "\n*** Preprocessor Stats:\n";
  llvm::cerr << NumDirectives << " directives found:\n";
  llvm::cerr << "  " << NumDefined << " #define.\n";
  llvm::cerr << "  " << NumUndefined << " #undef.\n";
  llvm::cerr << "  #include/#include_next/#import:\n";
  llvm::cerr << "    " << NumEnteredSourceFiles << " source files entered.\n";
  llvm::cerr << "    " << MaxIncludeStackDepth << " max include stack depth\n";
  llvm::cerr << "  " << NumIf << " #if/#ifndef/#ifdef.\n";
  llvm::cerr << "  " << NumElse << " #else/#elif.\n";
  llvm::cerr << "  " << NumEndif << " #endif.\n";
  llvm::cerr << "  " << NumPragma << " #pragma.\n";
  llvm::cerr << NumSkipped << " #if/#ifndef#ifdef regions skipped\n";

  llvm::cerr << NumMacroExpanded << "/" << NumFnMacroExpanded << "/"
             << NumBuiltinMacroExpanded << " obj/fn/builtin macros expanded, "
             << NumFastMacroExpanded << " on the fast path.\n";
  llvm::cerr << (NumFastTokenPaste+NumTokenPaste)
             << " token paste (##) operations performed, "
             << NumFastTokenPaste << " on the fast path.\n";
}

//===----------------------------------------------------------------------===//
// Token Spelling
//===----------------------------------------------------------------------===//


/// getSpelling() - Return the 'spelling' of this token.  The spelling of a
/// token are the characters used to represent the token in the source file
/// after trigraph expansion and escaped-newline folding.  In particular, this
/// wants to get the true, uncanonicalized, spelling of things like digraphs
/// UCNs, etc.
std::string Preprocessor::getSpelling(const Token &Tok) const {
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
unsigned Preprocessor::getSpelling(const Token &Tok,
                                   const char *&Buffer) const {
  assert((int)Tok.getLength() >= 0 && "Token character range is bogus!");
  
  // If this token is an identifier, just return the string from the identifier
  // table, which is very quick.
  if (const IdentifierInfo *II = Tok.getIdentifierInfo()) {
    Buffer = II->getName();
    
    // Return the length of the token.  If the token needed cleaning, don't
    // include the size of the newlines or trigraphs in it.
    if (!Tok.needsCleaning())
      return Tok.getLength();
    else
      return strlen(Buffer);
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


/// CreateString - Plop the specified string into a scratch buffer and return a
/// location for it.  If specified, the source location provides a source
/// location for the token.
SourceLocation Preprocessor::
CreateString(const char *Buf, unsigned Len, SourceLocation SLoc) {
  if (SLoc.isValid())
    return ScratchBuf->getToken(Buf, Len, SLoc);
  return ScratchBuf->getToken(Buf, Len);
}


/// AdvanceToTokenCharacter - Given a location that specifies the start of a
/// token, return a new location that specifies a character within the token.
SourceLocation Preprocessor::AdvanceToTokenCharacter(SourceLocation TokStart, 
                                                     unsigned CharNo) {
  // If they request the first char of the token, we're trivially done.  If this
  // is a macro expansion, it doesn't make sense to point to a character within
  // the instantiation point (the name).  We could point to the source
  // character, but without also pointing to instantiation info, this is
  // confusing.
  if (CharNo == 0 || TokStart.isMacroID()) return TokStart;
  
  // Figure out how many physical characters away the specified logical
  // character is.  This needs to take into consideration newlines and
  // trigraphs.
  const char *TokPtr = SourceMgr.getCharacterData(TokStart);
  unsigned PhysOffset = 0;
  
  // The usual case is that tokens don't contain anything interesting.  Skip
  // over the uninteresting characters.  If a token only consists of simple
  // chars, this method is extremely fast.
  while (CharNo && Lexer::isObviouslySimpleCharacter(*TokPtr))
    ++TokPtr, --CharNo, ++PhysOffset;
  
  // If we have a character that may be a trigraph or escaped newline, create a
  // lexer to parse it correctly.
  if (CharNo != 0) {
    // Create a lexer starting at this token position.
    Lexer TheLexer(TokStart, *this, TokPtr);
    Token Tok;
    // Skip over characters the remaining characters.
    const char *TokStartPtr = TokPtr;
    for (; CharNo; --CharNo)
      TheLexer.getAndAdvanceChar(TokPtr, Tok);
    
    PhysOffset += TokPtr-TokStartPtr;
  }
  
  return TokStart.getFileLocWithOffset(PhysOffset);
}


//===----------------------------------------------------------------------===//
// Preprocessor Initialization Methods
//===----------------------------------------------------------------------===//

// Append a #define line to Buf for Macro.  Macro should be of the form XXX,
// in which case we emit "#define XXX 1" or "XXX=Y z W" in which case we emit
// "#define XXX Y z W".  To get a #define with no value, use "XXX=".
static void DefineBuiltinMacro(std::vector<char> &Buf, const char *Macro,
                               const char *Command = "#define ") {
  Buf.insert(Buf.end(), Command, Command+strlen(Command));
  if (const char *Equal = strchr(Macro, '=')) {
    // Turn the = into ' '.
    Buf.insert(Buf.end(), Macro, Equal);
    Buf.push_back(' ');
    Buf.insert(Buf.end(), Equal+1, Equal+strlen(Equal));
  } else {
    // Push "macroname 1".
    Buf.insert(Buf.end(), Macro, Macro+strlen(Macro));
    Buf.push_back(' ');
    Buf.push_back('1');
  }
  Buf.push_back('\n');
}


static void InitializePredefinedMacros(Preprocessor &PP, 
                                       std::vector<char> &Buf) {
  // FIXME: Implement magic like cpp_init_builtins for things like __STDC__
  // and __DATE__ etc.
#if 0
  /* __STDC__ has the value 1 under normal circumstances.
  However, if (a) we are in a system header, (b) the option
  stdc_0_in_system_headers is true (set by target config), and
  (c) we are not in strictly conforming mode, then it has the
  value 0.  (b) and (c) are already checked in cpp_init_builtins.  */
  //case BT_STDC:
  if (cpp_in_system_header (pfile))
    number = 0;
  else
    number = 1;
  break;
#endif    
  // These should all be defined in the preprocessor according to the
  // current language configuration.
  DefineBuiltinMacro(Buf, "__STDC__=1");
  //DefineBuiltinMacro(Buf, "__ASSEMBLER__=1");
  if (PP.getLangOptions().C99 && !PP.getLangOptions().CPlusPlus)
    DefineBuiltinMacro(Buf, "__STDC_VERSION__=199901L");
  else if (0) // STDC94 ?
    DefineBuiltinMacro(Buf, "__STDC_VERSION__=199409L");
  
  DefineBuiltinMacro(Buf, "__STDC_HOSTED__=1");
  if (PP.getLangOptions().ObjC1)
    DefineBuiltinMacro(Buf, "__OBJC__=1");
  if (PP.getLangOptions().ObjC2)
    DefineBuiltinMacro(Buf, "__OBJC2__=1");

  // Add __builtin_va_list typedef.
  {
    const char *VAList = PP.getTargetInfo().getVAListDeclaration();
    Buf.insert(Buf.end(), VAList, VAList+strlen(VAList));
    Buf.push_back('\n');
  }
  
  // Get the target #defines.
  PP.getTargetInfo().getTargetDefines(Buf);
  
  // Compiler set macros.
  DefineBuiltinMacro(Buf, "__APPLE_CC__=5250");
  DefineBuiltinMacro(Buf, "__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__=1050");
  DefineBuiltinMacro(Buf, "__GNUC_MINOR__=0");
  DefineBuiltinMacro(Buf, "__GNUC_PATCHLEVEL__=1");
  DefineBuiltinMacro(Buf, "__GNUC__=4");
  DefineBuiltinMacro(Buf, "__GXX_ABI_VERSION=1002");
  DefineBuiltinMacro(Buf, "__VERSION__=\"4.0.1 (Apple Computer, Inc. "
                     "build 5250)\"");
  
  // Build configuration options.
  DefineBuiltinMacro(Buf, "__DYNAMIC__=1");
  DefineBuiltinMacro(Buf, "__FINITE_MATH_ONLY__=0");
  DefineBuiltinMacro(Buf, "__NO_INLINE__=1");
  DefineBuiltinMacro(Buf, "__PIC__=1");
  
  
  if (PP.getLangOptions().CPlusPlus) {
    DefineBuiltinMacro(Buf, "__DEPRECATED=1");
    DefineBuiltinMacro(Buf, "__EXCEPTIONS=1");
    DefineBuiltinMacro(Buf, "__GNUG__=4");
    DefineBuiltinMacro(Buf, "__GXX_WEAK__=1");
    DefineBuiltinMacro(Buf, "__cplusplus=1");
    DefineBuiltinMacro(Buf, "__private_extern__=extern");
  }
  if (PP.getLangOptions().Microsoft) {
    DefineBuiltinMacro(Buf, "__stdcall=");
    DefineBuiltinMacro(Buf, "__cdecl=");
    DefineBuiltinMacro(Buf, "_cdecl=");
    DefineBuiltinMacro(Buf, "__ptr64=");
    DefineBuiltinMacro(Buf, "__w64=");
    DefineBuiltinMacro(Buf, "__forceinline=");
    DefineBuiltinMacro(Buf, "__int8=char");
    DefineBuiltinMacro(Buf, "__int16=short");
    DefineBuiltinMacro(Buf, "__int32=int");
    DefineBuiltinMacro(Buf, "__int64=long long");
    DefineBuiltinMacro(Buf, "__declspec(X)=");
  }
  // FIXME: Should emit a #line directive here.
}


/// EnterMainSourceFile - Enter the specified FileID as the main source file,
/// which implicitly adds the builtin defines etc.
void Preprocessor::EnterMainSourceFile() {
  
  unsigned MainFileID = SourceMgr.getMainFileID();
  
  // Enter the main file source buffer.
  EnterSourceFile(MainFileID, 0);
  
  // Tell the header info that the main file was entered.  If the file is later
  // #imported, it won't be re-entered.
  if (const FileEntry *FE = 
        SourceMgr.getFileEntryForLoc(SourceLocation::getFileLoc(MainFileID, 0)))
    HeaderInfo.IncrementIncludeCount(FE);
    
  std::vector<char> PrologFile;
  PrologFile.reserve(4080);
  
  // Install things like __POWERPC__, __GNUC__, etc into the macro table.
  InitializePredefinedMacros(*this, PrologFile);
  
  // Add on the predefines from the driver.
  PrologFile.insert(PrologFile.end(), Predefines,Predefines+strlen(Predefines));
  
  // Memory buffer must end with a null byte!
  PrologFile.push_back(0);

  // Now that we have emitted the predefined macros, #includes, etc into
  // PrologFile, preprocess it to populate the initial preprocessor state.
  llvm::MemoryBuffer *SB = 
    llvm::MemoryBuffer::getMemBufferCopy(&PrologFile.front(),&PrologFile.back(),
                                         "<predefines>");
  assert(SB && "Cannot fail to create predefined source buffer");
  unsigned FileID = SourceMgr.createFileIDForMemBuffer(SB);
  assert(FileID && "Could not create FileID for predefines?");
  
  // Start parsing the predefines.
  EnterSourceFile(FileID, 0);
}

//===----------------------------------------------------------------------===//
// Source File Location Methods.
//===----------------------------------------------------------------------===//

/// LookupFile - Given a "foo" or <foo> reference, look up the indicated file,
/// return null on failure.  isAngled indicates whether the file reference is
/// for system #include's or not (i.e. using <> instead of "").
const FileEntry *Preprocessor::LookupFile(const char *FilenameStart,
                                          const char *FilenameEnd,
                                          bool isAngled,
                                          const DirectoryLookup *FromDir,
                                          const DirectoryLookup *&CurDir) {
  // If the header lookup mechanism may be relative to the current file, pass in
  // info about where the current file is.
  const FileEntry *CurFileEnt = 0;
  if (!FromDir) {
    SourceLocation FileLoc = getCurrentFileLexer()->getFileLoc();
    CurFileEnt = SourceMgr.getFileEntryForLoc(FileLoc);
  }
  
  // Do a standard file entry lookup.
  CurDir = CurDirLookup;
  const FileEntry *FE =
    HeaderInfo.LookupFile(FilenameStart, FilenameEnd,
                          isAngled, FromDir, CurDir, CurFileEnt);
  if (FE) return FE;
  
  // Otherwise, see if this is a subframework header.  If so, this is relative
  // to one of the headers on the #include stack.  Walk the list of the current
  // headers on the #include stack and pass them to HeaderInfo.
  if (CurLexer && !CurLexer->Is_PragmaLexer) {
    if ((CurFileEnt = SourceMgr.getFileEntryForLoc(CurLexer->getFileLoc())))
      if ((FE = HeaderInfo.LookupSubframeworkHeader(FilenameStart, FilenameEnd,
                                                    CurFileEnt)))
        return FE;
  }
  
  for (unsigned i = 0, e = IncludeMacroStack.size(); i != e; ++i) {
    IncludeStackInfo &ISEntry = IncludeMacroStack[e-i-1];
    if (ISEntry.TheLexer && !ISEntry.TheLexer->Is_PragmaLexer) {
      if ((CurFileEnt = 
           SourceMgr.getFileEntryForLoc(ISEntry.TheLexer->getFileLoc())))
        if ((FE = HeaderInfo.LookupSubframeworkHeader(FilenameStart,
                                                      FilenameEnd, CurFileEnt)))
          return FE;
    }
  }
  
  // Otherwise, we really couldn't find the file.
  return 0;
}

/// isInPrimaryFile - Return true if we're in the top-level file, not in a
/// #include.
bool Preprocessor::isInPrimaryFile() const {
  if (CurLexer && !CurLexer->Is_PragmaLexer)
    return IncludeMacroStack.empty();
  
  // If there are any stacked lexers, we're in a #include.
  assert(IncludeMacroStack[0].TheLexer &&
         !IncludeMacroStack[0].TheLexer->Is_PragmaLexer &&
         "Top level include stack isn't our primary lexer?");
  for (unsigned i = 1, e = IncludeMacroStack.size(); i != e; ++i)
    if (IncludeMacroStack[i].TheLexer &&
        !IncludeMacroStack[i].TheLexer->Is_PragmaLexer)
      return false;
  return true;
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
                                   const DirectoryLookup *CurDir) {
  assert(CurMacroExpander == 0 && "Cannot #include a file inside a macro!");
  ++NumEnteredSourceFiles;
  
  if (MaxIncludeStackDepth < IncludeMacroStack.size())
    MaxIncludeStackDepth = IncludeMacroStack.size();

  Lexer *TheLexer = new Lexer(SourceLocation::getFileLoc(FileID, 0), *this);
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
  if (Callbacks && !CurLexer->Is_PragmaLexer) {
    DirectoryLookup::DirType FileType = DirectoryLookup::NormalHeaderDir;
    
    // Get the file entry for the current file.
    if (const FileEntry *FE = 
           SourceMgr.getFileEntryForLoc(CurLexer->getFileLoc()))
      FileType = HeaderInfo.getFileDirFlavor(FE);
    
    Callbacks->FileChanged(CurLexer->getFileLoc(),
                           PPCallbacks::EnterFile, FileType);
  }
}



/// EnterMacro - Add a Macro to the top of the include stack and start lexing
/// tokens from it instead of the current buffer.
void Preprocessor::EnterMacro(Token &Tok, MacroArgs *Args) {
  IncludeMacroStack.push_back(IncludeStackInfo(CurLexer, CurDirLookup,
                                               CurMacroExpander));
  CurLexer     = 0;
  CurDirLookup = 0;
  
  if (NumCachedMacroExpanders == 0) {
    CurMacroExpander = new TokenLexer(Tok, Args, *this);
  } else {
    CurMacroExpander = MacroExpanderCache[--NumCachedMacroExpanders];
    CurMacroExpander->Init(Tok, Args);
  }
}

/// EnterTokenStream - Add a "macro" context to the top of the include stack,
/// which will cause the lexer to start returning the specified tokens.  Note
/// that these tokens will be re-macro-expanded when/if expansion is enabled.
/// This method assumes that the specified stream of tokens has a permanent
/// owner somewhere, so they do not need to be copied.
void Preprocessor::EnterTokenStream(const Token *Toks, unsigned NumToks) {
  // Save our current state.
  IncludeMacroStack.push_back(IncludeStackInfo(CurLexer, CurDirLookup,
                                               CurMacroExpander));
  CurLexer     = 0;
  CurDirLookup = 0;

  // Create a macro expander to expand from the specified token stream.
  if (NumCachedMacroExpanders == 0) {
    CurMacroExpander = new TokenLexer(Toks, NumToks, *this);
  } else {
    CurMacroExpander = MacroExpanderCache[--NumCachedMacroExpanders];
    CurMacroExpander->Init(Toks, NumToks);
  }
}

/// RemoveTopOfLexerStack - Pop the current lexer/macro exp off the top of the
/// lexer stack.  This should only be used in situations where the current
/// state of the top-of-stack lexer is known.
void Preprocessor::RemoveTopOfLexerStack() {
  assert(!IncludeMacroStack.empty() && "Ran out of stack entries to load");
  
  if (CurMacroExpander) {
    // Delete or cache the now-dead macro expander.
    if (NumCachedMacroExpanders == MacroExpanderCacheSize)
      delete CurMacroExpander;
    else
      MacroExpanderCache[NumCachedMacroExpanders++] = CurMacroExpander;
  } else {
    delete CurLexer;
  }
  CurLexer         = IncludeMacroStack.back().TheLexer;
  CurDirLookup     = IncludeMacroStack.back().TheDirLookup;
  CurMacroExpander = IncludeMacroStack.back().TheMacroExpander;
  IncludeMacroStack.pop_back();
}

//===----------------------------------------------------------------------===//
// Macro Expansion Handling.
//===----------------------------------------------------------------------===//

/// setMacroInfo - Specify a macro for this identifier.
///
void Preprocessor::setMacroInfo(IdentifierInfo *II, MacroInfo *MI) {
  if (MI == 0) {
    if (II->hasMacroDefinition()) {
      Macros.erase(II);
      II->setHasMacroDefinition(false);
    }
  } else {
    Macros[II] = MI;
    II->setHasMacroDefinition(true);
  }
}

/// RegisterBuiltinMacro - Register the specified identifier in the identifier
/// table and mark it as a builtin macro to be expanded.
IdentifierInfo *Preprocessor::RegisterBuiltinMacro(const char *Name) {
  // Get the identifier.
  IdentifierInfo *Id = getIdentifierInfo(Name);
  
  // Mark it as being a macro that is builtin.
  MacroInfo *MI = new MacroInfo(SourceLocation());
  MI->setIsBuiltinMacro();
  setMacroInfo(Id, MI);
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
                                          const IdentifierInfo *MacroIdent,
                                          Preprocessor &PP) {
  IdentifierInfo *II = MI->getReplacementToken(0).getIdentifierInfo();

  // If the token isn't an identifier, it's always literally expanded.
  if (II == 0) return true;
  
  // If the identifier is a macro, and if that macro is enabled, it may be
  // expanded so it's not a trivial expansion.
  if (II->hasMacroDefinition() && PP.getMacroInfo(II)->isEnabled() &&
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
    // We have run off the end.  If it's a source file we don't
    // examine enclosing ones (C99 5.1.1.2p4).  Otherwise walk up the
    // macro stack.
    if (CurLexer)
      return false;
    for (unsigned i = IncludeMacroStack.size(); i != 0; --i) {
      IncludeStackInfo &Entry = IncludeMacroStack[i-1];
      if (Entry.TheLexer)
        Val = Entry.TheLexer->isNextPPTokenLParen();
      else
        Val = Entry.TheMacroExpander->isNextTokenLParen();
      
      if (Val != 2)
        break;
      
      // Ran off the end of a source file?
      if (Entry.TheLexer)
        return false;
    }
  }

  // Okay, if we know that the token is a '(', lex it and return.  Otherwise we
  // have found something that isn't a '(' or we found the end of the
  // translation unit.  In either case, return false.
  if (Val != 1)
    return false;
  
  Token Tok;
  LexUnexpandedToken(Tok);
  assert(Tok.is(tok::l_paren) && "Error computing l-paren-ness?");
  return true;
}

/// HandleMacroExpandedIdentifier - If an identifier token is read that is to be
/// expanded as a macro, handle it and return the next token as 'Identifier'.
bool Preprocessor::HandleMacroExpandedIdentifier(Token &Identifier, 
                                                 MacroInfo *MI) {
  // If this is a macro exapnsion in the "#if !defined(x)" line for the file,
  // then the macro could expand to different things in other contexts, we need
  // to disable the optimization in this case.
  if (CurLexer) CurLexer->MIOpt.ExpandedMacro();
  
  // If this is a builtin macro, like __LINE__ or _Pragma, handle it specially.
  if (MI->isBuiltinMacro()) {
    ExpandBuiltinMacro(Identifier);
    return false;
  }
  
  /// Args - If this is a function-like macro expansion, this contains,
  /// for each macro argument, the list of tokens that were provided to the
  /// invocation.
  MacroArgs *Args = 0;
  
  // If this is a function-like macro, read the arguments.
  if (MI->isFunctionLike()) {
    // C99 6.10.3p10: If the preprocessing token immediately after the the macro
    // name isn't a '(', this macro should not be expanded.  Otherwise, consume
    // it.
    if (!isNextPPTokenLParen())
      return true;
    
    // Remember that we are now parsing the arguments to a macro invocation.
    // Preprocessor directives used inside macro arguments are not portable, and
    // this enables the warning.
    InMacroArgs = true;
    Args = ReadFunctionLikeMacroArgs(Identifier, MI);
    
    // Finished parsing args.
    InMacroArgs = false;
    
    // If there was an error parsing the arguments, bail out.
    if (Args == 0) return false;
    
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
    // No need for arg info.
    if (Args) Args->destroy();
    
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
      if (IsAtStartOfLine) Identifier.setFlag(Token::StartOfLine);
      if (HadLeadingSpace) Identifier.setFlag(Token::LeadingSpace);
    }
    ++NumFastMacroExpanded;
    return false;
    
  } else if (MI->getNumTokens() == 1 &&
             isTrivialSingleTokenExpansion(MI, Identifier.getIdentifierInfo(),
                                           *this)){
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
    Identifier.setFlagValue(Token::StartOfLine , isAtStartOfLine);
    Identifier.setFlagValue(Token::LeadingSpace, hasLeadingSpace);
    
    // Update the tokens location to include both its logical and physical
    // locations.
    SourceLocation Loc =
      SourceMgr.getInstantiationLoc(Identifier.getLocation(), InstantiateLoc);
    Identifier.setLocation(Loc);
    
    // If this is #define X X, we must mark the result as unexpandible.
    if (IdentifierInfo *NewII = Identifier.getIdentifierInfo())
      if (getMacroInfo(NewII) == MI)
        Identifier.setFlag(Token::DisableExpand);
    
    // Since this is not an identifier token, it can't be macro expanded, so
    // we're done.
    ++NumFastMacroExpanded;
    return false;
  }
  
  // Start expanding the macro.
  EnterMacro(Identifier, Args);
  
  // Now that the macro is at the top of the include stack, ask the
  // preprocessor to read the next token from it.
  Lex(Identifier);
  return false;
}

/// ReadFunctionLikeMacroArgs - After reading "MACRO(", this method is
/// invoked to read all of the actual arguments specified for the macro
/// invocation.  This returns null on error.
MacroArgs *Preprocessor::ReadFunctionLikeMacroArgs(Token &MacroName,
                                                   MacroInfo *MI) {
  // The number of fixed arguments to parse.
  unsigned NumFixedArgsLeft = MI->getNumArgs();
  bool isVariadic = MI->isVariadic();
  
  // Outer loop, while there are more arguments, keep reading them.
  Token Tok;
  Tok.setKind(tok::comma);
  --NumFixedArgsLeft;  // Start reading the first arg.

  // ArgTokens - Build up a list of tokens that make up each argument.  Each
  // argument is separated by an EOF token.  Use a SmallVector so we can avoid
  // heap allocations in the common case.
  llvm::SmallVector<Token, 64> ArgTokens;

  unsigned NumActuals = 0;
  while (Tok.is(tok::comma)) {
    // C99 6.10.3p11: Keep track of the number of l_parens we have seen.  Note
    // that we already consumed the first one.
    unsigned NumParens = 0;
    
    while (1) {
      // Read arguments as unexpanded tokens.  This avoids issues, e.g., where
      // an argument value in a macro could expand to ',' or '(' or ')'.
      LexUnexpandedToken(Tok);
      
      if (Tok.is(tok::eof) || Tok.is(tok::eom)) { // "#if f(<eof>" & "#if f(\n"
        Diag(MacroName, diag::err_unterm_macro_invoc);
        // Do not lose the EOF/EOM.  Return it to the client.
        MacroName = Tok;
        return 0;
      } else if (Tok.is(tok::r_paren)) {
        // If we found the ) token, the macro arg list is done.
        if (NumParens-- == 0)
          break;
      } else if (Tok.is(tok::l_paren)) {
        ++NumParens;
      } else if (Tok.is(tok::comma) && NumParens == 0) {
        // Comma ends this argument if there are more fixed arguments expected.
        if (NumFixedArgsLeft)
          break;
        
        // If this is not a variadic macro, too many args were specified.
        if (!isVariadic) {
          // Emit the diagnostic at the macro name in case there is a missing ).
          // Emitting it at the , could be far away from the macro name.
          Diag(MacroName, diag::err_too_many_args_in_macro_invoc);
          return 0;
        }
        // Otherwise, continue to add the tokens to this variable argument.
      } else if (Tok.is(tok::comment) && !KeepMacroComments) {
        // If this is a comment token in the argument list and we're just in
        // -C mode (not -CC mode), discard the comment.
        continue;
      } else if (Tok.is(tok::identifier)) {
        // Reading macro arguments can cause macros that we are currently
        // expanding from to be popped off the expansion stack.  Doing so causes
        // them to be reenabled for expansion.  Here we record whether any
        // identifiers we lex as macro arguments correspond to disabled macros.
        // If so, we mark the token as noexpand.  This is a subtle aspect of 
        // C99 6.10.3.4p2.
        if (MacroInfo *MI = getMacroInfo(Tok.getIdentifierInfo()))
          if (!MI->isEnabled())
            Tok.setFlag(Token::DisableExpand);
      }
  
      ArgTokens.push_back(Tok);
    }

    // Empty arguments are standard in C99 and supported as an extension in
    // other modes.
    if (ArgTokens.empty() && !Features.C99)
      Diag(Tok, diag::ext_empty_fnmacro_arg);
    
    // Add a marker EOF token to the end of the token list for this argument.
    Token EOFTok;
    EOFTok.startToken();
    EOFTok.setKind(tok::eof);
    EOFTok.setLocation(Tok.getLocation());
    EOFTok.setLength(0);
    ArgTokens.push_back(EOFTok);
    ++NumActuals;
    --NumFixedArgsLeft;
  };
  
  // Okay, we either found the r_paren.  Check to see if we parsed too few
  // arguments.
  unsigned MinArgsExpected = MI->getNumArgs();
  
  // See MacroArgs instance var for description of this.
  bool isVarargsElided = false;
  
  if (NumActuals < MinArgsExpected) {
    // There are several cases where too few arguments is ok, handle them now.
    if (NumActuals+1 == MinArgsExpected && MI->isVariadic()) {
      // Varargs where the named vararg parameter is missing: ok as extension.
      // #define A(x, ...)
      // A("blah")
      Diag(Tok, diag::ext_missing_varargs_arg);

      // Remember this occurred if this is a C99 macro invocation with at least
      // one actual argument.
      isVarargsElided = MI->isC99Varargs() && MI->getNumArgs() > 1;
    } else if (MI->getNumArgs() == 1) {
      // #define A(x)
      //   A()
      // is ok because it is an empty argument.
      
      // Empty arguments are standard in C99 and supported as an extension in
      // other modes.
      if (ArgTokens.empty() && !Features.C99)
        Diag(Tok, diag::ext_empty_fnmacro_arg);
    } else {
      // Otherwise, emit the error.
      Diag(Tok, diag::err_too_few_args_in_macro_invoc);
      return 0;
    }
    
    // Add a marker EOF token to the end of the token list for this argument.
    SourceLocation EndLoc = Tok.getLocation();
    Tok.startToken();
    Tok.setKind(tok::eof);
    Tok.setLocation(EndLoc);
    Tok.setLength(0);
    ArgTokens.push_back(Tok);
  }
  
  return MacroArgs::create(MI, &ArgTokens[0], ArgTokens.size(),isVarargsElided);
}

/// ComputeDATE_TIME - Compute the current time, enter it into the specified
/// scratch buffer, then return DATELoc/TIMELoc locations with the position of
/// the identifier tokens inserted.
static void ComputeDATE_TIME(SourceLocation &DATELoc, SourceLocation &TIMELoc,
                             Preprocessor &PP) {
  time_t TT = time(0);
  struct tm *TM = localtime(&TT);
  
  static const char * const Months[] = {
    "Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"
  };
  
  char TmpBuffer[100];
  sprintf(TmpBuffer, "\"%s %2d %4d\"", Months[TM->tm_mon], TM->tm_mday, 
          TM->tm_year+1900);
  DATELoc = PP.CreateString(TmpBuffer, strlen(TmpBuffer));

  sprintf(TmpBuffer, "\"%02d:%02d:%02d\"", TM->tm_hour, TM->tm_min, TM->tm_sec);
  TIMELoc = PP.CreateString(TmpBuffer, strlen(TmpBuffer));
}

/// ExpandBuiltinMacro - If an identifier token is read that is to be expanded
/// as a builtin macro, handle it and return the next token as 'Tok'.
void Preprocessor::ExpandBuiltinMacro(Token &Tok) {
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
  Tok.setIdentifierInfo(0);
  Tok.clearFlag(Token::NeedsCleaning);
  
  if (II == Ident__LINE__) {
    // __LINE__ expands to a simple numeric value.
    sprintf(TmpBuffer, "%u", SourceMgr.getLogicalLineNumber(Tok.getLocation()));
    unsigned Length = strlen(TmpBuffer);
    Tok.setKind(tok::numeric_constant);
    Tok.setLength(Length);
    Tok.setLocation(CreateString(TmpBuffer, Length, Tok.getLocation()));
  } else if (II == Ident__FILE__ || II == Ident__BASE_FILE__) {
    SourceLocation Loc = Tok.getLocation();
    if (II == Ident__BASE_FILE__) {
      Diag(Tok, diag::ext_pp_base_file);
      SourceLocation NextLoc = SourceMgr.getIncludeLoc(Loc);
      while (NextLoc.isValid()) {
        Loc = NextLoc;
        NextLoc = SourceMgr.getIncludeLoc(Loc);
      }
    }
    
    // Escape this filename.  Turn '\' -> '\\' '"' -> '\"'
    std::string FN = SourceMgr.getSourceName(SourceMgr.getLogicalLoc(Loc));
    FN = '"' + Lexer::Stringify(FN) + '"';
    Tok.setKind(tok::string_literal);
    Tok.setLength(FN.size());
    Tok.setLocation(CreateString(&FN[0], FN.size(), Tok.getLocation()));
  } else if (II == Ident__DATE__) {
    if (!DATELoc.isValid())
      ComputeDATE_TIME(DATELoc, TIMELoc, *this);
    Tok.setKind(tok::string_literal);
    Tok.setLength(strlen("\"Mmm dd yyyy\""));
    Tok.setLocation(SourceMgr.getInstantiationLoc(DATELoc, Tok.getLocation()));
  } else if (II == Ident__TIME__) {
    if (!TIMELoc.isValid())
      ComputeDATE_TIME(DATELoc, TIMELoc, *this);
    Tok.setKind(tok::string_literal);
    Tok.setLength(strlen("\"hh:mm:ss\""));
    Tok.setLocation(SourceMgr.getInstantiationLoc(TIMELoc, Tok.getLocation()));
  } else if (II == Ident__INCLUDE_LEVEL__) {
    Diag(Tok, diag::ext_pp_include_level);

    // Compute the include depth of this token.
    unsigned Depth = 0;
    SourceLocation Loc = SourceMgr.getIncludeLoc(Tok.getLocation());
    for (; Loc.isValid(); ++Depth)
      Loc = SourceMgr.getIncludeLoc(Loc);
    
    // __INCLUDE_LEVEL__ expands to a simple numeric value.
    sprintf(TmpBuffer, "%u", Depth);
    unsigned Length = strlen(TmpBuffer);
    Tok.setKind(tok::numeric_constant);
    Tok.setLength(Length);
    Tok.setLocation(CreateString(TmpBuffer, Length, Tok.getLocation()));
  } else if (II == Ident__TIMESTAMP__) {
    // MSVC, ICC, GCC, VisualAge C++ extension.  The generated string should be
    // of the form "Ddd Mmm dd hh::mm::ss yyyy", which is returned by asctime.
    Diag(Tok, diag::ext_pp_timestamp);

    // Get the file that we are lexing out of.  If we're currently lexing from
    // a macro, dig into the include stack.
    const FileEntry *CurFile = 0;
    Lexer *TheLexer = getCurrentFileLexer();
    
    if (TheLexer)
      CurFile = SourceMgr.getFileEntryForLoc(TheLexer->getFileLoc());
    
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
    Tok.setKind(tok::string_literal);
    Tok.setLength(Len);
    Tok.setLocation(CreateString(TmpBuffer, Len, Tok.getLocation()));
  } else {
    assert(0 && "Unknown identifier!");
  }
}

//===----------------------------------------------------------------------===//
// Lexer Event Handling.
//===----------------------------------------------------------------------===//

/// LookUpIdentifierInfo - Given a tok::identifier token, look up the
/// identifier information for the token and install it into the token.
IdentifierInfo *Preprocessor::LookUpIdentifierInfo(Token &Identifier,
                                                   const char *BufPtr) {
  assert(Identifier.is(tok::identifier) && "Not an identifier!");
  assert(Identifier.getIdentifierInfo() == 0 && "Identinfo already exists!");
  
  // Look up this token, see if it is a macro, or if it is a language keyword.
  IdentifierInfo *II;
  if (BufPtr && !Identifier.needsCleaning()) {
    // No cleaning needed, just use the characters from the lexed buffer.
    II = getIdentifierInfo(BufPtr, BufPtr+Identifier.getLength());
  } else {
    // Cleaning needed, alloca a buffer, clean into it, then use the buffer.
    llvm::SmallVector<char, 64> IdentifierBuffer;
    IdentifierBuffer.resize(Identifier.getLength());
    const char *TmpBuf = &IdentifierBuffer[0];
    unsigned Size = getSpelling(Identifier, TmpBuf);
    II = getIdentifierInfo(TmpBuf, TmpBuf+Size);
  }
  Identifier.setIdentifierInfo(II);
  return II;
}


/// HandleIdentifier - This callback is invoked when the lexer reads an
/// identifier.  This callback looks up the identifier in the map and/or
/// potentially macro expands it or turns it into a named token (like 'for').
void Preprocessor::HandleIdentifier(Token &Identifier) {
  assert(Identifier.getIdentifierInfo() &&
         "Can't handle identifiers without identifier info!");
  
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
  if (MacroInfo *MI = getMacroInfo(&II)) {
    if (!DisableMacroExpansion && !Identifier.isExpandDisabled()) {
      if (MI->isEnabled()) {
        if (!HandleMacroExpandedIdentifier(Identifier, MI))
          return;
      } else {
        // C99 6.10.3.4p2 says that a disabled macro may never again be
        // expanded, even if it's in a context where it could be expanded in the
        // future.
        Identifier.setFlag(Token::DisableExpand);
      }
    }
  }

  // C++ 2.11p2: If this is an alternative representation of a C++ operator,
  // then we act as if it is the actual operator and not the textual
  // representation of it.
  if (II.isCPlusPlusOperatorKeyword())
    Identifier.setIdentifierInfo(0);

  // Change the kind of this identifier to the appropriate token kind, e.g.
  // turning "for" into a keyword.
  Identifier.setKind(II.getTokenID());
    
  // If this is an extension token, diagnose its use.
  // FIXME: tried (unsuccesfully) to shut this up when compiling with gnu99
  // For now, I'm just commenting it out (while I work on attributes).
  if (II.isExtensionToken() && Features.C99) 
    Diag(Identifier, diag::ext_token_used);
}

/// HandleEndOfFile - This callback is invoked when the lexer hits the end of
/// the current file.  This either returns the EOF token or pops a level off
/// the include stack and keeps going.
bool Preprocessor::HandleEndOfFile(Token &Result, bool isEndOfMacro) {
  assert(!CurMacroExpander &&
         "Ending a file when currently in a macro!");
  
  // See if this file had a controlling macro.
  if (CurLexer) {  // Not ending a macro, ignore it.
    if (const IdentifierInfo *ControllingMacro = 
          CurLexer->MIOpt.GetControllingMacroAtEndOfFile()) {
      // Okay, this has a controlling macro, remember in PerFileInfo.
      if (const FileEntry *FE = 
            SourceMgr.getFileEntryForLoc(CurLexer->getFileLoc()))
        HeaderInfo.SetFileControllingMacro(FE, ControllingMacro);
    }
  }
  
  // If this is a #include'd file, pop it off the include stack and continue
  // lexing the #includer file.
  if (!IncludeMacroStack.empty()) {
    // We're done with the #included file.
    RemoveTopOfLexerStack();

    // Notify the client, if desired, that we are in a new source file.
    if (Callbacks && !isEndOfMacro && CurLexer) {
      DirectoryLookup::DirType FileType = DirectoryLookup::NormalHeaderDir;
      
      // Get the file entry for the current file.
      if (const FileEntry *FE = 
            SourceMgr.getFileEntryForLoc(CurLexer->getFileLoc()))
        FileType = HeaderInfo.getFileDirFlavor(FE);

      Callbacks->FileChanged(CurLexer->getSourceLocation(CurLexer->BufferPtr),
                             PPCallbacks::ExitFile, FileType);
    }

    // Client should lex another token.
    return false;
  }

  // If the file ends with a newline, form the EOF token on the newline itself,
  // rather than "on the line following it", which doesn't exist.  This makes
  // diagnostics relating to the end of file include the last file that the user
  // actually typed, which is goodness.
  const char *EndPos = CurLexer->BufferEnd;
  if (EndPos != CurLexer->BufferStart && 
      (EndPos[-1] == '\n' || EndPos[-1] == '\r')) {
    --EndPos;
    
    // Handle \n\r and \r\n:
    if (EndPos != CurLexer->BufferStart && 
        (EndPos[-1] == '\n' || EndPos[-1] == '\r') &&
        EndPos[-1] != EndPos[0])
      --EndPos;
  }
  
  Result.startToken();
  CurLexer->BufferPtr = EndPos;
  CurLexer->FormTokenWithChars(Result, EndPos);
  Result.setKind(tok::eof);
  
  // We're done with the #included file.
  delete CurLexer;
  CurLexer = 0;

  // This is the end of the top-level file.  If the diag::pp_macro_not_used
  // diagnostic is enabled, look for macros that have not been used.
  if (Diags.getDiagnosticLevel(diag::pp_macro_not_used) != Diagnostic::Ignored){
    for (llvm::DenseMap<IdentifierInfo*, MacroInfo*>::iterator I =
         Macros.begin(), E = Macros.end(); I != E; ++I) {
      if (!I->second->isUsed())
        Diag(I->second->getDefinitionLoc(), diag::pp_macro_not_used);
    }
  }
  return true;
}

/// HandleEndOfMacro - This callback is invoked when the lexer hits the end of
/// the current macro expansion or token stream expansion.
bool Preprocessor::HandleEndOfMacro(Token &Result) {
  assert(CurMacroExpander && !CurLexer &&
         "Ending a macro when currently in a #include file!");

  // Delete or cache the now-dead macro expander.
  if (NumCachedMacroExpanders == MacroExpanderCacheSize)
    delete CurMacroExpander;
  else
    MacroExpanderCache[NumCachedMacroExpanders++] = CurMacroExpander;

  // Handle this like a #include file being popped off the stack.
  CurMacroExpander = 0;
  return HandleEndOfFile(Result, true);
}

/// HandleMicrosoftCommentPaste - When the macro expander pastes together a
/// comment (/##/) in microsoft mode, this method handles updating the current
/// state, returning the token on the next source line.
void Preprocessor::HandleMicrosoftCommentPaste(Token &Tok) {
  assert(CurMacroExpander && !CurLexer &&
         "Pasted comment can only be formed from macro");
  
  // We handle this by scanning for the closest real lexer, switching it to
  // raw mode and preprocessor mode.  This will cause it to return \n as an
  // explicit EOM token.
  Lexer *FoundLexer = 0;
  bool LexerWasInPPMode = false;
  for (unsigned i = 0, e = IncludeMacroStack.size(); i != e; ++i) {
    IncludeStackInfo &ISI = *(IncludeMacroStack.end()-i-1);
    if (ISI.TheLexer == 0) continue;  // Scan for a real lexer.
    
    // Once we find a real lexer, mark it as raw mode (disabling macro
    // expansions) and preprocessor mode (return EOM).  We know that the lexer
    // was *not* in raw mode before, because the macro that the comment came
    // from was expanded.  However, it could have already been in preprocessor
    // mode (#if COMMENT) in which case we have to return it to that mode and
    // return EOM.
    FoundLexer = ISI.TheLexer;
    FoundLexer->LexingRawMode = true;
    LexerWasInPPMode = FoundLexer->ParsingPreprocessorDirective;
    FoundLexer->ParsingPreprocessorDirective = true;
    break;
  }
  
  // Okay, we either found and switched over the lexer, or we didn't find a
  // lexer.  In either case, finish off the macro the comment came from, getting
  // the next token.
  if (!HandleEndOfMacro(Tok)) Lex(Tok);
  
  // Discarding comments as long as we don't have EOF or EOM.  This 'comments
  // out' the rest of the line, including any tokens that came from other macros
  // that were active, as in:
  //  #define submacro a COMMENT b
  //    submacro c
  // which should lex to 'a' only: 'b' and 'c' should be removed.
  while (Tok.isNot(tok::eom) && Tok.isNot(tok::eof))
    Lex(Tok);
  
  // If we got an eom token, then we successfully found the end of the line.
  if (Tok.is(tok::eom)) {
    assert(FoundLexer && "Can't get end of line without an active lexer");
    // Restore the lexer back to normal mode instead of raw mode.
    FoundLexer->LexingRawMode = false;
    
    // If the lexer was already in preprocessor mode, just return the EOM token
    // to finish the preprocessor line.
    if (LexerWasInPPMode) return;
    
    // Otherwise, switch out of PP mode and return the next lexed token.
    FoundLexer->ParsingPreprocessorDirective = false;
    return Lex(Tok);
  }
  
  // If we got an EOF token, then we reached the end of the token stream but
  // didn't find an explicit \n.  This can only happen if there was no lexer
  // active (an active lexer would return EOM at EOF if there was no \n in
  // preprocessor directive mode), so just return EOF as our token.
  assert(!FoundLexer && "Lexer should return EOM before EOF in PP mode");
  return;
}

