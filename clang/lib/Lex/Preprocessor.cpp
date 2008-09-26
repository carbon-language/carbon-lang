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
#include "clang/Lex/Pragma.h"
#include "clang/Lex/ScratchBuffer.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Streams.h"
using namespace clang;

//===----------------------------------------------------------------------===//

PreprocessorFactory::~PreprocessorFactory() {}

Preprocessor::Preprocessor(Diagnostic &diags, const LangOptions &opts,
                           TargetInfo &target, SourceManager &SM, 
                           HeaderSearch &Headers) 
  : Diags(diags), Features(opts), Target(target), FileMgr(Headers.getFileMgr()),
    SourceMgr(SM), HeaderInfo(Headers), Identifiers(opts),
    CurLexer(0), CurDirLookup(0), CurTokenLexer(0), Callbacks(0) {
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
  NumCachedTokenLexers = 0;

  CacheTokens = false;
  CachedLexPos = 0;

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
  assert(BacktrackPositions.empty() && "EnableBacktrack/Backtrack imbalance!");

  // Free any active lexers.
  delete CurLexer;
  
  while (!IncludeMacroStack.empty()) {
    delete IncludeMacroStack.back().TheLexer;
    delete IncludeMacroStack.back().TheTokenLexer;
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
  for (unsigned i = 0, e = NumCachedTokenLexers; i != e; ++i)
    delete TokenLexerCache[i];
  
  // Release pragma information.
  delete PragmaHandlers;

  // Delete the scratch buffer info.
  delete ScratchBuf;

  delete Callbacks;
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

void Preprocessor::Diag(SourceLocation Loc, unsigned DiagID,
                        const std::string &Msg,
                        const SourceRange &R1, const SourceRange &R2) {
  SourceRange R[] = {R1, R2};
  Diags.Report(getFullLoc(Loc), DiagID, &Msg, 1, R, 2);
}


void Preprocessor::Diag(SourceLocation Loc, unsigned DiagID,
                        const SourceRange &R) {
  Diags.Report(getFullLoc(Loc), DiagID, 0, 0, &R, 1);
}

void Preprocessor::Diag(SourceLocation Loc, unsigned DiagID,
                        const SourceRange &R1, const SourceRange &R2) {
  SourceRange R[] = {R1, R2};
  Diags.Report(getFullLoc(Loc), DiagID, 0, 0, R, 2);
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
             << SourceMgr.getColumnNumber(LogLoc);
  
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
  // These should all be defined in the preprocessor according to the
  // current language configuration.
  DefineBuiltinMacro(Buf, "__STDC__=1");
  //DefineBuiltinMacro(Buf, "__ASSEMBLER__=1");
  if (PP.getLangOptions().C99 && !PP.getLangOptions().CPlusPlus)
    DefineBuiltinMacro(Buf, "__STDC_VERSION__=199901L");
  else if (0) // STDC94 ?
    DefineBuiltinMacro(Buf, "__STDC_VERSION__=199409L");
  
  DefineBuiltinMacro(Buf, "__STDC_HOSTED__=1");
  if (PP.getLangOptions().ObjC1) {
    DefineBuiltinMacro(Buf, "__OBJC__=1");

    if (PP.getLangOptions().getGCMode() == LangOptions::NonGC) {
      DefineBuiltinMacro(Buf, "__weak=");
      DefineBuiltinMacro(Buf, "__strong=");
    } else {
      DefineBuiltinMacro(Buf, "__weak=__attribute__((objc_gc(weak)))");
      DefineBuiltinMacro(Buf, "__strong=__attribute__((objc_gc(strong)))");
      DefineBuiltinMacro(Buf, "__OBJC_GC__=1");
    }

    if (PP.getLangOptions().NeXTRuntime)
      DefineBuiltinMacro(Buf, "__NEXT_RUNTIME__=1");

    // darwin_constant_cfstrings controls this. This is also dependent
    // on other things like the runtime I believe.
    DefineBuiltinMacro(Buf, "__CONSTANT_CFSTRINGS__=1");
  }
  if (PP.getLangOptions().ObjC2)
    DefineBuiltinMacro(Buf, "OBJC_NEW_PROPERTIES");

  // Add __builtin_va_list typedef.
  {
    const char *VAList = PP.getTargetInfo().getVAListDeclaration();
    Buf.insert(Buf.end(), VAList, VAList+strlen(VAList));
    Buf.push_back('\n');
  }
  
  // Get the target #defines.
  PP.getTargetInfo().getTargetDefines(Buf);

  DefineBuiltinMacro(Buf, "__llvm__=1");   // LLVM Backend
  DefineBuiltinMacro(Buf, "__clang__=1");  // Clang Frontend
  
  // Compiler set macros.
  DefineBuiltinMacro(Buf, "__APPLE_CC__=5621");
  DefineBuiltinMacro(Buf, "__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__=1050");
  DefineBuiltinMacro(Buf, "__GNUC_MINOR__=2");
  DefineBuiltinMacro(Buf, "__GNUC_PATCHLEVEL__=1");
  DefineBuiltinMacro(Buf, "__GNUC__=4");
  DefineBuiltinMacro(Buf, "__GXX_ABI_VERSION=1002");
  DefineBuiltinMacro(Buf, "__VERSION__=\"4.2.1 (Apple Computer, Inc. "
                     "build 5621) (dot 3)\"");
  
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
  // Directly modeled after the attribute-based implementation in GCC. 
  if (PP.getLangOptions().Blocks) {
     DefineBuiltinMacro(Buf, "__block=__attribute__((__blocks__(byref)))");
     DefineBuiltinMacro(Buf, "__BLOCKS__=1");
  } else
    // This allows "__block int unusedVar;" even when blocks are disabled.
    // This is modeled after GCC's handling of __strong/__weak.
    DefineBuiltinMacro(Buf, "__block=");

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
  PrologFile.insert(PrologFile.end(), Predefines.begin(), Predefines.end());
  
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
  // We avoid diagnosing tokens that originate from macro definitions.
  if (II.isExtensionToken() && Features.C99 && !DisableMacroExpansion)
    Diag(Identifier, diag::ext_token_used);
}
