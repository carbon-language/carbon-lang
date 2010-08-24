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
//   -d[DNI] - Dump various things.
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
#include "MacroArgs.h"
#include "clang/Lex/ExternalPreprocessorSource.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/Pragma.h"
#include "clang/Lex/PreprocessingRecord.h"
#include "clang/Lex/ScratchBuffer.h"
#include "clang/Lex/LexDiagnostic.h"
#include "clang/Lex/CodeCompletionHandler.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;

//===----------------------------------------------------------------------===//
ExternalPreprocessorSource::~ExternalPreprocessorSource() { }

Preprocessor::Preprocessor(Diagnostic &diags, const LangOptions &opts,
                           const TargetInfo &target, SourceManager &SM,
                           HeaderSearch &Headers,
                           IdentifierInfoLookup* IILookup,
                           bool OwnsHeaders)
  : Diags(&diags), Features(opts), Target(target),FileMgr(Headers.getFileMgr()),
    SourceMgr(SM), HeaderInfo(Headers), ExternalSource(0),
    Identifiers(opts, IILookup), BuiltinInfo(Target), CodeComplete(0),
    CodeCompletionFile(0), SkipMainFilePreamble(0, true), CurPPLexer(0), 
    CurDirLookup(0), Callbacks(0), MacroArgCache(0), Record(0) {
  ScratchBuf = new ScratchBuffer(SourceMgr);
  CounterValue = 0; // __COUNTER__ starts at 0.
  OwnsHeaderSearch = OwnsHeaders;

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

  CachedLexPos = 0;

  // We haven't read anything from the external source.
  ReadMacrosFromExternalSource = false;

  // "Poison" __VA_ARGS__, which can only appear in the expansion of a macro.
  // This gets unpoisoned where it is allowed.
  (Ident__VA_ARGS__ = getIdentifierInfo("__VA_ARGS__"))->setIsPoisoned();

  // Initialize the pragma handlers.
  PragmaHandlers = new PragmaNamespace(llvm::StringRef());
  RegisterBuiltinPragmas();

  // Initialize builtin macros like __LINE__ and friends.
  RegisterBuiltinMacros();
}

Preprocessor::~Preprocessor() {
  assert(BacktrackPositions.empty() && "EnableBacktrack/Backtrack imbalance!");

  while (!IncludeMacroStack.empty()) {
    delete IncludeMacroStack.back().TheLexer;
    delete IncludeMacroStack.back().TheTokenLexer;
    IncludeMacroStack.pop_back();
  }

  // Free any macro definitions.
  for (llvm::DenseMap<IdentifierInfo*, MacroInfo*>::iterator I =
       Macros.begin(), E = Macros.end(); I != E; ++I) {
    // We don't need to free the MacroInfo objects directly.  These
    // will be released when the BumpPtrAllocator 'BP' object gets
    // destroyed.  We still need to run the dtor, however, to free
    // memory alocated by MacroInfo.
    I->second->Destroy();
    I->first->setHasMacroDefinition(false);
  }
  for (std::vector<MacroInfo*>::iterator I = MICache.begin(),
                                         E = MICache.end(); I != E; ++I) {
    // We don't need to free the MacroInfo objects directly.  These
    // will be released when the BumpPtrAllocator 'BP' object gets
    // destroyed.  We still need to run the dtor, however, to free
    // memory alocated by MacroInfo.
    (*I)->Destroy();
  }

  // Free any cached macro expanders.
  for (unsigned i = 0, e = NumCachedTokenLexers; i != e; ++i)
    delete TokenLexerCache[i];

  // Free any cached MacroArgs.
  for (MacroArgs *ArgList = MacroArgCache; ArgList; )
    ArgList = ArgList->deallocate();

  // Release pragma information.
  delete PragmaHandlers;

  // Delete the scratch buffer info.
  delete ScratchBuf;

  // Delete the header search info, if we own it.
  if (OwnsHeaderSearch)
    delete &HeaderInfo;

  delete Callbacks;
}

void Preprocessor::setPTHManager(PTHManager* pm) {
  PTH.reset(pm);
  FileMgr.addStatCache(PTH->createStatCache());
}

void Preprocessor::DumpToken(const Token &Tok, bool DumpFlags) const {
  llvm::errs() << tok::getTokenName(Tok.getKind()) << " '"
               << getSpelling(Tok) << "'";

  if (!DumpFlags) return;

  llvm::errs() << "\t";
  if (Tok.isAtStartOfLine())
    llvm::errs() << " [StartOfLine]";
  if (Tok.hasLeadingSpace())
    llvm::errs() << " [LeadingSpace]";
  if (Tok.isExpandDisabled())
    llvm::errs() << " [ExpandDisabled]";
  if (Tok.needsCleaning()) {
    const char *Start = SourceMgr.getCharacterData(Tok.getLocation());
    llvm::errs() << " [UnClean='" << llvm::StringRef(Start, Tok.getLength())
                 << "']";
  }

  llvm::errs() << "\tLoc=<";
  DumpLocation(Tok.getLocation());
  llvm::errs() << ">";
}

void Preprocessor::DumpLocation(SourceLocation Loc) const {
  Loc.dump(SourceMgr);
}

void Preprocessor::DumpMacro(const MacroInfo &MI) const {
  llvm::errs() << "MACRO: ";
  for (unsigned i = 0, e = MI.getNumTokens(); i != e; ++i) {
    DumpToken(MI.getReplacementToken(i));
    llvm::errs() << "  ";
  }
  llvm::errs() << "\n";
}

void Preprocessor::PrintStats() {
  llvm::errs() << "\n*** Preprocessor Stats:\n";
  llvm::errs() << NumDirectives << " directives found:\n";
  llvm::errs() << "  " << NumDefined << " #define.\n";
  llvm::errs() << "  " << NumUndefined << " #undef.\n";
  llvm::errs() << "  #include/#include_next/#import:\n";
  llvm::errs() << "    " << NumEnteredSourceFiles << " source files entered.\n";
  llvm::errs() << "    " << MaxIncludeStackDepth << " max include stack depth\n";
  llvm::errs() << "  " << NumIf << " #if/#ifndef/#ifdef.\n";
  llvm::errs() << "  " << NumElse << " #else/#elif.\n";
  llvm::errs() << "  " << NumEndif << " #endif.\n";
  llvm::errs() << "  " << NumPragma << " #pragma.\n";
  llvm::errs() << NumSkipped << " #if/#ifndef#ifdef regions skipped\n";

  llvm::errs() << NumMacroExpanded << "/" << NumFnMacroExpanded << "/"
             << NumBuiltinMacroExpanded << " obj/fn/builtin macros expanded, "
             << NumFastMacroExpanded << " on the fast path.\n";
  llvm::errs() << (NumFastTokenPaste+NumTokenPaste)
             << " token paste (##) operations performed, "
             << NumFastTokenPaste << " on the fast path.\n";
}

Preprocessor::macro_iterator
Preprocessor::macro_begin(bool IncludeExternalMacros) const {
  if (IncludeExternalMacros && ExternalSource &&
      !ReadMacrosFromExternalSource) {
    ReadMacrosFromExternalSource = true;
    ExternalSource->ReadDefinedMacros();
  }

  return Macros.begin();
}

Preprocessor::macro_iterator
Preprocessor::macro_end(bool IncludeExternalMacros) const {
  if (IncludeExternalMacros && ExternalSource &&
      !ReadMacrosFromExternalSource) {
    ReadMacrosFromExternalSource = true;
    ExternalSource->ReadDefinedMacros();
  }

  return Macros.end();
}

bool Preprocessor::SetCodeCompletionPoint(const FileEntry *File,
                                          unsigned TruncateAtLine,
                                          unsigned TruncateAtColumn) {
  using llvm::MemoryBuffer;

  CodeCompletionFile = File;

  // Okay to clear out the code-completion point by passing NULL.
  if (!CodeCompletionFile)
    return false;

  // Load the actual file's contents.
  bool Invalid = false;
  const MemoryBuffer *Buffer = SourceMgr.getMemoryBufferForFile(File, &Invalid);
  if (Invalid)
    return true;

  // Find the byte position of the truncation point.
  const char *Position = Buffer->getBufferStart();
  for (unsigned Line = 1; Line < TruncateAtLine; ++Line) {
    for (; *Position; ++Position) {
      if (*Position != '\r' && *Position != '\n')
        continue;

      // Eat \r\n or \n\r as a single line.
      if ((Position[1] == '\r' || Position[1] == '\n') &&
          Position[0] != Position[1])
        ++Position;
      ++Position;
      break;
    }
  }

  Position += TruncateAtColumn - 1;

  // Truncate the buffer.
  if (Position < Buffer->getBufferEnd()) {
    llvm::StringRef Data(Buffer->getBufferStart(),
                         Position-Buffer->getBufferStart());
    MemoryBuffer *TruncatedBuffer
      = MemoryBuffer::getMemBufferCopy(Data, Buffer->getBufferIdentifier());
    SourceMgr.overrideFileContents(File, TruncatedBuffer);
  }

  return false;
}

bool Preprocessor::isCodeCompletionFile(SourceLocation FileLoc) const {
  return CodeCompletionFile && FileLoc.isFileID() &&
    SourceMgr.getFileEntryForID(SourceMgr.getFileID(FileLoc))
      == CodeCompletionFile;
}

//===----------------------------------------------------------------------===//
// Token Spelling
//===----------------------------------------------------------------------===//

/// getSpelling() - Return the 'spelling' of this token.  The spelling of a
/// token are the characters used to represent the token in the source file
/// after trigraph expansion and escaped-newline folding.  In particular, this
/// wants to get the true, uncanonicalized, spelling of things like digraphs
/// UCNs, etc.
std::string Preprocessor::getSpelling(const Token &Tok,
                                      const SourceManager &SourceMgr,
                                      const LangOptions &Features, 
                                      bool *Invalid) {
  assert((int)Tok.getLength() >= 0 && "Token character range is bogus!");

  // If this token contains nothing interesting, return it directly.
  bool CharDataInvalid = false;
  const char* TokStart = SourceMgr.getCharacterData(Tok.getLocation(), 
                                                    &CharDataInvalid);
  if (Invalid)
    *Invalid = CharDataInvalid;
  if (CharDataInvalid)
    return std::string();

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

/// getSpelling() - Return the 'spelling' of this token.  The spelling of a
/// token are the characters used to represent the token in the source file
/// after trigraph expansion and escaped-newline folding.  In particular, this
/// wants to get the true, uncanonicalized, spelling of things like digraphs
/// UCNs, etc.
std::string Preprocessor::getSpelling(const Token &Tok, bool *Invalid) const {
  return getSpelling(Tok, SourceMgr, Features, Invalid);
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
                                   const char *&Buffer, bool *Invalid) const {
  assert((int)Tok.getLength() >= 0 && "Token character range is bogus!");

  // If this token is an identifier, just return the string from the identifier
  // table, which is very quick.
  if (const IdentifierInfo *II = Tok.getIdentifierInfo()) {
    Buffer = II->getNameStart();
    return II->getLength();
  }

  // Otherwise, compute the start of the token in the input lexer buffer.
  const char *TokStart = 0;

  if (Tok.isLiteral())
    TokStart = Tok.getLiteralData();

  if (TokStart == 0) {
    bool CharDataInvalid = false;
    TokStart = SourceMgr.getCharacterData(Tok.getLocation(), &CharDataInvalid);
    if (Invalid)
      *Invalid = CharDataInvalid;
    if (CharDataInvalid) {
      Buffer = "";
      return 0;
    }
  }

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

/// getSpelling - This method is used to get the spelling of a token into a
/// SmallVector. Note that the returned StringRef may not point to the
/// supplied buffer if a copy can be avoided.
llvm::StringRef Preprocessor::getSpelling(const Token &Tok,
                                          llvm::SmallVectorImpl<char> &Buffer,
                                          bool *Invalid) const {
  // Try the fast path.
  if (const IdentifierInfo *II = Tok.getIdentifierInfo())
    return II->getName();

  // Resize the buffer if we need to copy into it.
  if (Tok.needsCleaning())
    Buffer.resize(Tok.getLength());

  const char *Ptr = Buffer.data();
  unsigned Len = getSpelling(Tok, Ptr, Invalid);
  return llvm::StringRef(Ptr, Len);
}

/// CreateString - Plop the specified string into a scratch buffer and return a
/// location for it.  If specified, the source location provides a source
/// location for the token.
void Preprocessor::CreateString(const char *Buf, unsigned Len, Token &Tok,
                                SourceLocation InstantiationLoc) {
  Tok.setLength(Len);

  const char *DestPtr;
  SourceLocation Loc = ScratchBuf->getToken(Buf, Len, DestPtr);

  if (InstantiationLoc.isValid())
    Loc = SourceMgr.createInstantiationLoc(Loc, InstantiationLoc,
                                           InstantiationLoc, Len);
  Tok.setLocation(Loc);

  // If this is a literal token, set the pointer data.
  if (Tok.isLiteral())
    Tok.setLiteralData(DestPtr);
}


/// AdvanceToTokenCharacter - Given a location that specifies the start of a
/// token, return a new location that specifies a character within the token.
SourceLocation Preprocessor::AdvanceToTokenCharacter(SourceLocation TokStart,
                                                     unsigned CharNo) {
  // Figure out how many physical characters away the specified instantiation
  // character is.  This needs to take into consideration newlines and
  // trigraphs.
  bool Invalid = false;
  const char *TokPtr = SourceMgr.getCharacterData(TokStart, &Invalid);

  // If they request the first char of the token, we're trivially done.
  if (Invalid || (CharNo == 0 && Lexer::isObviouslySimpleCharacter(*TokPtr)))
    return TokStart;

  unsigned PhysOffset = 0;

  // The usual case is that tokens don't contain anything interesting.  Skip
  // over the uninteresting characters.  If a token only consists of simple
  // chars, this method is extremely fast.
  while (Lexer::isObviouslySimpleCharacter(*TokPtr)) {
    if (CharNo == 0)
      return TokStart.getFileLocWithOffset(PhysOffset);
    ++TokPtr, --CharNo, ++PhysOffset;
  }

  // If we have a character that may be a trigraph or escaped newline, use a
  // lexer to parse it correctly.
  for (; CharNo; --CharNo) {
    unsigned Size;
    Lexer::getCharAndSizeNoWarn(TokPtr, Size, Features);
    TokPtr += Size;
    PhysOffset += Size;
  }

  // Final detail: if we end up on an escaped newline, we want to return the
  // location of the actual byte of the token.  For example foo\<newline>bar
  // advanced by 3 should return the location of b, not of \\.  One compounding
  // detail of this is that the escape may be made by a trigraph.
  if (!Lexer::isObviouslySimpleCharacter(*TokPtr))
    PhysOffset += Lexer::SkipEscapedNewLines(TokPtr)-TokPtr;

  return TokStart.getFileLocWithOffset(PhysOffset);
}

SourceLocation Preprocessor::getLocForEndOfToken(SourceLocation Loc,
                                                 unsigned Offset) {
  if (Loc.isInvalid() || !Loc.isFileID())
    return SourceLocation();

  unsigned Len = Lexer::MeasureTokenLength(Loc, getSourceManager(), Features);
  if (Len > Offset)
    Len = Len - Offset;
  else
    return Loc;

  return AdvanceToTokenCharacter(Loc, Len);
}



//===----------------------------------------------------------------------===//
// Preprocessor Initialization Methods
//===----------------------------------------------------------------------===//


/// EnterMainSourceFile - Enter the specified FileID as the main source file,
/// which implicitly adds the builtin defines etc.
void Preprocessor::EnterMainSourceFile() {
  // We do not allow the preprocessor to reenter the main file.  Doing so will
  // cause FileID's to accumulate information from both runs (e.g. #line
  // information) and predefined macros aren't guaranteed to be set properly.
  assert(NumEnteredSourceFiles == 0 && "Cannot reenter the main file!");
  FileID MainFileID = SourceMgr.getMainFileID();

  // Enter the main file source buffer.
  EnterSourceFile(MainFileID, 0, SourceLocation());

  // If we've been asked to skip bytes in the main file (e.g., as part of a
  // precompiled preamble), do so now.
  if (SkipMainFilePreamble.first > 0)
    CurLexer->SkipBytes(SkipMainFilePreamble.first, 
                        SkipMainFilePreamble.second);
  
  // Tell the header info that the main file was entered.  If the file is later
  // #imported, it won't be re-entered.
  if (const FileEntry *FE = SourceMgr.getFileEntryForID(MainFileID))
    HeaderInfo.IncrementIncludeCount(FE);

  // Preprocess Predefines to populate the initial preprocessor state.
  llvm::MemoryBuffer *SB =
    llvm::MemoryBuffer::getMemBufferCopy(Predefines, "<built-in>");
  assert(SB && "Cannot fail to create predefined source buffer");
  FileID FID = SourceMgr.createFileIDForMemBuffer(SB);
  assert(!FID.isInvalid() && "Could not create FileID for predefines?");

  // Start parsing the predefines.
  EnterSourceFile(FID, 0, SourceLocation());
}

void Preprocessor::EndSourceFile() {
  // Notify the client that we reached the end of the source file.
  if (Callbacks)
    Callbacks->EndOfMainFile();
}

//===----------------------------------------------------------------------===//
// Lexer Event Handling.
//===----------------------------------------------------------------------===//

/// LookUpIdentifierInfo - Given a tok::identifier token, look up the
/// identifier information for the token and install it into the token.
IdentifierInfo *Preprocessor::LookUpIdentifierInfo(Token &Identifier,
                                                   const char *BufPtr) const {
  assert(Identifier.is(tok::identifier) && "Not an identifier!");
  assert(Identifier.getIdentifierInfo() == 0 && "Identinfo already exists!");

  // Look up this token, see if it is a macro, or if it is a language keyword.
  IdentifierInfo *II;
  if (BufPtr && !Identifier.needsCleaning()) {
    // No cleaning needed, just use the characters from the lexed buffer.
    II = getIdentifierInfo(llvm::StringRef(BufPtr, Identifier.getLength()));
  } else {
    // Cleaning needed, alloca a buffer, clean into it, then use the buffer.
    llvm::SmallString<64> IdentifierBuffer;
    llvm::StringRef CleanedStr = getSpelling(Identifier, IdentifierBuffer);
    II = getIdentifierInfo(CleanedStr);
  }
  Identifier.setIdentifierInfo(II);
  return II;
}


/// HandleIdentifier - This callback is invoked when the lexer reads an
/// identifier.  This callback looks up the identifier in the map and/or
/// potentially macro expands it or turns it into a named token (like 'for').
///
/// Note that callers of this method are guarded by checking the
/// IdentifierInfo's 'isHandleIdentifierCase' bit.  If this method changes, the
/// IdentifierInfo methods that compute these properties will need to change to
/// match.
void Preprocessor::HandleIdentifier(Token &Identifier) {
  assert(Identifier.getIdentifierInfo() &&
         "Can't handle identifiers without identifier info!");

  IdentifierInfo &II = *Identifier.getIdentifierInfo();

  // If this identifier was poisoned, and if it was not produced from a macro
  // expansion, emit an error.
  if (II.isPoisoned() && CurPPLexer) {
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

  // If this is an extension token, diagnose its use.
  // We avoid diagnosing tokens that originate from macro definitions.
  // FIXME: This warning is disabled in cases where it shouldn't be,
  // like "#define TY typeof", "TY(1) x".
  if (II.isExtensionToken() && !DisableMacroExpansion)
    Diag(Identifier, diag::ext_token_used);
}

void Preprocessor::AddCommentHandler(CommentHandler *Handler) {
  assert(Handler && "NULL comment handler");
  assert(std::find(CommentHandlers.begin(), CommentHandlers.end(), Handler) ==
         CommentHandlers.end() && "Comment handler already registered");
  CommentHandlers.push_back(Handler);
}

void Preprocessor::RemoveCommentHandler(CommentHandler *Handler) {
  std::vector<CommentHandler *>::iterator Pos
  = std::find(CommentHandlers.begin(), CommentHandlers.end(), Handler);
  assert(Pos != CommentHandlers.end() && "Comment handler not registered");
  CommentHandlers.erase(Pos);
}

bool Preprocessor::HandleComment(Token &result, SourceRange Comment) {
  bool AnyPendingTokens = false;
  for (std::vector<CommentHandler *>::iterator H = CommentHandlers.begin(),
       HEnd = CommentHandlers.end();
       H != HEnd; ++H) {
    if ((*H)->HandleComment(*this, Comment))
      AnyPendingTokens = true;
  }
  if (!AnyPendingTokens || getCommentRetentionState())
    return false;
  Lex(result);
  return true;
}

CommentHandler::~CommentHandler() { }

CodeCompletionHandler::~CodeCompletionHandler() { }

void Preprocessor::createPreprocessingRecord() {
  if (Record)
    return;
  
  Record = new PreprocessingRecord;
  addPPCallbacks(Record);
}
