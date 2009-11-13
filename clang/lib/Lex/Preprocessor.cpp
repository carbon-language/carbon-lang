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
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/Pragma.h"
#include "clang/Lex/ScratchBuffer.h"
#include "clang/Lex/LexDiagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdio>
using namespace clang;

//===----------------------------------------------------------------------===//

Preprocessor::Preprocessor(Diagnostic &diags, const LangOptions &opts,
                           const TargetInfo &target, SourceManager &SM,
                           HeaderSearch &Headers,
                           IdentifierInfoLookup* IILookup,
                           bool OwnsHeaders)
  : Diags(&diags), Features(opts), Target(target),FileMgr(Headers.getFileMgr()),
    SourceMgr(SM), HeaderInfo(Headers), Identifiers(opts, IILookup),
    BuiltinInfo(Target), CurPPLexer(0), CurDirLookup(0), Callbacks(0) {
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
    // destroyed. We still need to run the dstor, however, to free
    // memory alocated by MacroInfo.
    I->second->Destroy(BP);
    I->first->setHasMacroDefinition(false);
  }

  // Free any cached macro expanders.
  for (unsigned i = 0, e = NumCachedTokenLexers; i != e; ++i)
    delete TokenLexerCache[i];

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
    llvm::errs() << " [UnClean='" << std::string(Start, Start+Tok.getLength())
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
  const char* TokStart = SourceMgr.getCharacterData(Tok.getLocation());
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
    Buffer = II->getNameStart();
    return II->getLength();
  }

  // Otherwise, compute the start of the token in the input lexer buffer.
  const char *TokStart = 0;

  if (Tok.isLiteral())
    TokStart = Tok.getLiteralData();

  if (TokStart == 0)
    TokStart = SourceMgr.getCharacterData(Tok.getLocation());

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
  const char *TokPtr = SourceMgr.getCharacterData(TokStart);

  // If they request the first char of the token, we're trivially done.
  if (CharNo == 0 && Lexer::isObviouslySimpleCharacter(*TokPtr))
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
    PhysOffset = Lexer::SkipEscapedNewLines(TokPtr)-TokPtr;

  return TokStart.getFileLocWithOffset(PhysOffset);
}

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
SourceLocation Preprocessor::getLocForEndOfToken(SourceLocation Loc) {
  if (Loc.isInvalid() || !Loc.isFileID())
    return SourceLocation();

  unsigned Len = Lexer::MeasureTokenLength(Loc, getSourceManager(), Features);
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
  EnterSourceFile(MainFileID, 0);

  // Tell the header info that the main file was entered.  If the file is later
  // #imported, it won't be re-entered.
  if (const FileEntry *FE = SourceMgr.getFileEntryForID(MainFileID))
    HeaderInfo.IncrementIncludeCount(FE);

  std::vector<char> PrologFile;
  PrologFile.reserve(4080);

  // FIXME: Don't make a copy.
  PrologFile.insert(PrologFile.end(), Predefines.begin(), Predefines.end());

  // Memory buffer must end with a null byte!
  PrologFile.push_back(0);

  // Now that we have emitted the predefined macros, #includes, etc into
  // PrologFile, preprocess it to populate the initial preprocessor state.
  llvm::MemoryBuffer *SB =
    llvm::MemoryBuffer::getMemBufferCopy(&PrologFile.front(),&PrologFile.back(),
                                         "<built-in>");
  assert(SB && "Cannot fail to create predefined source buffer");
  FileID FID = SourceMgr.createFileIDForMemBuffer(SB);
  assert(!FID.isInvalid() && "Could not create FileID for predefines?");

  // Start parsing the predefines.
  EnterSourceFile(FID, 0);
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
    llvm::SmallVector<char, 64> IdentifierBuffer;
    IdentifierBuffer.resize(Identifier.getLength());
    const char *TmpBuf = &IdentifierBuffer[0];
    unsigned Size = getSpelling(Identifier, TmpBuf);
    II = getIdentifierInfo(llvm::StringRef(TmpBuf, Size));
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

void Preprocessor::HandleComment(SourceRange Comment) {
  for (std::vector<CommentHandler *>::iterator H = CommentHandlers.begin(),
       HEnd = CommentHandlers.end();
       H != HEnd; ++H)
    (*H)->HandleComment(*this, Comment);
}

CommentHandler::~CommentHandler() { }
