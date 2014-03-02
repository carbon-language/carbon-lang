//===-- Core/IncludeDirectives.cpp - Include directives handling ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file defines the IncludeDirectives class that helps with
/// detecting and modifying \#include directives.
///
//===----------------------------------------------------------------------===//

#include "IncludeDirectives.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include <stack>

using namespace clang;
using namespace clang::tooling;
using llvm::StringRef;

/// \brief PPCallbacks that fills-in the include information in the given
/// \c IncludeDirectives.
class IncludeDirectivesPPCallback : public clang::PPCallbacks {
  // Struct helping the detection of header guards in the various callbacks
  struct GuardDetection {
    GuardDetection(FileID FID)
      : FID(FID), Count(0), TheMacro(0), CountAtEndif(0) {}

    FileID FID;
    // count for relevant preprocessor directives
    unsigned Count;
    // the macro that is tested in the top most ifndef for the header guard
    // (e.g: GUARD_H)
    const IdentifierInfo *TheMacro;
    // the hash locations of #ifndef, #define, #endif
    SourceLocation IfndefLoc, DefineLoc, EndifLoc;
    // the value of Count once the #endif is reached
    unsigned CountAtEndif;

    /// \brief Check that with all the information gathered if this is a
    /// potential header guard.
    ///
    /// Meaning a top-most \#ifndef has been found, followed by a define and the
    /// last preprocessor directive was the terminating \#endif.
    ///
    /// FIXME: accept the \#if !defined identifier form too.
    bool isPotentialHeaderGuard() const {
      return Count == CountAtEndif && DefineLoc.isValid();
    }
  };

public:
  IncludeDirectivesPPCallback(IncludeDirectives *Self) : Self(Self), Guard(0) {}

private:
  virtual ~IncludeDirectivesPPCallback() {}
  void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                          StringRef FileName, bool IsAngled,
                          CharSourceRange FilenameRange, const FileEntry *File,
                          StringRef SearchPath, StringRef RelativePath,
                          const Module *Imported) override {
    SourceManager &SM = Self->Sources;
    const FileEntry *FE = SM.getFileEntryForID(SM.getFileID(HashLoc));
    assert(FE && "Valid file expected.");

    IncludeDirectives::Entry E(HashLoc, File, IsAngled);
    Self->FileToEntries[FE].push_back(E);
    Self->IncludeAsWrittenToLocationsMap[FileName].push_back(HashLoc);
  }

  // Keep track of the current file in the stack
  virtual void FileChanged(SourceLocation Loc, FileChangeReason Reason,
                           SrcMgr::CharacteristicKind FileType,
                           FileID PrevFID) override {
    SourceManager &SM = Self->Sources;
    switch (Reason) {
    case EnterFile:
      Files.push(GuardDetection(SM.getFileID(Loc)));
      Guard = &Files.top();
      break;

    case ExitFile:
      if (Guard->isPotentialHeaderGuard())
        handlePotentialHeaderGuard(*Guard);
      Files.pop();
      Guard = &Files.top();
      break;

    default:
      break;
    }
  }

  /// \brief Mark this header as guarded in the IncludeDirectives if it's a
  /// proper header guard.
  void handlePotentialHeaderGuard(const GuardDetection &Guard) {
    SourceManager &SM = Self->Sources;
    const FileEntry *File = SM.getFileEntryForID(Guard.FID);
    const LangOptions &LangOpts = Self->CI.getLangOpts();

    // Null file can happen for the <built-in> buffer for example. They
    // shouldn't have header guards though...
    if (!File)
      return;

    // The #ifndef should be the next thing after the preamble. We aren't
    // checking for equality because it can also be part of the preamble if the
    // preamble is the whole file.
    unsigned Preamble =
        Lexer::ComputePreamble(SM.getBuffer(Guard.FID), LangOpts).first;
    unsigned IfndefOffset = SM.getFileOffset(Guard.IfndefLoc);
    if (IfndefOffset > (Preamble + 1))
      return;

    // No code is allowed in the code remaining after the #endif.
    const llvm::MemoryBuffer *Buffer = SM.getBuffer(Guard.FID);
    Lexer Lex(SM.getLocForStartOfFile(Guard.FID), LangOpts,
              Buffer->getBufferStart(),
              Buffer->getBufferStart() + SM.getFileOffset(Guard.EndifLoc),
              Buffer->getBufferEnd());

    // Find the first newline not part of a multi-line comment.
    Token Tok;
    Lex.LexFromRawLexer(Tok); // skip endif
    Lex.LexFromRawLexer(Tok);

    // Not a proper header guard, the remainder of the file contains something
    // else than comments or whitespaces.
    if (Tok.isNot(tok::eof))
      return;

    // Add to the location of the define to the IncludeDirectives for this file.
    Self->HeaderToGuard[File] = Guard.DefineLoc;
  }

  virtual void Ifndef(SourceLocation Loc, const Token &MacroNameTok,
                      const MacroDirective *MD) override {
    Guard->Count++;

    // If this #ifndef is the top-most directive and the symbol isn't defined
    // store those information in the guard detection, the next step will be to
    // check for the define.
    if (Guard->Count == 1 && MD == 0) {
      IdentifierInfo *MII = MacroNameTok.getIdentifierInfo();

      if (MII->hasMacroDefinition())
        return;
      Guard->IfndefLoc = Loc;
      Guard->TheMacro = MII;
    }
  }

  virtual void MacroDefined(const Token &MacroNameTok,
                            const MacroDirective *MD) override {
    Guard->Count++;

    // If this #define is the second directive of the file and the symbol
    // defined is the same as the one checked in the #ifndef then store the
    // information about this define.
    if (Guard->Count == 2 && Guard->TheMacro != 0) {
      IdentifierInfo *MII = MacroNameTok.getIdentifierInfo();

      // macro unrelated to the ifndef, doesn't look like a proper header guard
      if (MII->getName() != Guard->TheMacro->getName())
        return;

      Guard->DefineLoc = MacroNameTok.getLocation();
    }
  }

  virtual void Endif(SourceLocation Loc, SourceLocation IfLoc) override {
    Guard->Count++;

    // If it's the #endif corresponding to the top-most #ifndef
    if (Self->Sources.getDecomposedLoc(Guard->IfndefLoc) !=
        Self->Sources.getDecomposedLoc(IfLoc))
      return;

    // And that the top-most #ifndef was followed by the right #define
    if (Guard->DefineLoc.isInvalid())
      return;

    // Then save the information about this #endif. Once the file is exited we
    // will check if it was the final preprocessor directive.
    Guard->CountAtEndif = Guard->Count;
    Guard->EndifLoc = Loc;
  }

  virtual void MacroExpands(const Token &, const MacroDirective *, SourceRange,
                            const MacroArgs *) override {
    Guard->Count++;
  }
  virtual void MacroUndefined(const Token &,
                              const MacroDirective *) override {
    Guard->Count++;
  }
  virtual void Defined(const Token &, const MacroDirective *,
                       SourceRange) override {
    Guard->Count++;
  }
  virtual void If(SourceLocation, SourceRange,
                  ConditionValueKind) override {
    Guard->Count++;
  }
  virtual void Elif(SourceLocation, SourceRange, ConditionValueKind,
                    SourceLocation) override {
    Guard->Count++;
  }
  virtual void Ifdef(SourceLocation, const Token &,
                     const MacroDirective *) override {
    Guard->Count++;
  }
  virtual void Else(SourceLocation, SourceLocation) override {
    Guard->Count++;
  }

  IncludeDirectives *Self;
  // keep track of the guard info through the include stack
  std::stack<GuardDetection> Files;
  // convenience field pointing to Files.top().second
  GuardDetection *Guard;
};

// Flags that describes where to insert newlines.
enum NewLineFlags {
  // Prepend a newline at the beginning of the insertion.
  NL_Prepend = 0x1,

  // Prepend another newline at the end of the insertion.
  NL_PrependAnother = 0x2,

  // Add two newlines at the end of the insertion.
  NL_AppendTwice = 0x4,

  // Convenience value to enable both \c NL_Prepend and \c NL_PrependAnother.
  NL_PrependTwice = NL_Prepend | NL_PrependAnother
};

/// \brief Guess the end-of-line sequence used in the given FileID. If the
/// sequence can't be guessed return an Unix-style newline.
static StringRef guessEOL(SourceManager &SM, FileID ID) {
  StringRef Content = SM.getBufferData(ID);
  StringRef Buffer = Content.substr(Content.find_first_of("\r\n"));

  return llvm::StringSwitch<StringRef>(Buffer)
      .StartsWith("\r\n", "\r\n")
      .StartsWith("\n\r", "\n\r")
      .StartsWith("\r", "\r")
      .Default("\n");
}

/// \brief Find the end of the end of the directive, either the beginning of a
/// newline or the end of file.
//
// \return The offset into the file where the directive ends along with a
// boolean value indicating whether the directive ends because the end of file
// was reached or not.
static std::pair<unsigned, bool> findDirectiveEnd(SourceLocation HashLoc,
                                                  SourceManager &SM,
                                                  const LangOptions &LangOpts) {
  FileID FID = SM.getFileID(HashLoc);
  unsigned Offset = SM.getFileOffset(HashLoc);
  StringRef Content = SM.getBufferData(FID);
  Lexer Lex(SM.getLocForStartOfFile(FID), LangOpts, Content.begin(),
            Content.begin() + Offset, Content.end());
  Lex.SetCommentRetentionState(true);
  Token Tok;

  // This loop look for the newline after our directive but avoids the ones part
  // of a multi-line comments:
  //
  //     #include <foo> /* long \n comment */\n
  //                            ~~ no        ~~ yes
  for (;;) {
    // find the beginning of the end-of-line sequence
    StringRef::size_type EOLOffset = Content.find_first_of("\r\n", Offset);
    // ends because EOF was reached
    if (EOLOffset == StringRef::npos)
      return std::make_pair(Content.size(), true);

    // find the token that contains our end-of-line
    unsigned TokEnd = 0;
    do {
      Lex.LexFromRawLexer(Tok);
      TokEnd = SM.getFileOffset(Tok.getLocation()) + Tok.getLength();

      // happens when the whitespaces are eaten after a multiline comment
      if (Tok.is(tok::eof))
        return std::make_pair(EOLOffset, false);
    } while (TokEnd < EOLOffset);

    // the end-of-line is not part of a multi-line comment, return its location
    if (Tok.isNot(tok::comment))
      return std::make_pair(EOLOffset, false);

    // for the next search to start after the end of this token
    Offset = TokEnd;
  }
}

IncludeDirectives::IncludeDirectives(clang::CompilerInstance &CI)
    : CI(CI), Sources(CI.getSourceManager()) {
  // addPPCallbacks takes ownership of the callback
  CI.getPreprocessor().addPPCallbacks(new IncludeDirectivesPPCallback(this));
}

bool IncludeDirectives::lookForInclude(const FileEntry *File,
                                       const LocationVec &IncludeLocs,
                                       SeenFilesSet &Seen) const {
  // mark this file as visited
  Seen.insert(File);

  // First check if included directly in this file
  for (LocationVec::const_iterator I = IncludeLocs.begin(),
                                   E = IncludeLocs.end();
       I != E; ++I)
    if (Sources.getFileEntryForID(Sources.getFileID(*I)) == File)
      return true;

  // Otherwise look recursively all the included files
  FileToEntriesMap::const_iterator EntriesIt = FileToEntries.find(File);
  if (EntriesIt == FileToEntries.end())
    return false;
  for (EntryVec::const_iterator I = EntriesIt->second.begin(),
                                E = EntriesIt->second.end();
       I != E; ++I) {
    // skip if this header has already been checked before
    if (Seen.count(I->getIncludedFile()))
      continue;
    if (lookForInclude(I->getIncludedFile(), IncludeLocs, Seen))
      return true;
  }
  return false;
}

bool IncludeDirectives::hasInclude(const FileEntry *File,
                                   StringRef Include) const {
  llvm::StringMap<LocationVec>::const_iterator It =
      IncludeAsWrittenToLocationsMap.find(Include);

  // Include isn't included in any file
  if (It == IncludeAsWrittenToLocationsMap.end())
    return false;

  SeenFilesSet Seen;
  return lookForInclude(File, It->getValue(), Seen);
}

Replacement IncludeDirectives::addAngledInclude(const clang::FileEntry *File,
                                                llvm::StringRef Include) {
  FileID FID = Sources.translateFile(File);
  assert(!FID.isInvalid() && "Invalid file entry given!");

  if (hasInclude(File, Include))
    return Replacement();

  unsigned Offset, NLFlags;
  std::tie(Offset, NLFlags) = angledIncludeInsertionOffset(FID);

  StringRef EOL = guessEOL(Sources, FID);
  llvm::SmallString<32> InsertionText;
  if (NLFlags & NL_Prepend)
    InsertionText += EOL;
  if (NLFlags & NL_PrependAnother)
    InsertionText += EOL;
  InsertionText += "#include <";
  InsertionText += Include;
  InsertionText += ">";
  if (NLFlags & NL_AppendTwice) {
    InsertionText += EOL;
    InsertionText += EOL;
  }
  return Replacement(File->getName(), Offset, 0, InsertionText);
}

Replacement IncludeDirectives::addAngledInclude(llvm::StringRef File,
                                                llvm::StringRef Include) {
  const FileEntry *Entry = Sources.getFileManager().getFile(File);
  assert(Entry && "Invalid file given!");
  return addAngledInclude(Entry, Include);
}

std::pair<unsigned, unsigned>
IncludeDirectives::findFileHeaderEndOffset(FileID FID) const {
  unsigned NLFlags = NL_Prepend;
  StringRef Content = Sources.getBufferData(FID);
  Lexer Lex(Sources.getLocForStartOfFile(FID), CI.getLangOpts(),
            Content.begin(), Content.begin(), Content.end());
  Lex.SetCommentRetentionState(true);
  Lex.SetKeepWhitespaceMode(true);

  // find the first newline not part of a multi-line comment
  Token Tok;
  do {
    Lex.LexFromRawLexer(Tok);
    unsigned Offset = Sources.getFileOffset(Tok.getLocation());
    // allow one newline between the comments
    if (Tok.is(tok::unknown) && isWhitespace(Content[Offset])) {
      StringRef Whitespaces(Content.substr(Offset, Tok.getLength()));
      if (Whitespaces.count('\n') == 1 || Whitespaces.count('\r') == 1)
        Lex.LexFromRawLexer(Tok);
      else {
        // add an empty line to separate the file header and the inclusion
        NLFlags = NL_PrependTwice;
      }
    }
  } while (Tok.is(tok::comment));

  // apparently there is no header, insertion point is the beginning of the file
  if (Tok.isNot(tok::unknown))
    return std::make_pair(0, NL_AppendTwice);
  return std::make_pair(Sources.getFileOffset(Tok.getLocation()), NLFlags);
}

SourceLocation
IncludeDirectives::angledIncludeHintLoc(FileID FID) const {
  FileToEntriesMap::const_iterator EntriesIt =
      FileToEntries.find(Sources.getFileEntryForID(FID));

  if (EntriesIt == FileToEntries.end())
    return SourceLocation();

  HeaderSearch &HeaderInfo = CI.getPreprocessor().getHeaderSearchInfo();
  const EntryVec &Entries = EntriesIt->second;
  EntryVec::const_reverse_iterator QuotedCandidate = Entries.rend();
  for (EntryVec::const_reverse_iterator I = Entries.rbegin(),
                                        E = Entries.rend();
       I != E; ++I) {
    // Headers meant for multiple inclusion can potentially appears in the
    // middle of the code thus making them a poor choice for an insertion point.
    if (!HeaderInfo.isFileMultipleIncludeGuarded(I->getIncludedFile()))
      continue;

    // return preferably the last angled include
    if (I->isAngled())
      return I->getHashLocation();

    // keep track of the last quoted include that is guarded
    if (QuotedCandidate == Entries.rend())
      QuotedCandidate = I;
  }

  if (QuotedCandidate == Entries.rend())
    return SourceLocation();

  // return the last quoted-include if we couldn't find an angled one
  return QuotedCandidate->getHashLocation();
}

std::pair<unsigned, unsigned>
IncludeDirectives::angledIncludeInsertionOffset(FileID FID) const {
  SourceLocation Hint = angledIncludeHintLoc(FID);
  unsigned NL_Flags = NL_Prepend;

  // If we can't find a similar include and we are in a header check if it's a
  // guarded header. If so the hint will be the location of the #define from the
  // guard.
  if (Hint.isInvalid()) {
    const FileEntry *File = Sources.getFileEntryForID(FID);
    HeaderToGuardMap::const_iterator GuardIt = HeaderToGuard.find(File);
    if (GuardIt != HeaderToGuard.end()) {
      // get the hash location from the #define
      Hint = GuardIt->second;
      // we want a blank line between the #define and the #include
      NL_Flags = NL_PrependTwice;
    }
  }

  // no hints, insertion is done after the file header
  if (Hint.isInvalid())
    return findFileHeaderEndOffset(FID);

  unsigned Offset = findDirectiveEnd(Hint, Sources, CI.getLangOpts()).first;
  return std::make_pair(Offset, NL_Flags);
}
