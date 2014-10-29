//===--- Replacement.cpp - Framework for clang refactoring tools ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Implements classes to support/store refactorings.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_os_ostream.h"

namespace clang {
namespace tooling {

static const char * const InvalidLocation = "";

Replacement::Replacement()
  : FilePath(InvalidLocation) {}

Replacement::Replacement(StringRef FilePath, unsigned Offset, unsigned Length,
                         StringRef ReplacementText)
    : FilePath(FilePath), ReplacementRange(Offset, Length),
      ReplacementText(ReplacementText) {}

Replacement::Replacement(const SourceManager &Sources, SourceLocation Start,
                         unsigned Length, StringRef ReplacementText) {
  setFromSourceLocation(Sources, Start, Length, ReplacementText);
}

Replacement::Replacement(const SourceManager &Sources,
                         const CharSourceRange &Range,
                         StringRef ReplacementText) {
  setFromSourceRange(Sources, Range, ReplacementText);
}

bool Replacement::isApplicable() const {
  return FilePath != InvalidLocation;
}

bool Replacement::apply(Rewriter &Rewrite) const {
  SourceManager &SM = Rewrite.getSourceMgr();
  const FileEntry *Entry = SM.getFileManager().getFile(FilePath);
  if (!Entry)
    return false;
  FileID ID;
  // FIXME: Use SM.translateFile directly.
  SourceLocation Location = SM.translateFileLineCol(Entry, 1, 1);
  ID = Location.isValid() ?
    SM.getFileID(Location) :
    SM.createFileID(Entry, SourceLocation(), SrcMgr::C_User);
  // FIXME: We cannot check whether Offset + Length is in the file, as
  // the remapping API is not public in the RewriteBuffer.
  const SourceLocation Start =
    SM.getLocForStartOfFile(ID).
    getLocWithOffset(ReplacementRange.getOffset());
  // ReplaceText returns false on success.
  // ReplaceText only fails if the source location is not a file location, in
  // which case we already returned false earlier.
  bool RewriteSucceeded = !Rewrite.ReplaceText(
      Start, ReplacementRange.getLength(), ReplacementText);
  assert(RewriteSucceeded);
  return RewriteSucceeded;
}

std::string Replacement::toString() const {
  std::string result;
  llvm::raw_string_ostream stream(result);
  stream << FilePath << ": " << ReplacementRange.getOffset() << ":+"
         << ReplacementRange.getLength() << ":\"" << ReplacementText << "\"";
  return result;
}

bool operator<(const Replacement &LHS, const Replacement &RHS) {
  if (LHS.getOffset() != RHS.getOffset())
    return LHS.getOffset() < RHS.getOffset();
  if (LHS.getLength() != RHS.getLength())
    return LHS.getLength() < RHS.getLength();
  if (LHS.getFilePath() != RHS.getFilePath())
    return LHS.getFilePath() < RHS.getFilePath();
  return LHS.getReplacementText() < RHS.getReplacementText();
}

bool operator==(const Replacement &LHS, const Replacement &RHS) {
  return LHS.getOffset() == RHS.getOffset() &&
         LHS.getLength() == RHS.getLength() &&
         LHS.getFilePath() == RHS.getFilePath() &&
         LHS.getReplacementText() == RHS.getReplacementText();
}

void Replacement::setFromSourceLocation(const SourceManager &Sources,
                                        SourceLocation Start, unsigned Length,
                                        StringRef ReplacementText) {
  const std::pair<FileID, unsigned> DecomposedLocation =
      Sources.getDecomposedLoc(Start);
  const FileEntry *Entry = Sources.getFileEntryForID(DecomposedLocation.first);
  if (Entry) {
    // Make FilePath absolute so replacements can be applied correctly when
    // relative paths for files are used.
    llvm::SmallString<256> FilePath(Entry->getName());
    std::error_code EC = llvm::sys::fs::make_absolute(FilePath);
    this->FilePath = EC ? FilePath.c_str() : Entry->getName();
  } else {
    this->FilePath = InvalidLocation;
  }
  this->ReplacementRange = Range(DecomposedLocation.second, Length);
  this->ReplacementText = ReplacementText;
}

// FIXME: This should go into the Lexer, but we need to figure out how
// to handle ranges for refactoring in general first - there is no obvious
// good way how to integrate this into the Lexer yet.
static int getRangeSize(const SourceManager &Sources,
                        const CharSourceRange &Range) {
  SourceLocation SpellingBegin = Sources.getSpellingLoc(Range.getBegin());
  SourceLocation SpellingEnd = Sources.getSpellingLoc(Range.getEnd());
  std::pair<FileID, unsigned> Start = Sources.getDecomposedLoc(SpellingBegin);
  std::pair<FileID, unsigned> End = Sources.getDecomposedLoc(SpellingEnd);
  if (Start.first != End.first) return -1;
  if (Range.isTokenRange())
    End.second += Lexer::MeasureTokenLength(SpellingEnd, Sources,
                                            LangOptions());
  return End.second - Start.second;
}

void Replacement::setFromSourceRange(const SourceManager &Sources,
                                     const CharSourceRange &Range,
                                     StringRef ReplacementText) {
  setFromSourceLocation(Sources, Sources.getSpellingLoc(Range.getBegin()),
                        getRangeSize(Sources, Range), ReplacementText);
}

unsigned shiftedCodePosition(const Replacements &Replaces, unsigned Position) {
  unsigned NewPosition = Position;
  for (Replacements::iterator I = Replaces.begin(), E = Replaces.end(); I != E;
       ++I) {
    if (I->getOffset() >= Position)
      break;
    if (I->getOffset() + I->getLength() > Position)
      NewPosition += I->getOffset() + I->getLength() - Position;
    NewPosition += I->getReplacementText().size() - I->getLength();
  }
  return NewPosition;
}

// FIXME: Remove this function when Replacements is implemented as std::vector
// instead of std::set.
unsigned shiftedCodePosition(const std::vector<Replacement> &Replaces,
                             unsigned Position) {
  unsigned NewPosition = Position;
  for (std::vector<Replacement>::const_iterator I = Replaces.begin(),
                                                E = Replaces.end();
       I != E; ++I) {
    if (I->getOffset() >= Position)
      break;
    if (I->getOffset() + I->getLength() > Position)
      NewPosition += I->getOffset() + I->getLength() - Position;
    NewPosition += I->getReplacementText().size() - I->getLength();
  }
  return NewPosition;
}

void deduplicate(std::vector<Replacement> &Replaces,
                 std::vector<Range> &Conflicts) {
  if (Replaces.empty())
    return;

  auto LessNoPath = [](const Replacement &LHS, const Replacement &RHS) {
    if (LHS.getOffset() != RHS.getOffset())
      return LHS.getOffset() < RHS.getOffset();
    if (LHS.getLength() != RHS.getLength())
      return LHS.getLength() < RHS.getLength();
    return LHS.getReplacementText() < RHS.getReplacementText();
  };

  auto EqualNoPath = [](const Replacement &LHS, const Replacement &RHS) {
    return LHS.getOffset() == RHS.getOffset() &&
           LHS.getLength() == RHS.getLength() &&
           LHS.getReplacementText() == RHS.getReplacementText();
  };

  // Deduplicate. We don't want to deduplicate based on the path as we assume
  // that all replacements refer to the same file (or are symlinks).
  std::sort(Replaces.begin(), Replaces.end(), LessNoPath);
  Replaces.erase(std::unique(Replaces.begin(), Replaces.end(), EqualNoPath),
                 Replaces.end());

  // Detect conflicts
  Range ConflictRange(Replaces.front().getOffset(),
                      Replaces.front().getLength());
  unsigned ConflictStart = 0;
  unsigned ConflictLength = 1;
  for (unsigned i = 1; i < Replaces.size(); ++i) {
    Range Current(Replaces[i].getOffset(), Replaces[i].getLength());
    if (ConflictRange.overlapsWith(Current)) {
      // Extend conflicted range
      ConflictRange = Range(ConflictRange.getOffset(),
                            std::max(ConflictRange.getLength(),
                                     Current.getOffset() + Current.getLength() -
                                         ConflictRange.getOffset()));
      ++ConflictLength;
    } else {
      if (ConflictLength > 1)
        Conflicts.push_back(Range(ConflictStart, ConflictLength));
      ConflictRange = Current;
      ConflictStart = i;
      ConflictLength = 1;
    }
  }

  if (ConflictLength > 1)
    Conflicts.push_back(Range(ConflictStart, ConflictLength));
}

bool applyAllReplacements(const Replacements &Replaces, Rewriter &Rewrite) {
  bool Result = true;
  for (Replacements::const_iterator I = Replaces.begin(),
                                    E = Replaces.end();
       I != E; ++I) {
    if (I->isApplicable()) {
      Result = I->apply(Rewrite) && Result;
    } else {
      Result = false;
    }
  }
  return Result;
}

// FIXME: Remove this function when Replacements is implemented as std::vector
// instead of std::set.
bool applyAllReplacements(const std::vector<Replacement> &Replaces,
                          Rewriter &Rewrite) {
  bool Result = true;
  for (std::vector<Replacement>::const_iterator I = Replaces.begin(),
                                                E = Replaces.end();
       I != E; ++I) {
    if (I->isApplicable()) {
      Result = I->apply(Rewrite) && Result;
    } else {
      Result = false;
    }
  }
  return Result;
}

std::string applyAllReplacements(StringRef Code, const Replacements &Replaces) {
  FileManager Files((FileSystemOptions()));
  DiagnosticsEngine Diagnostics(
      IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs),
      new DiagnosticOptions);
  SourceManager SourceMgr(Diagnostics, Files);
  Rewriter Rewrite(SourceMgr, LangOptions());
  std::unique_ptr<llvm::MemoryBuffer> Buf =
      llvm::MemoryBuffer::getMemBuffer(Code, "<stdin>");
  const clang::FileEntry *Entry =
      Files.getVirtualFile("<stdin>", Buf->getBufferSize(), 0);
  SourceMgr.overrideFileContents(Entry, std::move(Buf));
  FileID ID =
      SourceMgr.createFileID(Entry, SourceLocation(), clang::SrcMgr::C_User);
  for (Replacements::const_iterator I = Replaces.begin(), E = Replaces.end();
       I != E; ++I) {
    Replacement Replace("<stdin>", I->getOffset(), I->getLength(),
                        I->getReplacementText());
    if (!Replace.apply(Rewrite))
      return "";
  }
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  Rewrite.getEditBuffer(ID).write(OS);
  OS.flush();
  return Result;
}

} // end namespace tooling
} // end namespace clang

