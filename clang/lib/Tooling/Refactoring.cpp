//===--- Refactoring.cpp - Framework for clang refactoring tools ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Implements tools to support refactorings.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Lex/Lexer.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/Refactoring.h"
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

Replacement::Replacement(SourceManager &Sources, SourceLocation Start,
                         unsigned Length, StringRef ReplacementText) {
  setFromSourceLocation(Sources, Start, Length, ReplacementText);
}

Replacement::Replacement(SourceManager &Sources, const CharSourceRange &Range,
                         StringRef ReplacementText) {
  setFromSourceRange(Sources, Range, ReplacementText);
}

bool Replacement::isApplicable() const {
  return FilePath != InvalidLocation;
}

bool Replacement::apply(Rewriter &Rewrite) const {
  SourceManager &SM = Rewrite.getSourceMgr();
  const FileEntry *Entry = SM.getFileManager().getFile(FilePath);
  if (Entry == NULL)
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

bool Replacement::Less::operator()(const Replacement &R1,
                                   const Replacement &R2) const {
  if (R1.FilePath != R2.FilePath) return R1.FilePath < R2.FilePath;
  if (R1.ReplacementRange.getOffset() != R2.ReplacementRange.getOffset())
    return R1.ReplacementRange.getOffset() < R2.ReplacementRange.getOffset();
  if (R1.ReplacementRange.getLength() != R2.ReplacementRange.getLength())
    return R1.ReplacementRange.getLength() < R2.ReplacementRange.getLength();
  return R1.ReplacementText < R2.ReplacementText;
}

void Replacement::setFromSourceLocation(SourceManager &Sources,
                                        SourceLocation Start, unsigned Length,
                                        StringRef ReplacementText) {
  const std::pair<FileID, unsigned> DecomposedLocation =
      Sources.getDecomposedLoc(Start);
  const FileEntry *Entry = Sources.getFileEntryForID(DecomposedLocation.first);
  this->FilePath = Entry != NULL ? Entry->getName() : InvalidLocation;
  this->ReplacementRange = Range(DecomposedLocation.second, Length);
  this->ReplacementText = ReplacementText;
}

// FIXME: This should go into the Lexer, but we need to figure out how
// to handle ranges for refactoring in general first - there is no obvious
// good way how to integrate this into the Lexer yet.
static int getRangeSize(SourceManager &Sources, const CharSourceRange &Range) {
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

void Replacement::setFromSourceRange(SourceManager &Sources,
                                     const CharSourceRange &Range,
                                     StringRef ReplacementText) {
  setFromSourceLocation(Sources, Sources.getSpellingLoc(Range.getBegin()),
                        getRangeSize(Sources, Range), ReplacementText);
}

bool applyAllReplacements(Replacements &Replaces, Rewriter &Rewrite) {
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

std::string applyAllReplacements(StringRef Code, Replacements &Replaces) {
  FileManager Files((FileSystemOptions()));
  DiagnosticsEngine Diagnostics(
      IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs),
      new DiagnosticOptions);
  Diagnostics.setClient(new TextDiagnosticPrinter(
      llvm::outs(), &Diagnostics.getDiagnosticOptions()));
  SourceManager SourceMgr(Diagnostics, Files);
  Rewriter Rewrite(SourceMgr, LangOptions());
  llvm::MemoryBuffer *Buf = llvm::MemoryBuffer::getMemBuffer(Code, "<stdin>");
  const clang::FileEntry *Entry =
      Files.getVirtualFile("<stdin>", Buf->getBufferSize(), 0);
  SourceMgr.overrideFileContents(Entry, Buf);
  FileID ID =
      SourceMgr.createFileID(Entry, SourceLocation(), clang::SrcMgr::C_User);
  for (Replacements::iterator I = Replaces.begin(), E = Replaces.end(); I != E;
       ++I) {
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

RefactoringTool::RefactoringTool(const CompilationDatabase &Compilations,
                                 ArrayRef<std::string> SourcePaths)
  : ClangTool(Compilations, SourcePaths) {}

Replacements &RefactoringTool::getReplacements() { return Replace; }

int RefactoringTool::runAndSave(FrontendActionFactory *ActionFactory) {
  if (int Result = run(ActionFactory)) {
    return Result;
  }

  LangOptions DefaultLangOptions;
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  TextDiagnosticPrinter DiagnosticPrinter(llvm::errs(), &*DiagOpts);
  DiagnosticsEngine Diagnostics(
      IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()),
      &*DiagOpts, &DiagnosticPrinter, false);
  SourceManager Sources(Diagnostics, getFiles());
  Rewriter Rewrite(Sources, DefaultLangOptions);

  if (!applyAllReplacements(Rewrite)) {
    llvm::errs() << "Skipped some replacements.\n";
  }

  return saveRewrittenFiles(Rewrite);
}

bool RefactoringTool::applyAllReplacements(Rewriter &Rewrite) {
  return tooling::applyAllReplacements(Replace, Rewrite);
}

int RefactoringTool::saveRewrittenFiles(Rewriter &Rewrite) {
  for (Rewriter::buffer_iterator I = Rewrite.buffer_begin(),
                                 E = Rewrite.buffer_end();
       I != E; ++I) {
    // FIXME: This code is copied from the FixItRewriter.cpp - I think it should
    // go into directly into Rewriter (there we also have the Diagnostics to
    // handle the error cases better).
    const FileEntry *Entry =
        Rewrite.getSourceMgr().getFileEntryForID(I->first);
    std::string ErrorInfo;
    llvm::raw_fd_ostream FileStream(
        Entry->getName(), ErrorInfo, llvm::raw_fd_ostream::F_Binary);
    if (!ErrorInfo.empty())
      return 1;
    I->second.write(FileStream);
    FileStream.flush();
  }
  return 0;
}

} // end namespace tooling
} // end namespace clang
