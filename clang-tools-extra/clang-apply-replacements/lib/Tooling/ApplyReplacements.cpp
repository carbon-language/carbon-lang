//===-- ApplyReplacements.cpp - Apply and deduplicate replacements --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides the implementation for deduplicating, detecting
/// conflicts in, and applying collections of Replacements.
///
/// FIXME: Use Diagnostics for output instead of llvm::errs().
///
//===----------------------------------------------------------------------===//
#include "clang-apply-replacements/Tooling/ApplyReplacements.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Format/Format.h"
#include "clang/Lex/Lexer.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/ReplacementsYaml.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace clang;


static void eatDiagnostics(const SMDiagnostic &, void *) {}

namespace clang {
namespace replace {

llvm::error_code
collectReplacementsFromDirectory(const llvm::StringRef Directory,
                                 TUReplacements &TUs,
                                 TUReplacementFiles & TURFiles,
                                 clang::DiagnosticsEngine &Diagnostics) {
  using namespace llvm::sys::fs;
  using namespace llvm::sys::path;

  error_code ErrorCode;

  for (recursive_directory_iterator I(Directory, ErrorCode), E;
       I != E && !ErrorCode; I.increment(ErrorCode)) {
    if (filename(I->path())[0] == '.') {
      // Indicate not to descend into directories beginning with '.'
      I.no_push();
      continue;
    }

    if (extension(I->path()) != ".yaml")
      continue;

    TURFiles.push_back(I->path());

    OwningPtr<MemoryBuffer> Out;
    error_code BufferError = MemoryBuffer::getFile(I->path(), Out);
    if (BufferError) {
      errs() << "Error reading " << I->path() << ": " << BufferError.message()
             << "\n";
      continue;
    }

    yaml::Input YIn(Out->getBuffer(), NULL, &eatDiagnostics);
    tooling::TranslationUnitReplacements TU;
    YIn >> TU;
    if (YIn.error()) {
      // File doesn't appear to be a header change description. Ignore it.
      continue;
    }

    // Only keep files that properly parse.
    TUs.push_back(TU);
  }

  return ErrorCode;
}

/// \brief Dumps information for a sequence of conflicting Replacements.
///
/// \param[in] File FileEntry for the file the conflicting Replacements are
/// for.
/// \param[in] ConflictingReplacements List of conflicting Replacements.
/// \param[in] SM SourceManager used for reporting.
static void reportConflict(
    const FileEntry *File,
    const llvm::ArrayRef<clang::tooling::Replacement> ConflictingReplacements,
    SourceManager &SM) {
  FileID FID = SM.translateFile(File);
  if (FID.isInvalid())
    FID = SM.createFileID(File, SourceLocation(), SrcMgr::C_User);

  // FIXME: Output something a little more user-friendly (e.g. unified diff?)
  errs() << "The following changes conflict:\n";
  for (const tooling::Replacement *I = ConflictingReplacements.begin(),
                                  *E = ConflictingReplacements.end();
       I != E; ++I) {
    if (I->getLength() == 0) {
      errs() << "  Insert at " << SM.getLineNumber(FID, I->getOffset()) << ":"
             << SM.getColumnNumber(FID, I->getOffset()) << " "
             << I->getReplacementText() << "\n";
    } else {
      if (I->getReplacementText().empty())
        errs() << "  Remove ";
      else
        errs() << "  Replace ";

      errs() << SM.getLineNumber(FID, I->getOffset()) << ":"
             << SM.getColumnNumber(FID, I->getOffset()) << "-"
             << SM.getLineNumber(FID, I->getOffset() + I->getLength() - 1)
             << ":"
             << SM.getColumnNumber(FID, I->getOffset() + I->getLength() - 1);

      if (I->getReplacementText().empty())
        errs() << "\n";
      else
        errs() << " with \"" << I->getReplacementText() << "\"\n";
    }
  }
}

/// \brief Deduplicates and tests for conflicts among the replacements for each
/// file in \c Replacements. Any conflicts found are reported.
///
/// \post Replacements[i].getOffset() <= Replacements[i+1].getOffset().
///
/// \param[in,out] Replacements Container of all replacements grouped by file
/// to be deduplicated and checked for conflicts.
/// \param[in] SM SourceManager required for conflict reporting.
///
/// \returns \li true if conflicts were detected
///          \li false if no conflicts were detected
static bool deduplicateAndDetectConflicts(FileToReplacementsMap &Replacements,
                                          SourceManager &SM) {
  bool conflictsFound = false;

  for (FileToReplacementsMap::iterator I = Replacements.begin(),
                                       E = Replacements.end();
       I != E; ++I) {

    const FileEntry *Entry = SM.getFileManager().getFile(I->getKey());
    if (!Entry) {
      errs() << "Described file '" << I->getKey()
             << "' doesn't exist. Ignoring...\n";
      continue;
    }

    std::vector<tooling::Range> Conflicts;
    tooling::deduplicate(I->getValue(), Conflicts);

    if (Conflicts.empty())
      continue;

    conflictsFound = true;

    errs() << "There are conflicting changes to " << I->getKey() << ":\n";

    for (std::vector<tooling::Range>::const_iterator
             ConflictI = Conflicts.begin(),
             ConflictE = Conflicts.end();
         ConflictI != ConflictE; ++ConflictI) {
      ArrayRef<tooling::Replacement> ConflictingReplacements(
          &I->getValue()[ConflictI->getOffset()], ConflictI->getLength());
      reportConflict(Entry, ConflictingReplacements, SM);
    }
  }

  return conflictsFound;
}

bool mergeAndDeduplicate(const TUReplacements &TUs,
                         FileToReplacementsMap &GroupedReplacements,
                         clang::SourceManager &SM) {

  // Group all replacements by target file.
  for (TUReplacements::const_iterator TUI = TUs.begin(), TUE = TUs.end();
       TUI != TUE; ++TUI)
    for (std::vector<tooling::Replacement>::const_iterator
             RI = TUI->Replacements.begin(),
             RE = TUI->Replacements.end();
         RI != RE; ++RI)
      GroupedReplacements[RI->getFilePath()].push_back(*RI);


  // Ask clang to deduplicate and report conflicts.
  if (deduplicateAndDetectConflicts(GroupedReplacements, SM))
    return false;

  return true;
}

bool applyReplacements(const FileToReplacementsMap &GroupedReplacements,
                       clang::Rewriter &Rewrites) {

  // Apply all changes
  //
  // FIXME: No longer certain GroupedReplacements is really the best kind of
  // data structure for applying replacements. Rewriter certainly doesn't care.
  // However, until we nail down the design of ReplacementGroups, might as well
  // leave this as is.
  for (FileToReplacementsMap::const_iterator I = GroupedReplacements.begin(),
                                             E = GroupedReplacements.end();
       I != E; ++I) {
    if (!tooling::applyAllReplacements(I->getValue(), Rewrites))
      return false;
  }

  return true;
}

RangeVector calculateChangedRanges(
    const std::vector<clang::tooling::Replacement> &Replaces) {
  RangeVector ChangedRanges;

  // Generate the new ranges from the replacements.
  //
  // NOTE: This is O(n^2) in the number of replacements. If this starts to
  // become a problem inline shiftedCodePosition() here and do shifts in a
  // single run through this loop.
  for (std::vector<clang::tooling::Replacement>::const_iterator
           I = Replaces.begin(),
           E = Replaces.end();
       I != E; ++I) {
    const tooling::Replacement &R = *I;
    unsigned Offset = tooling::shiftedCodePosition(Replaces, R.getOffset());
    unsigned Length = R.getReplacementText().size();

    ChangedRanges.push_back(tooling::Range(Offset, Length));
  }

  return ChangedRanges;
}

bool writeFiles(const clang::Rewriter &Rewrites) {

  for (Rewriter::const_buffer_iterator BufferI = Rewrites.buffer_begin(),
                                       BufferE = Rewrites.buffer_end();
       BufferI != BufferE; ++BufferI) {
    const char *FileName =
        Rewrites.getSourceMgr().getFileEntryForID(BufferI->first)->getName();

    std::string ErrorInfo;

    llvm::raw_fd_ostream FileStream(FileName, ErrorInfo, llvm::sys::fs::F_Text);
    if (!ErrorInfo.empty()) {
      errs() << "Warning: Could not write to " << FileName << "\n";
      continue;
    }
    BufferI->second.write(FileStream);
  }

  return true;
}

bool deleteReplacementFiles(const TUReplacementFiles &Files,
                            clang::DiagnosticsEngine &Diagnostics) {
  bool Success = true;
  for (TUReplacementFiles::const_iterator I = Files.begin(), E = Files.end();
       I != E; ++I) {
    error_code Error = llvm::sys::fs::remove(*I);
    if (Error) {
      Success = false;
      // FIXME: Use Diagnostics for outputting errors.
      errs() << "Error deleting file: " << *I << "\n";
      errs() << Error.message() << "\n";
      errs() << "Please delete the file manually\n";
    }
  }
  return Success;
}

} // end namespace replace
} // end namespace clang
