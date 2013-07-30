//===-- Core/FileOverrides.cpp --------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides types and functionality for dealing with source
/// and header file content overrides.
///
//===----------------------------------------------------------------------===//

#include "Core/FileOverrides.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"
#include <algorithm>

using namespace clang;
using namespace clang::tooling;

void HeaderOverride::recordReplacements(
    llvm::StringRef TransformID, const clang::tooling::Replacements &Replaces) {
  TransformReplacements TR;
  TR.TransformID = TransformID;
  TR.GeneratedReplacements.resize(Replaces.size());
  std::copy(Replaces.begin(), Replaces.end(), TR.GeneratedReplacements.begin());
  TransformReplacementsDoc.Replacements.push_back(TR);
}

SourceOverrides::SourceOverrides(llvm::StringRef MainFileName,
                                 bool TrackChanges)
    : MainFileName(MainFileName), TrackChanges(TrackChanges) {}

void SourceOverrides::applyReplacements(tooling::Replacements &Replaces,
                                        llvm::StringRef TransformName) {
  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> DiagOpts(
      new DiagnosticOptions());
  DiagnosticsEngine Diagnostics(
      llvm::IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()),
      DiagOpts.getPtr());
  FileManager Files((FileSystemOptions()));
  SourceManager SM(Diagnostics, Files);
  applyReplacements(Replaces, SM, TransformName);
}

void SourceOverrides::applyReplacements(tooling::Replacements &Replaces,
                                        SourceManager &SM,
                                        llvm::StringRef TransformName) {
  applyOverrides(SM);

  Rewriter Rewrites(SM, LangOptions());

  // FIXME: applyAllReplacements will indicate if it couldn't apply all
  // replacements. Handle that case.
  bool Success = tooling::applyAllReplacements(Replaces, Rewrites);

  if (!Success)
    llvm::errs() << "error: failed to apply some replacements.";

  std::string ResultBuf;

  for (Rewriter::buffer_iterator I = Rewrites.buffer_begin(),
                                 E = Rewrites.buffer_end();
       I != E; ++I) {
    const FileEntry *Entry =
        Rewrites.getSourceMgr().getFileEntryForID(I->first);
    assert(Entry != NULL && "unexpected null FileEntry");
    assert(Entry->getName() != NULL &&
           "unexpected null return from FileEntry::getName()");
    llvm::StringRef FileName = Entry->getName();

    // Get a copy of the rewritten buffer from the Rewriter.
    ResultBuf.clear();
    llvm::raw_string_ostream StringStream(ResultBuf);
    I->second.write(StringStream);
    StringStream.flush();

    if (MainFileName == FileName) {
      MainFileOverride.swap(ResultBuf);
      continue;
    }

    // Header overrides are treated differently. Eventually, raw replacements
    // will be stored as well for later output to disk. Applying replacements
    // in memory will always be necessary as the source goes down the transform
    // pipeline.
    HeaderOverride &HeaderOv = Headers[FileName];
    // "Create" HeaderOverride if not already existing
    if (HeaderOv.getFileName().empty())
      HeaderOv = HeaderOverride(FileName);

    HeaderOv.swapContentOverride(ResultBuf);
  }

  // Separate replacements to header files
  Replacements MainFileReplaces;
  ReplacementsMap HeadersReplaces;
  for (Replacements::const_iterator I = Replaces.begin(), E = Replaces.end();
      I != E; ++I) {
    llvm::StringRef ReplacementFileName = I->getFilePath();

    if (ReplacementFileName == MainFileName) {
      MainFileReplaces.insert(*I);
      continue;
    }

    HeadersReplaces[ReplacementFileName].insert(*I);
  }

  // Record all replacements to headers.
  for (ReplacementsMap::const_iterator I = HeadersReplaces.begin(),
                                       E = HeadersReplaces.end();
       I != E; ++I) {
    HeaderOverride &HeaderOv = Headers[I->getKey()];
    HeaderOv.recordReplacements(TransformName, I->getValue());
  }

  if (TrackChanges)
    adjustChangedRanges(MainFileReplaces, HeadersReplaces);
}

void
SourceOverrides::adjustChangedRanges(const Replacements &MainFileReplaces,
                                     const ReplacementsMap &HeadersReplaces) {
  // Adjust the changed ranges for each individual file
  MainFileChanges.adjustChangedRanges(MainFileReplaces);
  for (ReplacementsMap::const_iterator I = HeadersReplaces.begin(),
                                       E = HeadersReplaces.end();
       I != E; ++I) {
    Headers[I->getKey()].adjustChangedRanges(I->getValue());
  }
}

void SourceOverrides::applyOverrides(SourceManager &SM) const {
  FileManager &FM = SM.getFileManager();

  if (isSourceOverriden())
    SM.overrideFileContents(FM.getFile(MainFileName),
                            llvm::MemoryBuffer::getMemBuffer(MainFileOverride));

  for (HeaderOverrides::const_iterator I = Headers.begin(), E = Headers.end();
       I != E; ++I) {
    assert(!I->second.getContentOverride().empty() &&
           "Header override should not be empty!");
    SM.overrideFileContents(
        FM.getFile(I->second.getFileName()),
        llvm::MemoryBuffer::getMemBuffer(I->second.getContentOverride()));
  }
}

bool generateReplacementsFileName(llvm::StringRef SourceFile,
                                    llvm::StringRef HeaderFile,
                                    llvm::SmallVectorImpl<char> &Result,
                                    llvm::SmallVectorImpl<char> &Error) {
  using namespace llvm::sys;
  std::string UniqueHeaderNameModel;

  // Get the filename portion of the path.
  llvm::StringRef SourceFileRef(path::filename(SourceFile));
  llvm::StringRef HeaderFileRef(path::filename(HeaderFile));

  // Get the actual path for the header file.
  llvm::SmallString<128> HeaderPath(HeaderFile);
  path::remove_filename(HeaderPath);

  // Build the model of the filename.
  llvm::raw_string_ostream UniqueHeaderNameStream(UniqueHeaderNameModel);
  UniqueHeaderNameStream << SourceFileRef << "_" << HeaderFileRef
                         << "_%%_%%_%%_%%_%%_%%" << ".yaml";
  path::append(HeaderPath, UniqueHeaderNameStream.str());

  Error.clear();
  if (llvm::error_code EC =
          fs::createUniqueFile(HeaderPath.c_str(), Result)) {
    Error.append(EC.message().begin(), EC.message().end());
    return false;
  }

  return true;
}

FileOverrides::~FileOverrides() {
  for (SourceOverridesMap::iterator I = Overrides.begin(), E = Overrides.end();
       I != E; ++I)
    delete I->getValue();
}

SourceOverrides &FileOverrides::getOrCreate(llvm::StringRef Filename) {
  SourceOverrides *&Override = Overrides[Filename];

  if (Override == NULL)
    Override = new SourceOverrides(Filename, TrackChanges);
  return *Override;
}

namespace {

/// \brief Comparator to be able to order tooling::Range based on their offset.
bool rangeLess(clang::tooling::Range A, clang::tooling::Range B) {
  if (A.getOffset() == B.getOffset())
    return A.getLength() < B.getLength();
  return A.getOffset() < B.getOffset();
}

/// \brief Functor that returns the given range without its overlaps with the
/// replacement given in the constructor.
struct RangeReplacedAdjuster {
  RangeReplacedAdjuster(const tooling::Replacement &Replace)
      : Replace(Replace.getOffset(), Replace.getLength()),
        ReplaceNewSize(Replace.getReplacementText().size()) {}

  tooling::Range operator()(clang::tooling::Range Range) const {
    if (!Range.overlapsWith(Replace))
      return Range;
    // range inside replacement -> make the range length null
    if (Replace.contains(Range))
      return tooling::Range(Range.getOffset(), 0);
    // replacement inside range -> resize the range
    if (Range.contains(Replace)) {
      int Difference = ReplaceNewSize - Replace.getLength();
      return tooling::Range(Range.getOffset(), Range.getLength() + Difference);
    }
    // beginning of the range replaced -> truncate range beginning
    if (Range.getOffset() > Replace.getOffset()) {
      unsigned ReplaceEnd = Replace.getOffset() + Replace.getLength();
      unsigned RangeEnd = Range.getOffset() + Range.getLength();
      return tooling::Range(ReplaceEnd, RangeEnd - ReplaceEnd);
    }
    // end of the range replaced -> truncate range end
    if (Range.getOffset() < Replace.getOffset())
      return tooling::Range(Range.getOffset(),
                            Replace.getOffset() - Range.getOffset());
    llvm_unreachable("conditions not handled properly");
  }

  const tooling::Range Replace;
  const unsigned ReplaceNewSize;
};

} // end anonymous namespace

void ChangedRanges::adjustChangedRanges(const tooling::Replacements &Replaces) {
  // first adjust existing ranges in case they overlap with the replacements
  for (Replacements::iterator I = Replaces.begin(), E = Replaces.end(); I != E;
       ++I) {
    const tooling::Replacement &Replace = *I;

    std::transform(Ranges.begin(), Ranges.end(), Ranges.begin(),
                   RangeReplacedAdjuster(Replace));
  }

  // then shift existing ranges to reflect the new positions
  for (RangeVec::iterator I = Ranges.begin(), E = Ranges.end(); I != E; ++I) {
    unsigned ShiftedOffset =
        tooling::shiftedCodePosition(Replaces, I->getOffset());
    *I = tooling::Range(ShiftedOffset, I->getLength());
  }

  // then generate the new ranges from the replacements
  for (Replacements::iterator I = Replaces.begin(), E = Replaces.end(); I != E;
       ++I) {
    const tooling::Replacement &R = *I;
    unsigned Offset = tooling::shiftedCodePosition(Replaces, R.getOffset());
    unsigned Length = R.getReplacementText().size();

    Ranges.push_back(tooling::Range(Offset, Length));
  }

  // cleanups unecessary ranges to finish
  coalesceRanges();
}

void ChangedRanges::coalesceRanges() {
  // sort the ranges by offset and then for each group of adjacent/overlapping
  // ranges the first one in the group is extended to cover the whole group.
  std::sort(Ranges.begin(), Ranges.end(), &rangeLess);
  RangeVec::iterator FirstInGroup = Ranges.begin();
  assert(!Ranges.empty() && "unexpected empty vector");
  for (RangeVec::iterator I = Ranges.begin() + 1, E = Ranges.end(); I != E;
       ++I) {
    unsigned GroupEnd = FirstInGroup->getOffset() + FirstInGroup->getLength();

    // no contact
    if (I->getOffset() > GroupEnd)
      FirstInGroup = I;
    else {
      unsigned GrpBegin = FirstInGroup->getOffset();
      unsigned GrpEnd = std::max(GroupEnd, I->getOffset() + I->getLength());
      *FirstInGroup = tooling::Range(GrpBegin, GrpEnd - GrpBegin);
    }
  }

  // remove the ranges that are covered by the first member of the group
  Ranges.erase(std::unique(Ranges.begin(), Ranges.end(),
                           std::mem_fun_ref(&Range::contains)),
               Ranges.end());
}
