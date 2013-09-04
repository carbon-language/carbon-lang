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
#include "clang/Tooling/ReplacementsYaml.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"
#include <algorithm>

using namespace clang;
using namespace clang::tooling;

bool generateReplacementsFileName(const llvm::StringRef MainSourceFile,
                                  llvm::SmallVectorImpl<char> &Result,
                                  llvm::SmallVectorImpl<char> &Error) {
  using namespace llvm::sys;

  Error.clear();
  if (llvm::error_code EC = fs::createUniqueFile(
          MainSourceFile + "_%%_%%_%%_%%_%%_%%.yaml", Result)) {
    Error.append(EC.message().begin(), EC.message().end());
    return false;
  }

  return true;
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

void
ChangedRanges::adjustChangedRanges(const tooling::ReplacementsVec &Replaces) {
  // first adjust existing ranges in case they overlap with the replacements
  for (ReplacementsVec::const_iterator I = Replaces.begin(), E = Replaces.end();
       I != E; ++I) {
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
  for (ReplacementsVec::const_iterator I = Replaces.begin(), E = Replaces.end();
       I != E; ++I) {
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

void FileOverrides::applyOverrides(clang::SourceManager &SM) const {
  FileManager &FM = SM.getFileManager();

  for (FileStateMap::const_iterator I = FileStates.begin(),
                                    E = FileStates.end();
       I != E; ++I) {
    SM.overrideFileContents(FM.getFile(I->getKey()),
                            llvm::MemoryBuffer::getMemBuffer(I->getValue()));
  }
}

void FileOverrides::adjustChangedRanges(
    const clang::replace::FileToReplacementsMap &Replaces) {

  for (replace::FileToReplacementsMap::const_iterator I = Replaces.begin(),
       E = Replaces.end(); I != E; ++I) {
    ChangeTracking[I->getKey()].adjustChangedRanges(I->getValue());
  }
}

void FileOverrides::updateState(const clang::Rewriter &Rewrites) {
  for (Rewriter::const_buffer_iterator BufferI = Rewrites.buffer_begin(),
                                       BufferE = Rewrites.buffer_end();
       BufferI != BufferE; ++BufferI) {
    const char *FileName =
        Rewrites.getSourceMgr().getFileEntryForID(BufferI->first)->getName();
    const RewriteBuffer &RewriteBuf = BufferI->second;
    FileStates[FileName].assign(RewriteBuf.begin(), RewriteBuf.end());
  }
}

bool FileOverrides::writeToDisk(DiagnosticsEngine &Diagnostics) const {
  bool Errors = false;
  for (FileStateMap::const_iterator I = FileStates.begin(),
                                    E = FileStates.end();
       I != E; ++I) {
    std::string ErrorInfo;
    // The extra transform through std::string is to ensure null-termination
    // of the filename stored as the key of the StringMap.
    llvm::raw_fd_ostream FileStream(I->getKey().str().c_str(), ErrorInfo);
    if (!ErrorInfo.empty()) {
      llvm::errs() << "Failed to write new state for " << I->getKey() << ".\n";
      Errors = true;
    }
    FileStream << I->getValue();
  }
  return !Errors;
}
