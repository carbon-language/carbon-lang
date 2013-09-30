//===- clang-modernize/FileOverridesTest.cpp - File overrides unit tests --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Core/FileOverrides.h"
#include "Core/Refactoring.h"
#include "gtest/gtest.h"
#include "common/VirtualFileHelper.h"
#include "clang/Rewrite/Core/Rewriter.h"

using namespace clang;
using namespace clang::tooling;

static Replacement makeReplacement(unsigned Offset, unsigned Length,
                                   unsigned ReplacementLength,
                                   llvm::StringRef FilePath) {
  return Replacement(FilePath, Offset, Length,
                     std::string(ReplacementLength, '~'));
}

// generate a set of replacements containing one element
static ReplacementsVec makeReplacements(unsigned Offset, unsigned Length,
                                        unsigned ReplacementLength,
                                        llvm::StringRef FilePath = "~") {
  ReplacementsVec Replaces;
  Replaces.push_back(
      makeReplacement(Offset, Length, ReplacementLength, FilePath));
  return Replaces;
}

static bool equalRanges(Range A, Range B) {
  return A.getOffset() == B.getOffset() && A.getLength() == B.getLength();
}

TEST(ChangedRangesTest, adjustChangedRangesShrink) {
  ChangedRanges Changes;
  Changes.adjustChangedRanges(makeReplacements(0, 0, 4));
  EXPECT_NE(Changes.begin(), Changes.end());
  EXPECT_TRUE(equalRanges(Range(0, 4), *Changes.begin()));
  // create a replacement that cuts the end of the last insertion
  Changes.adjustChangedRanges(makeReplacements(2, 4, 0));
  Range ExpectedChanges[] = { Range(0, 2) };
  EXPECT_TRUE(
      std::equal(Changes.begin(), Changes.end(), ExpectedChanges, equalRanges));
}

TEST(ChangedRangesTest, adjustChangedRangesExtend) {
  ChangedRanges Changes;
  Changes.adjustChangedRanges(makeReplacements(1, 0, 4));
  // cut the old one by a bigger one
  Changes.adjustChangedRanges(makeReplacements(3, 4, 6));
  Range ExpectedChanges[] = { Range(1, 8) };
  EXPECT_TRUE(
      std::equal(Changes.begin(), Changes.end(), ExpectedChanges, equalRanges));
}

TEST(ChangedRangesTest, adjustChangedRangesNoOverlap) {
  ChangedRanges Changes;
  Changes.adjustChangedRanges(makeReplacements(0, 0, 4));
  Changes.adjustChangedRanges(makeReplacements(6, 0, 4));
  Range ExpectedChanges[] = { Range(0, 4), Range(6, 4) };
  EXPECT_TRUE(
      std::equal(Changes.begin(), Changes.end(), ExpectedChanges, equalRanges));
}

TEST(ChangedRangesTest, adjustChangedRangesNullRange) {
  ChangedRanges Changes;
  Changes.adjustChangedRanges(makeReplacements(0, 4, 0));
  Range ExpectedChanges[] = { Range(0, 0) };
  EXPECT_TRUE(
      std::equal(Changes.begin(), Changes.end(), ExpectedChanges, equalRanges));
}

TEST(ChangedRangesTest, adjustChangedRangesExtendExisting) {
  ChangedRanges Changes;
  Changes.adjustChangedRanges(makeReplacements(0, 0, 3));
  Changes.adjustChangedRanges(makeReplacements(2, 5, 8));
  Range ExpectedChanges[] = { Range(0, 10) };
  EXPECT_TRUE(
      std::equal(Changes.begin(), Changes.end(), ExpectedChanges, equalRanges));
}

TEST(ChangedRangesTest, adjustChangedRangesSplit) {
  ChangedRanges Changes;
  Changes.adjustChangedRanges(makeReplacements(0, 0, 3));
  Changes.adjustChangedRanges(makeReplacements(1, 1, 0));
  Range ExpectedChanges[] = { Range(0, 2) };
  EXPECT_TRUE(
      std::equal(Changes.begin(), Changes.end(), ExpectedChanges, equalRanges));
}

TEST(ChangedRangesTest, adjustChangedRangesRangeContained) {
  ChangedRanges Changes;
  Changes.adjustChangedRanges(makeReplacements(3, 0, 2));
  Changes.adjustChangedRanges(makeReplacements(1, 4, 5));
  Range ExpectedChanges[] = { Range(1, 5) };
  EXPECT_TRUE(
      std::equal(Changes.begin(), Changes.end(), ExpectedChanges, equalRanges));
}

TEST(ChangedRangesTest, adjustChangedRangesRangeResized) {
  ChangedRanges Changes;
  Changes.adjustChangedRanges(makeReplacements(2, 0, 5));
  // first make the range bigger
  Changes.adjustChangedRanges(makeReplacements(4, 1, 3));
  Range ExpectedChanges[] = { Range(2, 7) };
  EXPECT_TRUE(
      std::equal(Changes.begin(), Changes.end(), ExpectedChanges, equalRanges));
  // then smaller
  Changes.adjustChangedRanges(makeReplacements(3, 3, 1));
  ExpectedChanges[0] = Range(2, 5);
  EXPECT_TRUE(
      std::equal(Changes.begin(), Changes.end(), ExpectedChanges, equalRanges));
}

TEST(FileOverridesTest, applyOverrides) {

  // Set up initial state
  VirtualFileHelper VFHelper;

  SmallString<128> fileAPath("fileA.cpp");
  ASSERT_FALSE(llvm::sys::fs::make_absolute(fileAPath));
  SmallString<128> fileBPath("fileB.cpp");
  ASSERT_FALSE(llvm::sys::fs::make_absolute(fileBPath));
  VFHelper.mapFile(fileAPath, "Content A");
  VFHelper.mapFile(fileBPath, "Content B");
  SourceManager &SM = VFHelper.getNewSourceManager();

  // Fill a Rewriter with changes
  Rewriter Rewrites(SM, LangOptions());
  ReplacementsVec R(1, Replacement(fileAPath, 0, 7, "Stuff"));
  ASSERT_TRUE(applyAllReplacements(R, Rewrites));

  FileOverrides Overrides;
  Overrides.updateState(Rewrites);
  
  const FileOverrides::FileStateMap &State = Overrides.getState();
  
  // Ensure state updated
  ASSERT_TRUE(State.end() == State.find(fileBPath));
  ASSERT_TRUE(State.begin() == State.find(fileAPath));
  ASSERT_EQ("Stuff A", State.begin()->getValue());

  Overrides.applyOverrides(SM);

  const FileEntry *EntryA = SM.getFileManager().getFile(fileAPath);
  FileID IdA = SM.translateFile(EntryA);
  ASSERT_FALSE(IdA.isInvalid());

  // Ensure the contents of the buffer matches what we'd expect.
  const llvm::MemoryBuffer *BufferA = SM.getBuffer(IdA);
  ASSERT_FALSE(0 == BufferA);
  ASSERT_EQ("Stuff A", BufferA->getBuffer());
}

TEST(FileOverridesTest, adjustChangedRanges) {
  SmallString<128> fileAPath("fileA.cpp");
  ASSERT_FALSE(llvm::sys::fs::make_absolute(fileAPath));
  SmallString<128> fileBPath("fileB.cpp");
  ASSERT_FALSE(llvm::sys::fs::make_absolute(fileBPath));

  replace::FileToReplacementsMap GroupedReplacements;
  GroupedReplacements[fileAPath] = makeReplacements(0, 5, 4, fileAPath);
  GroupedReplacements[fileBPath] = makeReplacements(10, 0, 6, fileBPath);

  FileOverrides Overrides;

  const FileOverrides::ChangeMap &Map = Overrides.getChangedRanges();

  ASSERT_TRUE(Map.empty());

  Overrides.adjustChangedRanges(GroupedReplacements);

  ASSERT_TRUE(Map.end() != Map.find(fileAPath));
  ASSERT_TRUE(Map.end() != Map.find(fileBPath));
  const Range &RA = *Map.find(fileAPath)->second.begin();
  EXPECT_EQ(0u, RA.getOffset());
  EXPECT_EQ(4u, RA.getLength());
  const Range &RB = *Map.find(fileBPath)->second.begin();
  EXPECT_EQ(10u, RB.getOffset());
  EXPECT_EQ(6u, RB.getLength());
}
