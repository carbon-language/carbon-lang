//===- cpp11-migrate/FileOverridesTest.cpp - File overrides unit tests ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Core/FileOverrides.h"
#include "gtest/gtest.h"
#include "VirtualFileHelper.h"

using namespace clang;
using namespace clang::tooling;

TEST(SourceOverridesTest, Interface) {
  llvm::StringRef FileName = "<test-file>";
  VirtualFileHelper VFHelper;
  VFHelper.mapFile(
      FileName,
      "std::vector<such_a_long_name_for_a_type>::const_iterator long_type =\n"
      "    vec.begin();\n");
  SourceOverrides Overrides(FileName, /*TrackFileChanges=*/false);

  EXPECT_EQ(FileName, Overrides.getMainFileName());
  EXPECT_FALSE(Overrides.isSourceOverriden());
  EXPECT_FALSE(Overrides.isTrackingFileChanges());

  Replacements Replaces;
  unsigned ReplacementLength =
      strlen("std::vector<such_a_long_name_for_a_type>::const_iterator");
  Replaces.insert(
      Replacement(FileName, 0, ReplacementLength, "auto"));
  Overrides.applyReplacements(Replaces, VFHelper.getNewSourceManager(),
                              "use-auto");
  EXPECT_TRUE(Overrides.isSourceOverriden());

  std::string ExpectedContent = "auto long_type =\n"
                                "    vec.begin();\n";
  EXPECT_EQ(ExpectedContent, Overrides.getMainFileContent());
}

namespace {
Replacement makeReplacement(unsigned Offset, unsigned Length,
                            unsigned ReplacementLength) {
  return Replacement("", Offset, Length, std::string(ReplacementLength, '~'));
}

// generate a set of replacements containing one element
Replacements makeReplacements(unsigned Offset, unsigned Length,
                              unsigned ReplacementLength) {
  Replacements Replaces;
  Replaces.insert(makeReplacement(Offset, Length, ReplacementLength));
  return Replaces;
}

bool equalRanges(Range A, Range B) {
  return A.getOffset() == B.getOffset() && A.getLength() == B.getLength();
}
} // end anonymous namespace

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
