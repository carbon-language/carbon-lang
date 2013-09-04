//===- clang-modernize/ReformattingTest.cpp - Reformatting unit tests -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Core/Reformatting.h"
#include "Core/FileOverrides.h"
#include "Core/Refactoring.h"
#include "gtest/gtest.h"
#include "VirtualFileHelper.h"

using namespace clang;
using namespace clang::tooling;

namespace {
// convenience function to create a ChangedRanges containing one Range
ChangedRanges makeChangedRanges(unsigned Offset, unsigned Length) {
  ChangedRanges Changes;
  ReplacementsVec Replaces;

  Replaces.push_back(Replacement("", Offset, 0, std::string(Length, '~')));
  Changes.adjustChangedRanges(Replaces);
  return Changes;
}
} // end anonymous namespace

TEST(Reformatter, SingleReformat) {
  VirtualFileHelper VFHelper;
  llvm::StringRef FileName = "<test>";
  VFHelper.mapFile(FileName, "int  a;\n"
                             "int  b;\n");

  Reformatter ChangesReformatter(format::getLLVMStyle());
  ChangedRanges Changes = makeChangedRanges(0, 6);
  tooling::ReplacementsVec Replaces;
  ChangesReformatter.reformatSingleFile(
      FileName, Changes, VFHelper.getNewSourceManager(), Replaces);

  // We expect the code above to reformatted like so:
  //
  // int a;
  // int  b;
  //
  // This test is slightly fragile since there's more than one Replacement that
  // results in the above change. However, testing the result of applying the
  // replacement is more trouble than it's worth in this context.
  ASSERT_EQ(1u, Replaces.size());
  EXPECT_EQ(3u, Replaces[0].getOffset());
  EXPECT_EQ(2u, Replaces[0].getLength());
  EXPECT_EQ(" ", Replaces[0].getReplacementText());
}
