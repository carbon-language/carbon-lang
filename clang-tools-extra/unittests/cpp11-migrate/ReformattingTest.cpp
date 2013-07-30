//===- cpp11-migrate/ReformattingTest.cpp - Reformatting unit tests -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Core/Reformatting.h"
#include "Core/FileOverrides.h"
#include "gtest/gtest.h"
#include "VirtualFileHelper.h"

using namespace clang;
using namespace clang::tooling;

namespace {
// convenience function to create a ChangedRanges containing one Range
ChangedRanges makeChangedRanges(unsigned Offset, unsigned Length) {
  ChangedRanges Changes;
  Replacements Replaces;

  Replaces.insert(Replacement("", Offset, 0, std::string(Length, '~')));
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
  tooling::Replacements Replaces = ChangesReformatter.reformatSingleFile(
      FileName, Changes, VFHelper.getNewSourceManager());

  SourceOverrides Overrides(FileName, /*TrackChanges=*/false);
  Overrides.applyReplacements(Replaces, VFHelper.getNewSourceManager());

  std::string Expected, Result;

  Expected = "int a;\n"
             "int  b;\n";
  Result = Overrides.getMainFileContent();
  EXPECT_EQ(Expected, Result);
}
