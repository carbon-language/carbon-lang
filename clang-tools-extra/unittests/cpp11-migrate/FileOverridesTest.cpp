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
  SourceOverrides Overrides(FileName);

  EXPECT_EQ(FileName, Overrides.getMainFileName());
  EXPECT_FALSE(Overrides.isSourceOverriden());

  Replacements Replaces;
  unsigned ReplacementLength =
      strlen("std::vector<such_a_long_name_for_a_type>::const_iterator");
  Replaces.insert(Replacement(FileName, 0, ReplacementLength, "auto"));
  Overrides.applyReplacements(Replaces, VFHelper.getNewSourceManager());
  EXPECT_TRUE(Overrides.isSourceOverriden());

  std::string ExpectedContent = "auto long_type =\n"
                                "    vec.begin();\n";
  EXPECT_EQ(ExpectedContent, Overrides.getMainFileContent());
}
