//===- unittests/cpp11-migrate/ReplacementsYamlTest.cpp -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Tests that change description files can be written and read.
//
//===----------------------------------------------------------------------===//

#include "Utility.h"
#include "Core/FileOverrides.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(ReplacementsYamlTest, writeReadTest) {
  using clang::tooling::Replacement;

  const std::string TargetFile = "/path/to/common.h";
  const std::string MainSourceFile = "/path/to/source.cpp";
  const unsigned int ReplacementOffset1 = 232;
  const unsigned int ReplacementLength1 = 56;
  const std::string ReplacementText1 = "(auto & elem : V)";
  const unsigned int ReplacementOffset2 = 301;
  const unsigned int ReplacementLength2 = 2;
  const std::string ReplacementText2 = "elem";

  MigratorDocument Doc;
  Doc.Replacements.push_back(Replacement(TargetFile, ReplacementOffset1,
                                        ReplacementLength1, ReplacementText1));
  Doc.Replacements.push_back(Replacement(TargetFile, ReplacementOffset2,
                                        ReplacementLength2, ReplacementText2));

  Doc.TargetFile = TargetFile.c_str();
  Doc.MainSourceFile= MainSourceFile.c_str();

  std::string YamlContent;
  llvm::raw_string_ostream YamlContentStream(YamlContent);

  // Write to the YAML file.
  {
    yaml::Output YAML(YamlContentStream);
    YAML << Doc;
    YamlContentStream.str();
    ASSERT_NE(YamlContent.length(), 0u);
  }

  // Read from the YAML file and verify that what was written is exactly what
  // we read back.
  {
    MigratorDocument DocActual;
    yaml::Input YAML(YamlContent);
    YAML >> DocActual;
    ASSERT_NO_ERROR(YAML.error());
    EXPECT_EQ(TargetFile, DocActual.TargetFile);
    EXPECT_EQ(MainSourceFile, DocActual.MainSourceFile);
    ASSERT_EQ(2u, DocActual.Replacements.size());

    EXPECT_EQ(ReplacementOffset1, DocActual.Replacements[0].getOffset());
    EXPECT_EQ(ReplacementLength1, DocActual.Replacements[0].getLength());
    EXPECT_EQ(ReplacementText1,
              DocActual.Replacements[0].getReplacementText().str());

    EXPECT_EQ(ReplacementOffset2, DocActual.Replacements[1].getOffset());
    EXPECT_EQ(ReplacementLength2, DocActual.Replacements[1].getLength());
    EXPECT_EQ(ReplacementText2,
              DocActual.Replacements[1].getReplacementText().str());
  }
}
