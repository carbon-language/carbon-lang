//===- lld/unittest/GnuLdDriverTest.cpp -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief GNU ld driver tests.
///
//===----------------------------------------------------------------------===//

#include "DriverTest.h"

#include "lld/ReaderWriter/ELFTargetInfo.h"

using namespace llvm;
using namespace lld;

namespace {

class GnuLdParserTest : public ParserTest<
                                GnuLdDriver, std::unique_ptr<ELFTargetInfo> > {
protected:
  virtual const TargetInfo *targetInfo() {
    return _info.get();
  }
};

TEST_F(GnuLdParserTest, Empty) {
  EXPECT_TRUE(parse("ld", nullptr));
  EXPECT_EQ(targetInfo(), nullptr);
  EXPECT_EQ("No input files\n", errorMessage());
}

TEST_F(GnuLdParserTest, Basic) {
  EXPECT_FALSE(parse("ld", "infile.o", nullptr));
  EXPECT_NE(targetInfo(), nullptr);
  EXPECT_EQ("a.out", targetInfo()->outputPath());
  EXPECT_EQ(1, inputFileCount());
  EXPECT_EQ("infile.o", inputFile(0));
  EXPECT_FALSE(_info->outputYAML());
}

TEST_F(GnuLdParserTest, ManyOptions) {
  EXPECT_FALSE(parse("ld", "-entry", "_start", "-o", "outfile",
        "-emit-yaml", "infile.o", nullptr));
  EXPECT_NE(targetInfo(), nullptr);
  EXPECT_EQ("outfile", targetInfo()->outputPath());
  EXPECT_EQ("_start", targetInfo()->entrySymbolName());
  EXPECT_TRUE(_info->outputYAML());
}

}  // end anonymous namespace
