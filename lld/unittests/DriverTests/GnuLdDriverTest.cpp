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

class GnuLdParserTest : public ParserTest<GnuLdDriver, ELFTargetInfo> {
protected:
  virtual ELFTargetInfo* doParse(int argc, const char **argv,
                                 raw_ostream &diag) {
    std::unique_ptr<ELFTargetInfo> info;
    GnuLdDriver::parse(argc, argv, info, diag);
    return info.release();
  }
};

TEST_F(GnuLdParserTest, Basic) {
  parse("ld", "infile.o", nullptr);
  ASSERT_TRUE(!!info);
  EXPECT_EQ("a.out", info->outputPath());
  EXPECT_EQ(1, (int)inputFiles.size());
  EXPECT_EQ("infile.o", inputFiles[0]);
  EXPECT_FALSE(info->outputYAML());
}

TEST_F(GnuLdParserTest, ManyOptions) {
  parse("ld", "-entry", "_start", "-o", "outfile",
        "-emit-yaml", "infile.o", nullptr);
  ASSERT_TRUE(!!info);
  EXPECT_EQ("outfile", info->outputPath());
  EXPECT_EQ("_start", info->entrySymbolName());
  EXPECT_TRUE(info->outputYAML());
}

}  // end anonymous namespace
