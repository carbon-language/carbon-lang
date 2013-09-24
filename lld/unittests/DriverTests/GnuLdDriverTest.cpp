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

#include "lld/ReaderWriter/ELFLinkingContext.h"

using namespace llvm;
using namespace lld;

namespace {

class GnuLdParserTest
    : public ParserTest<GnuLdDriver, std::unique_ptr<ELFLinkingContext>> {
protected:
  virtual const LinkingContext *linkingContext() { return _context.get(); }
};

TEST_F(GnuLdParserTest, Empty) {
  EXPECT_FALSE(parse("ld", nullptr));
  EXPECT_EQ(linkingContext(), nullptr);
  EXPECT_EQ("No input files\n", errorMessage());
}

TEST_F(GnuLdParserTest, Basic) {
  EXPECT_TRUE(parse("ld", "infile.o", nullptr));
  EXPECT_NE(linkingContext(), nullptr);
  EXPECT_EQ("a.out", linkingContext()->outputPath());
  EXPECT_EQ(1, inputFileCount());
  EXPECT_EQ("infile.o", inputFile(0));
  EXPECT_FALSE(_context->outputFileType() ==
               LinkingContext::OutputFileType::YAML);
}

TEST_F(GnuLdParserTest, ManyOptions) {
  EXPECT_TRUE(parse("ld", "-entry", "_start", "-o", "outfile",
                    "--output-filetype=yaml", "infile.o", nullptr));
  EXPECT_NE(linkingContext(), nullptr);
  EXPECT_EQ("outfile", linkingContext()->outputPath());
  EXPECT_EQ("_start", linkingContext()->entrySymbolName());
  EXPECT_TRUE(_context->outputFileType() ==
              LinkingContext::OutputFileType::YAML);
}

}  // end anonymous namespace
