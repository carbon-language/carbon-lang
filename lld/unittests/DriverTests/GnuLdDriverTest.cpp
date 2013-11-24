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

}  // end anonymous namespace
