//===- lld/unittest/DarwinLdDriverTest.cpp --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Darwin's ld driver tests.
///
//===----------------------------------------------------------------------===//

#include "DriverTest.h"

#include "lld/ReaderWriter/MachOTargetInfo.h"
#include "../../lib/ReaderWriter/MachO/MachOFormat.hpp"

using namespace llvm;
using namespace lld;

namespace {

class DarwinLdParserTest : public ParserTest<DarwinLdDriver, MachOTargetInfo> {
protected:
  virtual const TargetInfo *targetInfo() {
    return &_info;
  }
};

TEST_F(DarwinLdParserTest, Basic) {
  EXPECT_FALSE(parse("ld", "foo.o", "bar.o", nullptr));
  EXPECT_FALSE(_info.allowRemainingUndefines());
  EXPECT_FALSE(_info.deadStrip());
  EXPECT_EQ(2, inputFileCount());
  EXPECT_EQ("foo.o", inputFile(0));
  EXPECT_EQ("bar.o", inputFile(1));
}

TEST_F(DarwinLdParserTest, Dylib) {
  EXPECT_FALSE(parse("ld", "-dylib", "foo.o", nullptr));
  EXPECT_EQ(mach_o::MH_DYLIB, _info.outputFileType());
}

TEST_F(DarwinLdParserTest, Relocatable) {
  EXPECT_FALSE(parse("ld", "-r", "foo.o", nullptr));
  EXPECT_EQ(mach_o::MH_OBJECT, _info.outputFileType());
}

TEST_F(DarwinLdParserTest, Bundle) {
  EXPECT_FALSE(parse("ld", "-bundle", "foo.o", nullptr));
  EXPECT_EQ(mach_o::MH_BUNDLE, _info.outputFileType());
}

TEST_F(DarwinLdParserTest, Preload) {
  EXPECT_FALSE(parse("ld", "-preload", "foo.o", nullptr));
  EXPECT_EQ(mach_o::MH_PRELOAD, _info.outputFileType());
}

TEST_F(DarwinLdParserTest, Static) {
  EXPECT_FALSE(parse("ld", "-static", "foo.o", nullptr));
  EXPECT_EQ(mach_o::MH_EXECUTE, _info.outputFileType());
}

TEST_F(DarwinLdParserTest, Entry) {
  EXPECT_FALSE(parse("ld", "-e", "entryFunc", "foo.o", nullptr));
  EXPECT_EQ("entryFunc", _info.entrySymbolName());
}

TEST_F(DarwinLdParserTest, OutputPath) {
  EXPECT_FALSE(parse("ld", "-o", "foo", "foo.o", nullptr));
  EXPECT_EQ("foo", _info.outputPath());
}

TEST_F(DarwinLdParserTest, DeadStrip) {
  EXPECT_FALSE(parse("ld", "-dead_strip", "foo.o", nullptr));
  EXPECT_TRUE(_info.deadStrip());
}

TEST_F(DarwinLdParserTest, Arch) {
  EXPECT_FALSE(parse("ld", "-arch", "x86_64", "foo.o", nullptr));
  EXPECT_EQ(MachOTargetInfo::arch_x86_64, _info.arch());
}

}  // end anonymous namespace
