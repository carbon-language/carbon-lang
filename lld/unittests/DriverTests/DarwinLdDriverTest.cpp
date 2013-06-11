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
  virtual MachOTargetInfo* doParse(int argc, const char **argv,
                                   raw_ostream &diag) {
    auto *info = new MachOTargetInfo();
    DarwinLdDriver::parse(argc, argv, *info, diag);
    return info;
  }
};

TEST_F(DarwinLdParserTest, Basic) {
  parse("ld", "foo.o", "bar.o", nullptr);
  EXPECT_FALSE(info->allowRemainingUndefines());
  EXPECT_FALSE(info->deadStrip());
  EXPECT_EQ(2, (int)inputFiles.size());
  EXPECT_EQ("foo.o", inputFiles[0]);
  EXPECT_EQ("bar.o", inputFiles[1]);
}

TEST_F(DarwinLdParserTest, Dylib) {
  parse("ld", "-dylib", "foo.o", nullptr);
  EXPECT_EQ(mach_o::MH_DYLIB, info->outputFileType());
}

TEST_F(DarwinLdParserTest, Relocatable) {
  parse("ld", "-r", "foo.o", nullptr);
  EXPECT_EQ(mach_o::MH_OBJECT, info->outputFileType());
}

TEST_F(DarwinLdParserTest, Bundle) {
  parse("ld", "-bundle", "foo.o", nullptr);
  EXPECT_EQ(mach_o::MH_BUNDLE, info->outputFileType());
}

TEST_F(DarwinLdParserTest, Preload) {
  parse("ld", "-preload", "foo.o", nullptr);
  EXPECT_EQ(mach_o::MH_PRELOAD, info->outputFileType());
}

TEST_F(DarwinLdParserTest, Static) {
  parse("ld", "-static", "foo.o", nullptr);
  EXPECT_EQ(mach_o::MH_EXECUTE, info->outputFileType());
}

TEST_F(DarwinLdParserTest, Entry) {
  parse("ld", "-e", "entryFunc", "foo.o", nullptr);
  EXPECT_EQ("entryFunc", info->entrySymbolName());
}

TEST_F(DarwinLdParserTest, OutputPath) {
  parse("ld", "-o", "foo", "foo.o", nullptr);
  EXPECT_EQ("foo", info->outputPath());
}

TEST_F(DarwinLdParserTest, DeadStrip) {
  parse("ld", "-dead_strip", "foo.o", nullptr);
  EXPECT_TRUE(info->deadStrip());
}

TEST_F(DarwinLdParserTest, Arch) {
  parse("ld", "-arch", "x86_64", "foo.o", nullptr);
  EXPECT_EQ(MachOTargetInfo::arch_x86_64, info->arch());
}

}  // end anonymous namespace
