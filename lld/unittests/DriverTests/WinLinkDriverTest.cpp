//===- lld/unittest/WinLinkDriverTest.cpp ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Windows link.exe driver tests.
///
//===----------------------------------------------------------------------===//

#include "DriverTest.h"

#include "lld/ReaderWriter/PECOFFTargetInfo.h"
#include "llvm/Support/COFF.h"

using namespace llvm;
using namespace lld;

namespace {

class WinLinkParserTest : public ParserTest<WinLinkDriver, PECOFFTargetInfo> {
protected:
  virtual const TargetInfo *targetInfo() {
    return &_info;
  }
};

TEST_F(WinLinkParserTest, Basic) {
  EXPECT_FALSE(parse("link.exe", "-subsystem", "console", "-out", "a.exe",
        "-entry", "_start", "a.obj", "b.obj", "c.obj", nullptr));
  EXPECT_EQ(llvm::COFF::IMAGE_SUBSYSTEM_WINDOWS_CUI, _info.getSubsystem());
  EXPECT_EQ("a.exe", _info.outputPath());
  EXPECT_EQ("_start", _info.entrySymbolName());
  EXPECT_EQ(3, inputFileCount());
  EXPECT_EQ("a.obj", inputFile(0));
  EXPECT_EQ("b.obj", inputFile(1));
  EXPECT_EQ("c.obj", inputFile(2));
  EXPECT_EQ(6, _info.getMinOSVersion().majorVersion);
  EXPECT_EQ(0, _info.getMinOSVersion().minorVersion);
  EXPECT_EQ(1024 * 1024ULL, _info.getStackReserve());
  EXPECT_EQ(4096ULL, _info.getStackCommit());
  EXPECT_FALSE(_info.allowRemainingUndefines());
  EXPECT_TRUE(_info.getNxCompat());
  EXPECT_FALSE(_info.getLargeAddressAware());
}

TEST_F(WinLinkParserTest, WindowsStyleOption) {
  EXPECT_FALSE(parse("link.exe", "/subsystem:console", "/out:a.exe", "a.obj", 
                  nullptr));
  EXPECT_EQ(llvm::COFF::IMAGE_SUBSYSTEM_WINDOWS_CUI, _info.getSubsystem());
  EXPECT_EQ("a.exe", _info.outputPath());
  EXPECT_EQ(1, inputFileCount());
  EXPECT_EQ("a.obj", inputFile(0));
}

TEST_F(WinLinkParserTest, NoFileExtension) {
  EXPECT_FALSE(parse("link.exe", "foo", "bar", nullptr));
  EXPECT_EQ("foo.exe", _info.outputPath());
  EXPECT_EQ(2, inputFileCount());
  EXPECT_EQ("foo.obj", inputFile(0));
  EXPECT_EQ("bar.obj", inputFile(1));
}

TEST_F(WinLinkParserTest, NonStandardFileExtension) {
  EXPECT_FALSE(parse("link.exe", "foo.o", nullptr));
  EXPECT_EQ("foo.exe", _info.outputPath());
  EXPECT_EQ(1, inputFileCount());
  EXPECT_EQ("foo.o", inputFile(0));
}

TEST_F(WinLinkParserTest, MinMajorOSVersion) {
  EXPECT_FALSE(parse("link.exe", "-subsystem", "windows,3", "foo.o", nullptr));
  EXPECT_EQ(llvm::COFF::IMAGE_SUBSYSTEM_WINDOWS_GUI, _info.getSubsystem());
  EXPECT_EQ(3, _info.getMinOSVersion().majorVersion);
  EXPECT_EQ(0, _info.getMinOSVersion().minorVersion);
}

TEST_F(WinLinkParserTest, MinMajorMinorOSVersion) {
  EXPECT_FALSE(parse("link.exe", "-subsystem", "windows,3.1", "foo.o", nullptr));
  EXPECT_EQ(llvm::COFF::IMAGE_SUBSYSTEM_WINDOWS_GUI, _info.getSubsystem());
  EXPECT_EQ(3, _info.getMinOSVersion().majorVersion);
  EXPECT_EQ(1, _info.getMinOSVersion().minorVersion);
}

TEST_F(WinLinkParserTest, StackReserve) {
  EXPECT_FALSE(parse("link.exe", "-stack", "8192", nullptr));
  EXPECT_EQ(8192ULL, _info.getStackReserve());
  EXPECT_EQ(4096ULL, _info.getStackCommit());
}

TEST_F(WinLinkParserTest, StackReserveAndCommit) {
  EXPECT_FALSE(parse("link.exe", "-stack", "16384,8192", nullptr));
  EXPECT_EQ(16384ULL, _info.getStackReserve());
  EXPECT_EQ(8192ULL, _info.getStackCommit());
}

TEST_F(WinLinkParserTest, HeapReserve) {
  EXPECT_FALSE(parse("link.exe", "-heap", "8192", nullptr));
  EXPECT_EQ(8192ULL, _info.getHeapReserve());
  EXPECT_EQ(4096ULL, _info.getHeapCommit());
}

TEST_F(WinLinkParserTest, HeapReserveAndCommit) {
  EXPECT_FALSE(parse("link.exe", "-heap", "16384,8192", nullptr));
  EXPECT_EQ(16384ULL, _info.getHeapReserve());
  EXPECT_EQ(8192ULL, _info.getHeapCommit());
}

TEST_F(WinLinkParserTest, Force) {
  EXPECT_FALSE(parse("link.exe", "-force", nullptr));
  EXPECT_TRUE(_info.allowRemainingUndefines());
}

TEST_F(WinLinkParserTest, NoNxCompat) {
  EXPECT_FALSE(parse("link.exe", "-nxcompat:no", nullptr));
  EXPECT_FALSE(_info.getNxCompat());
}

TEST_F(WinLinkParserTest, LargeAddressAware) {
  parse("link.exe", "-largeaddressaware", nullptr);
  EXPECT_TRUE(_info.getLargeAddressAware());
}

TEST_F(WinLinkParserTest, NoLargeAddressAware) {
  parse("link.exe", "-largeaddressaware:no", nullptr);
  EXPECT_FALSE(_info.getLargeAddressAware());
}

} // end anonymous namespace
