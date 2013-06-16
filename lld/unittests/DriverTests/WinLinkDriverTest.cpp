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
  virtual PECOFFTargetInfo *doParse(int argc, const char **argv,
                                    raw_ostream &diag) {
    PECOFFTargetInfo *info = new PECOFFTargetInfo();
    EXPECT_FALSE(WinLinkDriver::parse(argc, argv, *info, diag));
    return info;
  }
};

TEST_F(WinLinkParserTest, Basic) {
  parse("link.exe", "-subsystem", "console", "-out", "a.exe",
        "-entry", "_start", "a.obj", "b.obj", "c.obj", nullptr);
  EXPECT_EQ(llvm::COFF::IMAGE_SUBSYSTEM_WINDOWS_CUI, info->getSubsystem());
  EXPECT_EQ("a.exe", info->outputPath());
  EXPECT_EQ("_start", info->entrySymbolName());
  EXPECT_EQ(3, (int)inputFiles.size());
  EXPECT_EQ("a.obj", inputFiles[0]);
  EXPECT_EQ("b.obj", inputFiles[1]);
  EXPECT_EQ("c.obj", inputFiles[2]);
  EXPECT_EQ(6, info->getMinOSVersion().majorVersion);
  EXPECT_EQ(0, info->getMinOSVersion().minorVersion);
  EXPECT_EQ(1024 * 1024, info->getStackReserve());
  EXPECT_EQ(4096, info->getStackCommit());
  EXPECT_FALSE(info->allowRemainingUndefines());
  EXPECT_TRUE(info->getNxCompat());
}

TEST_F(WinLinkParserTest, WindowsStyleOption) {
  parse("link.exe", "/subsystem:console", "/out:a.exe", "a.obj", nullptr);
  EXPECT_EQ(llvm::COFF::IMAGE_SUBSYSTEM_WINDOWS_CUI, info->getSubsystem());
  EXPECT_EQ("a.exe", info->outputPath());
  EXPECT_EQ(1, (int)inputFiles.size());
  EXPECT_EQ("a.obj", inputFiles[0]);
}

TEST_F(WinLinkParserTest, NoFileExtension) {
  parse("link.exe", "foo", "bar", nullptr);
  EXPECT_EQ("foo.exe", info->outputPath());
  EXPECT_EQ(2, (int)inputFiles.size());
  EXPECT_EQ("foo.obj", inputFiles[0]);
  EXPECT_EQ("bar.obj", inputFiles[1]);
}

TEST_F(WinLinkParserTest, NonStandardFileExtension) {
  parse("link.exe", "foo.o", nullptr);
  EXPECT_EQ("foo.exe", info->outputPath());
  EXPECT_EQ(1, (int)inputFiles.size());
  EXPECT_EQ("foo.o", inputFiles[0]);
}

TEST_F(WinLinkParserTest, MinMajorOSVersion) {
  parse("link.exe", "-subsystem", "windows,3", "foo.o", nullptr);
  EXPECT_EQ(llvm::COFF::IMAGE_SUBSYSTEM_WINDOWS_GUI, info->getSubsystem());
  EXPECT_EQ(3, info->getMinOSVersion().majorVersion);
  EXPECT_EQ(0, info->getMinOSVersion().minorVersion);
}

TEST_F(WinLinkParserTest, MinMajorMinorOSVersion) {
  parse("link.exe", "-subsystem", "windows,3.1", "foo.o", nullptr);
  EXPECT_EQ(llvm::COFF::IMAGE_SUBSYSTEM_WINDOWS_GUI, info->getSubsystem());
  EXPECT_EQ(3, info->getMinOSVersion().majorVersion);
  EXPECT_EQ(1, info->getMinOSVersion().minorVersion);
}

TEST_F(WinLinkParserTest, StackReserve) {
  parse("link.exe", "-stack", "8192", nullptr);
  EXPECT_EQ(8192, info->getStackReserve());
  EXPECT_EQ(4096, info->getStackCommit());
}

TEST_F(WinLinkParserTest, StackReserveAndCommit) {
  parse("link.exe", "-stack", "16384,8192", nullptr);
  EXPECT_EQ(16384, info->getStackReserve());
  EXPECT_EQ(8192, info->getStackCommit());
}

TEST_F(WinLinkParserTest, HeapReserve) {
  parse("link.exe", "-heap", "8192", nullptr);
  EXPECT_EQ(8192, info->getHeapReserve());
  EXPECT_EQ(4096, info->getHeapCommit());
}

TEST_F(WinLinkParserTest, HeapReserveAndCommit) {
  parse("link.exe", "-heap", "16384,8192", nullptr);
  EXPECT_EQ(16384, info->getHeapReserve());
  EXPECT_EQ(8192, info->getHeapCommit());
}

TEST_F(WinLinkParserTest, Force) {
  parse("link.exe", "-force", nullptr);
  EXPECT_TRUE(info->allowRemainingUndefines());
}

TEST_F(WinLinkParserTest, NoNxCompat) {
  parse("link.exe", "-nxcompat:no", nullptr);
  EXPECT_FALSE(info->getNxCompat());
}

} // end anonymous namespace
