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

#include <vector>

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
  EXPECT_FALSE(parse("link.exe", "/subsystem:console", "/out:a.exe",
        "-entry:_start", "a.obj", "b.obj", "c.obj", nullptr));
  EXPECT_EQ(llvm::COFF::IMAGE_SUBSYSTEM_WINDOWS_CUI, _info.getSubsystem());
  EXPECT_EQ("a.exe", _info.outputPath());
  EXPECT_EQ("_start", _info.entrySymbolName());
  EXPECT_EQ(3, inputFileCount());
  EXPECT_EQ("a.obj", inputFile(0));
  EXPECT_EQ("b.obj", inputFile(1));
  EXPECT_EQ("c.obj", inputFile(2));
  EXPECT_TRUE(_info.getInputSearchPaths().empty());

  // Unspecified flags will have default values.
  EXPECT_EQ(6, _info.getMinOSVersion().majorVersion);
  EXPECT_EQ(0, _info.getMinOSVersion().minorVersion);
  EXPECT_EQ(0x400000U, _info.getBaseAddress());
  EXPECT_EQ(1024 * 1024U, _info.getStackReserve());
  EXPECT_EQ(4096U, _info.getStackCommit());
  EXPECT_FALSE(_info.allowRemainingUndefines());
  EXPECT_TRUE(_info.isNxCompat());
  EXPECT_FALSE(_info.getLargeAddressAware());
  EXPECT_TRUE(_info.getBaseRelocationEnabled());
  EXPECT_TRUE(_info.isTerminalServerAware());
  EXPECT_TRUE(_info.initialUndefinedSymbols().empty());
}

TEST_F(WinLinkParserTest, UnixStyleOption) {
  EXPECT_FALSE(parse("link.exe", "-subsystem", "console", "-out", "a.exe",
                     "a.obj", nullptr));
  EXPECT_EQ(llvm::COFF::IMAGE_SUBSYSTEM_WINDOWS_CUI, _info.getSubsystem());
  EXPECT_EQ("a.exe", _info.outputPath());
  EXPECT_EQ(1, inputFileCount());
  EXPECT_EQ("a.obj", inputFile(0));
}

TEST_F(WinLinkParserTest, Mllvm) {
  EXPECT_FALSE(parse("link.exe", "-mllvm", "-debug", "a.obj", nullptr));
  const std::vector<const char *> &options = _info.llvmOptions();
  EXPECT_EQ(1U, options.size());
  EXPECT_EQ("-debug", options[0]);
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

TEST_F(WinLinkParserTest, Libpath) {
  EXPECT_FALSE(parse("link.exe", "/libpath:dir1", "/libpath:dir2",
                     "a.obj", nullptr));
  const std::vector<StringRef> &paths = _info.getInputSearchPaths();
  EXPECT_EQ(2U, paths.size());
  EXPECT_EQ("dir1", paths[0]);
  EXPECT_EQ("dir2", paths[1]);
}

TEST_F(WinLinkParserTest, MinMajorOSVersion) {
  EXPECT_FALSE(parse("link.exe", "/subsystem:windows,3", "foo.o", nullptr));
  EXPECT_EQ(llvm::COFF::IMAGE_SUBSYSTEM_WINDOWS_GUI, _info.getSubsystem());
  EXPECT_EQ(3, _info.getMinOSVersion().majorVersion);
  EXPECT_EQ(0, _info.getMinOSVersion().minorVersion);
}

TEST_F(WinLinkParserTest, MinMajorMinorOSVersion) {
  EXPECT_FALSE(parse("link.exe", "/subsystem:windows,3.1", "foo.o", nullptr));
  EXPECT_EQ(llvm::COFF::IMAGE_SUBSYSTEM_WINDOWS_GUI, _info.getSubsystem());
  EXPECT_EQ(3, _info.getMinOSVersion().majorVersion);
  EXPECT_EQ(1, _info.getMinOSVersion().minorVersion);
}

TEST_F(WinLinkParserTest, DefaultLib) {
  EXPECT_FALSE(parse("link.exe", "/defaultlib:user32.lib", "a.obj", nullptr));
  EXPECT_EQ(2, inputFileCount());
  EXPECT_EQ("a.obj", inputFile(0));
  EXPECT_EQ("user32.lib", inputFile(1));
}

TEST_F(WinLinkParserTest, Base) {
  EXPECT_FALSE(parse("link.exe", "/base:8388608", "a.obj", nullptr));
  EXPECT_EQ(0x800000U, _info.getBaseAddress());
}

TEST_F(WinLinkParserTest, StackReserve) {
  EXPECT_FALSE(parse("link.exe", "/stack:8192", "a.obj", nullptr));
  EXPECT_EQ(8192U, _info.getStackReserve());
  EXPECT_EQ(4096U, _info.getStackCommit());
}

TEST_F(WinLinkParserTest, StackReserveAndCommit) {
  EXPECT_FALSE(parse("link.exe", "/stack:16384,8192", "a.obj", nullptr));
  EXPECT_EQ(16384U, _info.getStackReserve());
  EXPECT_EQ(8192U, _info.getStackCommit());
}

TEST_F(WinLinkParserTest, HeapReserve) {
  EXPECT_FALSE(parse("link.exe", "/heap:8192", "a.obj", nullptr));
  EXPECT_EQ(8192U, _info.getHeapReserve());
  EXPECT_EQ(4096U, _info.getHeapCommit());
}

TEST_F(WinLinkParserTest, HeapReserveAndCommit) {
  EXPECT_FALSE(parse("link.exe", "/heap:16384,8192", "a.obj", nullptr));
  EXPECT_EQ(16384U, _info.getHeapReserve());
  EXPECT_EQ(8192U, _info.getHeapCommit());
}

TEST_F(WinLinkParserTest, Force) {
  EXPECT_FALSE(parse("link.exe", "/force", "a.obj", nullptr));
  EXPECT_TRUE(_info.allowRemainingUndefines());
}

TEST_F(WinLinkParserTest, NoNxCompat) {
  EXPECT_FALSE(parse("link.exe", "/nxcompat:no", "a.obj", nullptr));
  EXPECT_FALSE(_info.isNxCompat());
}

TEST_F(WinLinkParserTest, LargeAddressAware) {
  EXPECT_FALSE(parse("link.exe", "/largeaddressaware", "a.obj", nullptr));
  EXPECT_TRUE(_info.getLargeAddressAware());
}

TEST_F(WinLinkParserTest, NoLargeAddressAware) {
  EXPECT_FALSE(parse("link.exe", "/largeaddressaware:no", "a.obj", nullptr));
  EXPECT_FALSE(_info.getLargeAddressAware());
}

TEST_F(WinLinkParserTest, Fixed) {
  EXPECT_FALSE(parse("link.exe", "/fixed", "a.out", nullptr));
  EXPECT_FALSE(_info.getBaseRelocationEnabled());
}

TEST_F(WinLinkParserTest, NoFixed) {
  EXPECT_FALSE(parse("link.exe", "/fixed:no", "a.out", nullptr));
  EXPECT_TRUE(_info.getBaseRelocationEnabled());
}

TEST_F(WinLinkParserTest, TerminalServerAware) {
  EXPECT_FALSE(parse("link.exe", "/tsaware", "a.out", nullptr));
  EXPECT_TRUE(_info.isTerminalServerAware());
}

TEST_F(WinLinkParserTest, NoTerminalServerAware) {
  EXPECT_FALSE(parse("link.exe", "/tsaware:no", "a.out", nullptr));
  EXPECT_FALSE(_info.isTerminalServerAware());
}

TEST_F(WinLinkParserTest, Include) {
  EXPECT_FALSE(parse("link.exe", "/include:foo", "a.out", nullptr));
  auto symbols = _info.initialUndefinedSymbols();
  EXPECT_FALSE(symbols.empty());
  EXPECT_EQ("foo", symbols[0]);
  symbols.pop_front();
  EXPECT_TRUE(symbols.empty());
}

TEST_F(WinLinkParserTest, NoInputFiles) {
  EXPECT_TRUE(parse("link.exe", nullptr));
  EXPECT_EQ("No input files\n", errorMessage());
}

TEST_F(WinLinkParserTest, FailIfMismatch_Match) {
  EXPECT_FALSE(parse("link.exe", "/failifmismatch:foo=bar",
                     "/failifmismatch:foo=bar", "/failifmismatch:abc=def",
                     "a.out", nullptr));
}

TEST_F(WinLinkParserTest, FailIfMismatch_Mismatch) {
  EXPECT_TRUE(parse("link.exe", "/failifmismatch:foo=bar",
                    "/failifmismatch:foo=baz", "a.out", nullptr));
}

TEST_F(WinLinkParserTest, Nologo) {
  // NOLOGO flag is for link.exe compatibility. It's recognized but is ignored.
  EXPECT_FALSE(parse("link.exe", "/nologo", "a.obj", nullptr));
  EXPECT_EQ("", errorMessage());
  EXPECT_EQ(1, inputFileCount());
  EXPECT_EQ("a.obj", inputFile(0));
}

} // end anonymous namespace
