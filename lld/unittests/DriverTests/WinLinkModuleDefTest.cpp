//===- lld/unittest/WinLinkModuleDefTest.cpp ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "lld/Driver/WinLinkModuleDef.h"
#include "llvm/Support/MemoryBuffer.h"
#include <memory>

using namespace llvm;
using namespace lld;

class ParserTest : public testing::Test {
protected:
  std::vector<moduledef::Directive *> _dirs;

  void parse(const char *contents) {
    auto membuf =
        std::unique_ptr<MemoryBuffer>(MemoryBuffer::getMemBuffer(contents));
    moduledef::Lexer lexer(std::move(membuf));
    moduledef::Parser parser(lexer, _alloc);
    EXPECT_TRUE(parser.parse(_dirs));
    EXPECT_TRUE(!_dirs.empty());
  }

  void verifyExportDesc(const PECOFFLinkingContext::ExportDesc &exp,
                        StringRef sym, int ordinal, bool noname, bool isData) {
    EXPECT_EQ(sym, exp.name);
    EXPECT_EQ(ordinal, exp.ordinal);
    EXPECT_EQ(noname, exp.noname);
    EXPECT_EQ(isData, exp.isData);
  }

private:
  llvm::BumpPtrAllocator _alloc;
};

TEST_F(ParserTest, Exports) {
  parse("EXPORTS\n"
        "  sym1\n"
        "  sym2 @5\n"
        "  sym3 @8 NONAME\n"
        "  sym4 DATA\n"
        "  sym5 @10 NONAME DATA\n");
  EXPECT_EQ(1U, _dirs.size());
  const std::vector<PECOFFLinkingContext::ExportDesc> &exports =
      cast<moduledef::Exports>(_dirs[0])->getExports();
  EXPECT_EQ(5U, exports.size());
  verifyExportDesc(exports[0], "sym1", -1, false, false);
  verifyExportDesc(exports[1], "sym2", 5, false, false);
  verifyExportDesc(exports[2], "sym3", 8, true, false);
  verifyExportDesc(exports[3], "sym4", -1, false, true);
  verifyExportDesc(exports[4], "sym5", 10, true, true);
}

TEST_F(ParserTest, Heapsize) {
  parse("HEAPSIZE 65536");
  EXPECT_EQ(1U, _dirs.size());
  auto *heapsize = cast<moduledef::Heapsize>(_dirs[0]);
  EXPECT_EQ(65536U, heapsize->getReserve());
  EXPECT_EQ(0U, heapsize->getCommit());
}

TEST_F(ParserTest, HeapsizeWithCommit) {
  parse("HEAPSIZE 65536, 8192");
  EXPECT_EQ(1U, _dirs.size());
  auto *heapsize = cast<moduledef::Heapsize>(_dirs[0]);
  EXPECT_EQ(65536U, heapsize->getReserve());
  EXPECT_EQ(8192U, heapsize->getCommit());
}

TEST_F(ParserTest, StacksizeBasic) {
  parse("STACKSIZE 65536");
  EXPECT_EQ(1U, _dirs.size());
  auto *stacksize = cast<moduledef::Stacksize>(_dirs[0]);
  EXPECT_EQ(65536U, stacksize->getReserve());
  EXPECT_EQ(0U, stacksize->getCommit());
}

TEST_F(ParserTest, StacksizeWithCommit) {
  parse("STACKSIZE 65536, 8192");
  EXPECT_EQ(1U, _dirs.size());
  auto *stacksize = cast<moduledef::Stacksize>(_dirs[0]);
  EXPECT_EQ(65536U, stacksize->getReserve());
  EXPECT_EQ(8192U, stacksize->getCommit());
}

TEST_F(ParserTest, Library) {
  parse("LIBRARY foo.dll");
  EXPECT_EQ(1U, _dirs.size());
  auto *lib = cast<moduledef::Library>(_dirs[0]);
  EXPECT_EQ("foo.dll", lib->getName());
}

TEST_F(ParserTest, NameBasic) {
  parse("NAME foo.exe");
  EXPECT_EQ(1U, _dirs.size());
  auto *name = cast<moduledef::Name>(_dirs[0]);
  EXPECT_EQ("foo.exe", name->getOutputPath());
  EXPECT_EQ(0U, name->getBaseAddress());
}

TEST_F(ParserTest, NameWithBase) {
  parse("NAME foo.exe BASE=4096");
  EXPECT_EQ(1U, _dirs.size());
  auto *name = cast<moduledef::Name>(_dirs[0]);
  EXPECT_EQ("foo.exe", name->getOutputPath());
  EXPECT_EQ(4096U, name->getBaseAddress());
}

TEST_F(ParserTest, NameLongFileName) {
  parse("NAME \"a long file name.exe\"");
  EXPECT_EQ(1U, _dirs.size());
  auto *name = cast<moduledef::Name>(_dirs[0]);
  EXPECT_EQ("a long file name.exe", name->getOutputPath());
  EXPECT_EQ(0U, name->getBaseAddress());
}

TEST_F(ParserTest, VersionMajor) {
  parse("VERSION 12");
  EXPECT_EQ(1U, _dirs.size());
  auto *ver = cast<moduledef::Version>(_dirs[0]);
  EXPECT_EQ(12, ver->getMajorVersion());
  EXPECT_EQ(0, ver->getMinorVersion());
}

TEST_F(ParserTest, VersionMajorMinor) {
  parse("VERSION 12.34");
  EXPECT_EQ(1U, _dirs.size());
  auto *ver = cast<moduledef::Version>(_dirs[0]);
  EXPECT_EQ(12, ver->getMajorVersion());
  EXPECT_EQ(34, ver->getMinorVersion());
}

TEST_F(ParserTest, Multiple) {
  parse("LIBRARY foo\n"
        "EXPORTS sym\n"
        "VERSION 12");
  EXPECT_EQ(3U, _dirs.size());
  auto *lib = cast<moduledef::Library>(_dirs[0]);
  EXPECT_EQ("foo.dll", lib->getName());

  const std::vector<PECOFFLinkingContext::ExportDesc> &exports =
      cast<moduledef::Exports>(_dirs[1])->getExports();
  EXPECT_EQ(1U, exports.size());
  verifyExportDesc(exports[0], "sym", -1, false, false);

  auto *ver = cast<moduledef::Version>(_dirs[2]);
  EXPECT_EQ(12, ver->getMajorVersion());
}
