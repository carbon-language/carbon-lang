//===- lld/unittest/WinLinkModuleDefTest.cpp ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <memory>

#include "gtest/gtest.h"
#include "lld/Driver/WinLinkModuleDef.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
using namespace lld;

template <typename T> class ParserTest : public testing::Test {
protected:
  T *parse(const char *contents) {
    auto membuf =
        std::unique_ptr<MemoryBuffer>(MemoryBuffer::getMemBuffer(contents));
    moduledef::Lexer lexer(std::move(membuf));
    moduledef::Parser parser(lexer, _alloc);
    llvm::Optional<moduledef::Directive *> dir = parser.parse();
    EXPECT_TRUE(dir.hasValue());
    T *ret = dyn_cast<T>(dir.getValue());
    EXPECT_TRUE(ret != nullptr);
    return ret;
  }

private:
  llvm::BumpPtrAllocator _alloc;
};

class ExportsTest : public ParserTest<moduledef::Exports> {
public:
  void verifyExportDesc(const PECOFFLinkingContext::ExportDesc &exp,
                        StringRef sym, int ordinal, bool noname, bool isData) {
    EXPECT_EQ(sym, exp.name);
    EXPECT_EQ(ordinal, exp.ordinal);
    EXPECT_EQ(noname, exp.noname);
    EXPECT_EQ(isData, exp.isData);
  }
};

class HeapsizeTest : public ParserTest<moduledef::Heapsize> {};
class StacksizeTest : public ParserTest<moduledef::Stacksize> {};
class NameTest : public ParserTest<moduledef::Name> {};
class VersionTest : public ParserTest<moduledef::Version> {};

TEST_F(ExportsTest, Basic) {
  moduledef::Exports *dir = parse("EXPORTS\n"
                                  "  sym1\n"
                                  "  sym2 @5\n"
                                  "  sym3 @8 NONAME\n"
                                  "  sym4 DATA\n"
                                  "  sym5 @10 NONAME DATA\n");
  const std::vector<PECOFFLinkingContext::ExportDesc> &exports =
      dir->getExports();
  EXPECT_EQ(5U, exports.size());
  verifyExportDesc(exports[0], "sym1", -1, false, false);
  verifyExportDesc(exports[1], "sym2", 5, false, false);
  verifyExportDesc(exports[2], "sym3", 8, true, false);
  verifyExportDesc(exports[3], "sym4", -1, false, true);
  verifyExportDesc(exports[4], "sym5", 10, true, true);
}

TEST_F(HeapsizeTest, Basic) {
  moduledef::Heapsize *heapsize = parse("HEAPSIZE 65536");
  EXPECT_EQ(65536U, heapsize->getReserve());
  EXPECT_EQ(0U, heapsize->getCommit());
}

TEST_F(HeapsizeTest, WithCommit) {
  moduledef::Heapsize *heapsize = parse("HEAPSIZE 65536, 8192");
  EXPECT_EQ(65536U, heapsize->getReserve());
  EXPECT_EQ(8192U, heapsize->getCommit());
}

TEST_F(StacksizeTest, Basic) {
  moduledef::Stacksize *stacksize = parse("STACKSIZE 65536");
  EXPECT_EQ(65536U, stacksize->getReserve());
  EXPECT_EQ(0U, stacksize->getCommit());
}

TEST_F(StacksizeTest, WithCommit) {
  moduledef::Stacksize *stacksize = parse("STACKSIZE 65536, 8192");
  EXPECT_EQ(65536U, stacksize->getReserve());
  EXPECT_EQ(8192U, stacksize->getCommit());
}

TEST_F(NameTest, Basic) {
  moduledef::Name *name = parse("NAME foo.exe");
  EXPECT_EQ("foo.exe", name->getOutputPath());
  EXPECT_EQ(0U, name->getBaseAddress());
}

TEST_F(NameTest, WithBase) {
  moduledef::Name *name = parse("NAME foo.exe BASE=4096");
  EXPECT_EQ("foo.exe", name->getOutputPath());
  EXPECT_EQ(4096U, name->getBaseAddress());
}

TEST_F(NameTest, LongFileName) {
  moduledef::Name *name = parse("NAME \"a long file name.exe\"");
  EXPECT_EQ("a long file name.exe", name->getOutputPath());
  EXPECT_EQ(0U, name->getBaseAddress());
}

TEST_F(VersionTest, Major) {
  moduledef::Version *ver = parse("VERSION 12");
  EXPECT_EQ(12, ver->getMajorVersion());
  EXPECT_EQ(0, ver->getMinorVersion());
}

TEST_F(VersionTest, MajorMinor) {
  moduledef::Version *ver = parse("VERSION 12.34");
  EXPECT_EQ(12, ver->getMajorVersion());
  EXPECT_EQ(34, ver->getMinorVersion());
}
