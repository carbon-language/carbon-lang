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

class ParserTest : public testing::Test {
protected:
  llvm::Optional<moduledef::Directive *> parse(const char *contents) {
    auto membuf =
        std::unique_ptr<MemoryBuffer>(MemoryBuffer::getMemBuffer(contents));
    moduledef::Lexer lexer(std::move(membuf));
    moduledef::Parser parser(lexer, _alloc);
    return parser.parse();
  }

private:
  llvm::BumpPtrAllocator _alloc;
};

TEST_F(ParserTest, Exports) {
  llvm::Optional<moduledef::Directive *> dir =
      parse("EXPORTS\n"
            "  sym1\n"
            "  sym2 @5\n"
            "  sym3 @8 NONAME\n"
            "  sym4 DATA\n"
            "  sym5 @10 NONAME DATA\n");
  EXPECT_TRUE(dir.hasValue());
  EXPECT_EQ(moduledef::Directive::Kind::exports, dir.getValue()->getKind());

  auto *exportsDir = dyn_cast<moduledef::Exports>(dir.getValue());
  EXPECT_TRUE(exportsDir != nullptr);

  const std::vector<PECOFFLinkingContext::ExportDesc> &exports =
      exportsDir->getExports();
  EXPECT_EQ(5U, exports.size());
  EXPECT_EQ(exports[0].name, "sym1");
  EXPECT_EQ(exports[0].ordinal, -1);
  EXPECT_EQ(exports[0].noname, false);
  EXPECT_EQ(exports[0].isData, false);
  EXPECT_EQ(exports[1].name, "sym2");
  EXPECT_EQ(exports[1].ordinal, 5);
  EXPECT_EQ(exports[1].noname, false);
  EXPECT_EQ(exports[1].isData, false);
  EXPECT_EQ(exports[2].name, "sym3");
  EXPECT_EQ(exports[2].ordinal, 8);
  EXPECT_EQ(exports[2].noname, true);
  EXPECT_EQ(exports[2].isData, false);
  EXPECT_EQ(exports[3].name, "sym4");
  EXPECT_EQ(exports[3].ordinal, -1);
  EXPECT_EQ(exports[3].noname, false);
  EXPECT_EQ(exports[3].isData, true);
  EXPECT_EQ(exports[4].name, "sym5");
  EXPECT_EQ(exports[4].ordinal, 10);
  EXPECT_EQ(exports[4].noname, true);
  EXPECT_EQ(exports[4].isData, true);
}

TEST_F(ParserTest, Heapsize) {
  llvm::Optional<moduledef::Directive *> dir = parse("HEAPSIZE 65536");
  EXPECT_TRUE(dir.hasValue());
  auto *heapsize = dyn_cast<moduledef::Heapsize>(dir.getValue());
  EXPECT_TRUE(heapsize != nullptr);
  EXPECT_EQ(65536U, heapsize->getReserve());
  EXPECT_EQ(0U, heapsize->getCommit());
}

TEST_F(ParserTest, Heapsize_WithCommit) {

  llvm::Optional<moduledef::Directive *> dir = parse("HEAPSIZE 65536, 8192");
  EXPECT_TRUE(dir.hasValue());
  auto *heapsize = dyn_cast<moduledef::Heapsize>(dir.getValue());
  EXPECT_TRUE(heapsize != nullptr);
  EXPECT_EQ(65536U, heapsize->getReserve());
  EXPECT_EQ(8192U, heapsize->getCommit());
}

TEST_F(ParserTest, Name) {
  llvm::Optional<moduledef::Directive *> dir = parse("NAME foo.exe");
  EXPECT_TRUE(dir.hasValue());
  auto *name = dyn_cast<moduledef::Name>(dir.getValue());
  EXPECT_TRUE(name != nullptr);
  EXPECT_EQ("foo.exe", name->getOutputPath());
  EXPECT_EQ(0U, name->getBaseAddress());
}

TEST_F(ParserTest, Name_WithBase) {
  llvm::Optional<moduledef::Directive *> dir = parse("NAME foo.exe BASE=4096");
  EXPECT_TRUE(dir.hasValue());
  auto *name = dyn_cast<moduledef::Name>(dir.getValue());
  EXPECT_TRUE(name != nullptr);
  EXPECT_EQ("foo.exe", name->getOutputPath());
  EXPECT_EQ(4096U, name->getBaseAddress());
}

TEST_F(ParserTest, Name_LongFileName) {
  llvm::Optional<moduledef::Directive *> dir =
      parse("NAME \"a long file name.exe\"");
  EXPECT_TRUE(dir.hasValue());
  auto *name = dyn_cast<moduledef::Name>(dir.getValue());
  EXPECT_TRUE(name != nullptr);
  EXPECT_EQ("a long file name.exe", name->getOutputPath());
  EXPECT_EQ(0U, name->getBaseAddress());
}

TEST_F(ParserTest, Version_Major) {
  llvm::Optional<moduledef::Directive *> dir = parse("VERSION 12");
  EXPECT_TRUE(dir.hasValue());
  auto *ver = dyn_cast<moduledef::Version>(dir.getValue());
  EXPECT_TRUE(ver != nullptr);
  EXPECT_EQ(12, ver->getMajorVersion());
  EXPECT_EQ(0, ver->getMinorVersion());
}

TEST_F(ParserTest, Version_MajorMinor) {
  llvm::Optional<moduledef::Directive *> dir = parse("VERSION 12.34");
  EXPECT_TRUE(dir.hasValue());
  auto *ver = dyn_cast<moduledef::Version>(dir.getValue());
  EXPECT_TRUE(ver != nullptr);
  EXPECT_EQ(12, ver->getMajorVersion());
  EXPECT_EQ(34, ver->getMinorVersion());
}
