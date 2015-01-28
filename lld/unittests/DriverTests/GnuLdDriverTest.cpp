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
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
using namespace lld;

namespace {
class GnuLdParserTest
    : public ParserTest<GnuLdDriver, std::unique_ptr<ELFLinkingContext>> {
protected:
  const LinkingContext *linkingContext() override { return _context.get(); }
};
}

TEST_F(GnuLdParserTest, Empty) {
  EXPECT_FALSE(parse("ld", nullptr));
  EXPECT_EQ(linkingContext(), nullptr);
  EXPECT_EQ("No input files\n", errorMessage());
}

// -o

TEST_F(GnuLdParserTest, Output) {
  EXPECT_TRUE(parse("ld", "a.o", "-o", "foo", nullptr));
  EXPECT_EQ("foo", _context->outputPath());
}

// --noinhibit-exec

TEST_F(GnuLdParserTest, NoinhibitExec) {
  EXPECT_TRUE(parse("ld", "a.o", "--noinhibit-exec", nullptr));
  EXPECT_TRUE(_context->allowRemainingUndefines());
}

// --entry

TEST_F(GnuLdParserTest, Entry) {
  EXPECT_TRUE(parse("ld", "a.o", "--entry", "foo", nullptr));
  EXPECT_EQ("foo", _context->entrySymbolName());
}

TEST_F(GnuLdParserTest, EntryShort) {
  EXPECT_TRUE(parse("ld", "a.o", "-e", "foo", nullptr));
  EXPECT_EQ("foo", _context->entrySymbolName());
}

TEST_F(GnuLdParserTest, EntryJoined) {
  EXPECT_TRUE(parse("ld", "a.o", "--entry=foo", nullptr));
  EXPECT_EQ("foo", _context->entrySymbolName());
}

// --init

TEST_F(GnuLdParserTest, Init) {
  EXPECT_TRUE(parse("ld", "a.o", "-init", "foo", "-init", "bar", nullptr));
  EXPECT_EQ("bar", _context->initFunction());
}

TEST_F(GnuLdParserTest, InitJoined) {
  EXPECT_TRUE(parse("ld", "a.o", "-init=foo", nullptr));
  EXPECT_EQ("foo", _context->initFunction());
}

// --soname

TEST_F(GnuLdParserTest, SOName) {
  EXPECT_TRUE(parse("ld", "a.o", "--soname=foo", nullptr));
  EXPECT_EQ("foo", _context->sharedObjectName());
}

TEST_F(GnuLdParserTest, SONameSingleDash) {
  EXPECT_TRUE(parse("ld", "a.o", "-soname=foo", nullptr));
  EXPECT_EQ("foo", _context->sharedObjectName());
}

TEST_F(GnuLdParserTest, SONameH) {
  EXPECT_TRUE(parse("ld", "a.o", "-h", "foo", nullptr));
  EXPECT_EQ("foo", _context->sharedObjectName());
}

// -rpath

TEST_F(GnuLdParserTest, Rpath) {
  EXPECT_TRUE(parse("ld", "a.o", "-rpath", "foo:bar", nullptr));
  EXPECT_EQ(2, _context->getRpathList().size());
  EXPECT_EQ("foo", _context->getRpathList()[0]);
  EXPECT_EQ("bar", _context->getRpathList()[1]);
}

TEST_F(GnuLdParserTest, RpathEq) {
  EXPECT_TRUE(parse("ld", "a.o", "-rpath=foo", nullptr));
  EXPECT_EQ(1, _context->getRpathList().size());
  EXPECT_EQ("foo", _context->getRpathList()[0]);
}

// --defsym

TEST_F(GnuLdParserTest, DefsymDecimal) {
  EXPECT_TRUE(parse("ld", "a.o", "--defsym=sym=1000", nullptr));
  assert(_context.get());
  auto map = _context->getAbsoluteSymbols();
  EXPECT_EQ((size_t)1, map.size());
  EXPECT_EQ((uint64_t)1000, map["sym"]);
}

TEST_F(GnuLdParserTest, DefsymHexadecimal) {
  EXPECT_TRUE(parse("ld", "a.o", "--defsym=sym=0x1000", nullptr));
  auto map = _context->getAbsoluteSymbols();
  EXPECT_EQ((size_t)1, map.size());
  EXPECT_EQ((uint64_t)0x1000, map["sym"]);
}

TEST_F(GnuLdParserTest, DefsymAlias) {
  EXPECT_TRUE(parse("ld", "a.o", "--defsym=sym=abc", nullptr));
  auto map = _context->getAliases();
  EXPECT_EQ((size_t)1, map.size());
  EXPECT_EQ("abc", map["sym"]);
}

TEST_F(GnuLdParserTest, DefsymOctal) {
  EXPECT_TRUE(parse("ld", "a.o", "--defsym=sym=0777", nullptr));
  auto map = _context->getAbsoluteSymbols();
  EXPECT_EQ((size_t)1, map.size());
  EXPECT_EQ((uint64_t)0777, map["sym"]);
}

TEST_F(GnuLdParserTest, DefsymMisssingSymbol) {
  EXPECT_FALSE(parse("ld", "a.o", "--defsym==0", nullptr));
}

TEST_F(GnuLdParserTest, DefsymMisssingValue) {
  EXPECT_FALSE(parse("ld", "a.o", "--defsym=sym=", nullptr));
}

// --as-needed

TEST_F(GnuLdParserTest, AsNeeded) {
  EXPECT_TRUE(parse("ld", "a.o", "--as-needed", "b.o", "c.o",
                    "--no-as-needed", "d.o", nullptr));
  std::vector<std::unique_ptr<Node>> &nodes = _context->getNodes();
  EXPECT_EQ((size_t)4, nodes.size());
  EXPECT_FALSE(cast<FileNode>(nodes[0].get())->asNeeded());
  EXPECT_TRUE(cast<FileNode>(nodes[1].get())->asNeeded());
  EXPECT_TRUE(cast<FileNode>(nodes[2].get())->asNeeded());
  EXPECT_FALSE(cast<FileNode>(nodes[3].get())->asNeeded());
}

// Linker script

TEST_F(GnuLdParserTest, LinkerScriptGroup) {
  parse("ld", "a.o", nullptr);
  std::unique_ptr<MemoryBuffer> mb = MemoryBuffer::getMemBuffer(
    "GROUP(/x /y)", "foo.so");
  std::string s;
  raw_string_ostream out(s);
  std::error_code ec = GnuLdDriver::evalLinkerScript(
    *_context, std::move(mb), out);
  EXPECT_FALSE(ec);
  std::vector<std::unique_ptr<Node>> &nodes = _context->getNodes();
  EXPECT_EQ((size_t)4, nodes.size());
  EXPECT_EQ("/x", cast<FileNode>(nodes[1].get())->getFile()->path());
  EXPECT_EQ("/y", cast<FileNode>(nodes[2].get())->getFile()->path());
  EXPECT_EQ(2, cast<GroupEnd>(nodes[3].get())->getSize());
}
