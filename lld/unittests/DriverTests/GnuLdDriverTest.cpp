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
  const LinkingContext *linkingContext() override { return _ctx.get(); }
};

class LinkerScriptTest : public testing::Test {
protected:
  void SetUp() override {
    llvm::Triple triple(llvm::sys::getDefaultTargetTriple());
    _ctx = GnuLdDriver::createELFLinkingContext(triple);
  }

  void parse(StringRef script, bool nostdlib = false) {
    std::unique_ptr<MemoryBuffer> mb = MemoryBuffer::getMemBuffer(
      script, "foo.so");
    std::string s;
    raw_string_ostream out(s);
    std::error_code ec =
        GnuLdDriver::evalLinkerScript(*_ctx, std::move(mb), out, nostdlib);
    EXPECT_FALSE(ec);
  };

  std::unique_ptr<ELFLinkingContext> _ctx;
};

} // anonymous namespace

TEST_F(GnuLdParserTest, Empty) {
  EXPECT_FALSE(parse("ld", nullptr));
  EXPECT_EQ(linkingContext(), nullptr);
  EXPECT_EQ("No input files\n", errorMessage());
}

// -o

TEST_F(GnuLdParserTest, Output) {
  EXPECT_TRUE(parse("ld", "a.o", "-o", "foo", nullptr));
  EXPECT_EQ("foo", _ctx->outputPath());
}

TEST_F(GnuLdParserTest, OutputDefault) {
  EXPECT_TRUE(parse("ld", "abc.o", nullptr));
  EXPECT_EQ("a.out", _ctx->outputPath());
}

// --noinhibit-exec

TEST_F(GnuLdParserTest, NoinhibitExec) {
  EXPECT_TRUE(parse("ld", "a.o", "--noinhibit-exec", nullptr));
  EXPECT_TRUE(_ctx->allowRemainingUndefines());
}

// --entry

TEST_F(GnuLdParserTest, Entry) {
  EXPECT_TRUE(parse("ld", "a.o", "--entry", "foo", nullptr));
  EXPECT_EQ("foo", _ctx->entrySymbolName());
}

TEST_F(GnuLdParserTest, EntryShort) {
  EXPECT_TRUE(parse("ld", "a.o", "-e", "foo", nullptr));
  EXPECT_EQ("foo", _ctx->entrySymbolName());
}

TEST_F(GnuLdParserTest, EntryJoined) {
  EXPECT_TRUE(parse("ld", "a.o", "--entry=foo", nullptr));
  EXPECT_EQ("foo", _ctx->entrySymbolName());
}

// --export-dynamic

TEST_F(GnuLdParserTest, ExportDynamic) {
  EXPECT_TRUE(parse("ld", "a.o", "--export-dynamic", nullptr));
  EXPECT_TRUE(_ctx->shouldExportDynamic());
}

TEST_F(GnuLdParserTest, NoExportDynamic) {
  EXPECT_TRUE(parse("ld", "a.o", "--no-export-dynamic", nullptr));
  EXPECT_FALSE(_ctx->shouldExportDynamic());
}

// --init

TEST_F(GnuLdParserTest, Init) {
  EXPECT_TRUE(parse("ld", "a.o", "-init", "foo", "-init", "bar", nullptr));
  EXPECT_EQ("bar", _ctx->initFunction());
}

TEST_F(GnuLdParserTest, InitJoined) {
  EXPECT_TRUE(parse("ld", "a.o", "-init=foo", nullptr));
  EXPECT_EQ("foo", _ctx->initFunction());
}

// --soname

TEST_F(GnuLdParserTest, SOName) {
  EXPECT_TRUE(parse("ld", "a.o", "--soname=foo", nullptr));
  EXPECT_EQ("foo", _ctx->sharedObjectName());
}

TEST_F(GnuLdParserTest, SONameSingleDash) {
  EXPECT_TRUE(parse("ld", "a.o", "-soname=foo", nullptr));
  EXPECT_EQ("foo", _ctx->sharedObjectName());
}

TEST_F(GnuLdParserTest, SONameH) {
  EXPECT_TRUE(parse("ld", "a.o", "-h", "foo", nullptr));
  EXPECT_EQ("foo", _ctx->sharedObjectName());
}

// -rpath

TEST_F(GnuLdParserTest, Rpath) {
  EXPECT_TRUE(parse("ld", "a.o", "-rpath", "foo:bar", nullptr));
  EXPECT_EQ(2, _ctx->getRpathList().size());
  EXPECT_EQ("foo", _ctx->getRpathList()[0]);
  EXPECT_EQ("bar", _ctx->getRpathList()[1]);
}

TEST_F(GnuLdParserTest, RpathEq) {
  EXPECT_TRUE(parse("ld", "a.o", "-rpath=foo", nullptr));
  EXPECT_EQ(1, _ctx->getRpathList().size());
  EXPECT_EQ("foo", _ctx->getRpathList()[0]);
}

// --defsym

TEST_F(GnuLdParserTest, DefsymDecimal) {
  EXPECT_TRUE(parse("ld", "a.o", "--defsym=sym=1000", nullptr));
  assert(_ctx.get());
  auto map = _ctx->getAbsoluteSymbols();
  EXPECT_EQ((size_t)1, map.size());
  EXPECT_EQ((uint64_t)1000, map["sym"]);
}

TEST_F(GnuLdParserTest, DefsymHexadecimal) {
  EXPECT_TRUE(parse("ld", "a.o", "--defsym=sym=0x1000", nullptr));
  auto map = _ctx->getAbsoluteSymbols();
  EXPECT_EQ((size_t)1, map.size());
  EXPECT_EQ((uint64_t)0x1000, map["sym"]);
}

TEST_F(GnuLdParserTest, DefsymAlias) {
  EXPECT_TRUE(parse("ld", "a.o", "--defsym=sym=abc", nullptr));
  auto map = _ctx->getAliases();
  EXPECT_EQ((size_t)1, map.size());
  EXPECT_EQ("abc", map["sym"]);
}

TEST_F(GnuLdParserTest, DefsymOctal) {
  EXPECT_TRUE(parse("ld", "a.o", "--defsym=sym=0777", nullptr));
  auto map = _ctx->getAbsoluteSymbols();
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
  std::vector<std::unique_ptr<Node>> &nodes = _ctx->getNodes();
  EXPECT_EQ((size_t)4, nodes.size());
  EXPECT_FALSE(cast<FileNode>(nodes[0].get())->asNeeded());
  EXPECT_TRUE(cast<FileNode>(nodes[1].get())->asNeeded());
  EXPECT_TRUE(cast<FileNode>(nodes[2].get())->asNeeded());
  EXPECT_FALSE(cast<FileNode>(nodes[3].get())->asNeeded());
}

// Linker script

TEST_F(LinkerScriptTest, Input) {
  parse("INPUT(/x /y)");
  std::vector<std::unique_ptr<Node>> &nodes = _ctx->getNodes();
  EXPECT_EQ((size_t)2, nodes.size());
  EXPECT_EQ("/x", cast<FileNode>(nodes[0].get())->getFile()->path());
  EXPECT_EQ("/y", cast<FileNode>(nodes[1].get())->getFile()->path());
}

TEST_F(LinkerScriptTest, Group) {
  parse("GROUP(/x /y)");
  std::vector<std::unique_ptr<Node>> &nodes = _ctx->getNodes();
  EXPECT_EQ((size_t)3, nodes.size());
  EXPECT_EQ("/x", cast<FileNode>(nodes[0].get())->getFile()->path());
  EXPECT_EQ("/y", cast<FileNode>(nodes[1].get())->getFile()->path());
  EXPECT_EQ(2, cast<GroupEnd>(nodes[2].get())->getSize());
}

TEST_F(LinkerScriptTest, SearchDir) {
  parse("SEARCH_DIR(\"/foo/bar\")");
  std::vector<StringRef> paths = _ctx->getSearchPaths();
  EXPECT_EQ((size_t)1, paths.size());
  EXPECT_EQ("/foo/bar", paths[0]);
}

TEST_F(LinkerScriptTest, Entry) {
  parse("ENTRY(blah)");
  EXPECT_EQ("blah", _ctx->entrySymbolName());
}

TEST_F(LinkerScriptTest, Output) {
  parse("OUTPUT(\"/path/to/output\")");
  EXPECT_EQ("/path/to/output", _ctx->outputPath());
}

// Test that search paths are ignored when nostdlib is set.
TEST_F(LinkerScriptTest, IgnoreSearchDirNoStdLib) {
  parse("SEARCH_DIR(\"/foo/bar\")", true /*nostdlib*/);
  std::vector<StringRef> paths = _ctx->getSearchPaths();
  EXPECT_EQ((size_t)0, paths.size());
}

TEST_F(LinkerScriptTest, ExprEval) {
  parse("SECTIONS { symbol = 0x4000 + 0x40; \n"
        ". = (symbol >= 0x4040)? (0x5001 * 2 & 0xFFF0) << 1 : 0}");

  EXPECT_EQ((size_t)1, _ctx->linkerScriptSema().getLinkerScripts().size());

  script::LinkerScript *ls =
      _ctx->linkerScriptSema().getLinkerScripts()[0]->get();
  EXPECT_EQ((size_t)1, ls->_commands.size());

  auto *secs = dyn_cast<const script::Sections>(*ls->_commands.begin());
  EXPECT_TRUE(secs != nullptr);
  EXPECT_EQ(2, secs->end() - secs->begin());

  auto command = secs->begin();
  auto *sa1 = dyn_cast<const script::SymbolAssignment>(*command);
  EXPECT_TRUE(sa1 != nullptr);
  EXPECT_EQ(script::SymbolAssignment::Simple, sa1->assignmentKind());
  EXPECT_EQ(script::SymbolAssignment::Default, sa1->assignmentVisibility());

  ++command;
  auto *sa2 = dyn_cast<const script::SymbolAssignment>(*command);
  EXPECT_TRUE(sa2 != nullptr);
  EXPECT_EQ(script::SymbolAssignment::Simple, sa2->assignmentKind());
  EXPECT_EQ(script::SymbolAssignment::Default, sa2->assignmentVisibility());

  script::Expression::SymbolTableTy mySymbolTable;
  auto ans = sa1->expr()->evalExpr(mySymbolTable);
  EXPECT_FALSE(ans.getError());
  int64_t result = *ans;
  EXPECT_EQ(0x4040, result);
  mySymbolTable[sa1->symbol()] = result;

  auto ans2 = sa2->expr()->evalExpr(mySymbolTable);
  EXPECT_FALSE(ans2.getError());
  result = *ans2;
  EXPECT_EQ(0x14000, result);
  EXPECT_EQ(0, sa2->symbol().compare(StringRef(".")));
}

