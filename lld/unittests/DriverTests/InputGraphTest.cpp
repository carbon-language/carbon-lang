//===- lld/unittest/InputGraphTest.cpp -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief InputGraph Tests
///
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "lld/Core/InputGraph.h"
#include "lld/Core/Resolver.h"
#include "lld/Core/Simple.h"

using namespace lld;

namespace {

class TestLinkingContext : public LinkingContext {
public:
  Writer &writer() const override { llvm_unreachable("no writer!"); }
  bool validateImpl(raw_ostream &) override { return true; }
};

class TestExpandFileNode : public SimpleFileNode {
public:
  TestExpandFileNode(StringRef path) : SimpleFileNode(path) {}

  void addElement(std::unique_ptr<InputElement> element) {
    _expandElements.push_back(std::move(element));
  }

private:
  InputGraph::InputElementVectorT _expandElements;
};

class InputGraphTest : public testing::Test {
public:
  InputGraphTest() {
    _ctx.setInputGraph(std::unique_ptr<InputGraph>(new InputGraph()));
    _graph = &_ctx.getInputGraph();
  }

  StringRef getNext() {
    File *file = _graph->getNextFile();
    EXPECT_TRUE(file);
    return file->path();
  }

  void expectEnd() {
    File *file = _graph->getNextFile();
    EXPECT_TRUE(file == nullptr);
  }

protected:
  TestLinkingContext _ctx;
  InputGraph *_graph;
};

} // end anonymous namespace

static std::unique_ptr<SimpleFileNode> createFile(StringRef name) {
  std::vector<std::unique_ptr<File>> files;
  files.push_back(std::unique_ptr<SimpleFile>(new SimpleFile(name)));
  std::unique_ptr<SimpleFileNode> file(new SimpleFileNode("filenode"));
  file->addFiles(std::move(files));
  return file;
}

TEST_F(InputGraphTest, Empty) {
  expectEnd();
}

TEST_F(InputGraphTest, File) {
  _graph->addInputElement(createFile("file1"));
  EXPECT_EQ("file1", getNext());
  expectEnd();
}
