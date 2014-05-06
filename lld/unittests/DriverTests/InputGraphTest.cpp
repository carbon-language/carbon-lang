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
#include "lld/ReaderWriter/Simple.h"

using namespace lld;

namespace {

class TestLinkingContext : public LinkingContext {
public:
  Writer &writer() const override { llvm_unreachable("no writer!"); }
  bool validateImpl(raw_ostream &) override { return true; }
};

class TestFileNode : public SimpleFileNode {
public:
  TestFileNode(StringRef path) : SimpleFileNode(path) {}
  void resetNextIndex() override { FileNode::resetNextIndex(); }
};

class TestExpandFileNode : public SimpleFileNode {
public:
  TestExpandFileNode(StringRef path) : SimpleFileNode(path) {}

  /// Returns true as we want to expand this file
  bool shouldExpand() const override { return true; }

  /// Returns the elements replacing this node
  range<InputGraph::InputElementIterT> expandElements() override {
    return make_range(_expandElements.begin(), _expandElements.end());
  }

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
    ErrorOr<File &> file = _graph->getNextFile();
    EXPECT_TRUE(!file.getError());
    return file.get().path();
  }

  void expectEnd() {
    ErrorOr<File &> file = _graph->getNextFile();
    EXPECT_EQ(InputGraphError::no_more_files, file.getError());
  }

protected:
  TestLinkingContext _ctx;
  InputGraph *_graph;
};

} // end anonymous namespace

static std::unique_ptr<TestFileNode> createFile1(StringRef name) {
  std::vector<std::unique_ptr<File>> files;
  files.push_back(std::unique_ptr<SimpleFile>(new SimpleFile(name)));
  std::unique_ptr<TestFileNode> file(new TestFileNode("filenode"));
  file->addFiles(std::move(files));
  return file;
}

static std::unique_ptr<TestFileNode> createFile2(StringRef name1,
                                                 StringRef name2) {
  std::vector<std::unique_ptr<File>> files;
  files.push_back(std::unique_ptr<SimpleFile>(new SimpleFile(name1)));
  files.push_back(std::unique_ptr<SimpleFile>(new SimpleFile(name2)));
  std::unique_ptr<TestFileNode> file(new TestFileNode("filenode"));
  file->addFiles(std::move(files));
  return file;
}

TEST_F(InputGraphTest, Empty) {
  expectEnd();
}

TEST_F(InputGraphTest, File) {
  _graph->addInputElement(createFile1("file1"));
  EXPECT_EQ("file1", getNext());
  expectEnd();
}

TEST_F(InputGraphTest, Files) {
  _graph->addInputElement(createFile2("file1", "file2"));
  EXPECT_EQ("file1", getNext());
  EXPECT_EQ("file2", getNext());
  expectEnd();
}

TEST_F(InputGraphTest, Group) {
  _graph->addInputElement(createFile2("file1", "file2"));

  std::unique_ptr<Group> group(new Group());
  group->addFile(createFile2("file3", "file4"));
  group->addFile(createFile1("file5"));
  group->addFile(createFile1("file6"));
  _graph->addInputElement(std::move(group));

  EXPECT_EQ("file1", getNext());
  EXPECT_EQ("file2", getNext());
  EXPECT_EQ("file3", getNext());
  EXPECT_EQ("file4", getNext());
  EXPECT_EQ("file5", getNext());
  EXPECT_EQ("file6", getNext());
  expectEnd();
}

// Iterate through the group
TEST_F(InputGraphTest, GroupIteration) {
  _graph->addInputElement(createFile2("file1", "file2"));

  std::unique_ptr<Group> group(new Group());
  group->addFile(createFile2("file3", "file4"));
  group->addFile(createFile1("file5"));
  group->addFile(createFile1("file6"));
  _graph->addInputElement(std::move(group));

  EXPECT_EQ("file1", getNext());
  EXPECT_EQ("file2", getNext());
  EXPECT_EQ("file3", getNext());
  EXPECT_EQ("file4", getNext());
  EXPECT_EQ("file5", getNext());
  EXPECT_EQ("file6", getNext());
  _graph->notifyProgress();
  EXPECT_EQ("file3", getNext());
  EXPECT_EQ("file4", getNext());
  EXPECT_EQ("file5", getNext());
  EXPECT_EQ("file6", getNext());
  expectEnd();
}

// Node expansion tests
TEST_F(InputGraphTest, Normalize) {
  std::vector<std::unique_ptr<File>> objfiles;
  _graph->addInputElement(createFile2("file1", "file2"));

  std::unique_ptr<TestExpandFileNode> expandFile(
      new TestExpandFileNode("node"));
  expandFile->addElement(createFile1("file3"));
  expandFile->addElement(createFile1("file4"));
  _graph->addInputElement(std::move(expandFile));
  _graph->addInputElement(createFile2("file5", "file6"));
  _graph->normalize();

  EXPECT_EQ("file1", getNext());
  EXPECT_EQ("file2", getNext());
  EXPECT_EQ("file3", getNext());
  EXPECT_EQ("file4", getNext());
  EXPECT_EQ("file5", getNext());
  EXPECT_EQ("file6", getNext());
  _graph->notifyProgress();
}
