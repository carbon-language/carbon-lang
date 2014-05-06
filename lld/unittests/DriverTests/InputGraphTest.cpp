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

class MyLinkingContext : public LinkingContext {
public:
  Writer &writer() const override { llvm_unreachable("no writer!"); }

  bool validateImpl(raw_ostream &) override { return true; }
};

class MyFileNode : public SimpleFileNode {
public:
  MyFileNode(StringRef path) : SimpleFileNode(path) {}

  void resetNextIndex() override { FileNode::resetNextIndex(); }
};

class MyExpandFileNode : public SimpleFileNode {
public:
  MyExpandFileNode(StringRef path) : SimpleFileNode(path) {}

  /// \brief How do we want to expand the current node?
  bool shouldExpand() const override { return true; }

  /// \brief Get the elements that we want to expand with.
  range<InputGraph::InputElementIterT> expandElements() override {
    return make_range(_expandElements.begin(), _expandElements.end());
  }

  /// Process the input Elemenet
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
  }

  InputGraph &getInputGraph() { return _ctx.getInputGraph(); }
  int inputFileCount() { return _ctx.getInputGraph().size(); }

protected:
  MyLinkingContext _ctx;
};

} // end anonymous namespace

static std::unique_ptr<MyFileNode> createFile1(StringRef name) {
  std::vector<std::unique_ptr<File>> files;
  files.push_back(std::unique_ptr<SimpleFile>(new SimpleFile(name)));
  std::unique_ptr<MyFileNode> file(new MyFileNode("filenode"));
  file->addFiles(std::move(files));
  return file;
}

static std::unique_ptr<MyFileNode> createFile2(StringRef name1, StringRef name2) {
  std::vector<std::unique_ptr<File>> files;
  files.push_back(std::unique_ptr<SimpleFile>(new SimpleFile(name1)));
  files.push_back(std::unique_ptr<SimpleFile>(new SimpleFile(name2)));
  std::unique_ptr<MyFileNode> file(new MyFileNode("filenode"));
  file->addFiles(std::move(files));
  return file;
}

TEST_F(InputGraphTest, Basic) {
  EXPECT_EQ(0, inputFileCount());
  ErrorOr<InputElement *> nextElement = getInputGraph().getNextInputElement();
  EXPECT_EQ(InputGraphError::no_more_elements, nextElement.getError());
}

TEST_F(InputGraphTest, AddAFile) {
  std::unique_ptr<MyFileNode> myfile(new MyFileNode("file1"));
  getInputGraph().addInputElement(std::move(myfile));
  EXPECT_EQ(1, inputFileCount());
  ErrorOr<InputElement *> nextElement = getInputGraph().getNextInputElement();
  EXPECT_NE(InputGraphError::no_more_elements, nextElement.getError());
  EXPECT_EQ(InputElement::Kind::File, (*nextElement)->kind());
  nextElement = getInputGraph().getNextInputElement();
  EXPECT_EQ(InputGraphError::no_more_elements, nextElement.getError());
}

TEST_F(InputGraphTest, AddAFileWithLLDFiles) {
  _ctx.getInputGraph().addInputElement(createFile2("objfile1", "objfile2"));
  EXPECT_EQ(1, inputFileCount());
  ErrorOr<InputElement *> nextElement = getInputGraph().getNextInputElement();
  EXPECT_NE(InputGraphError::no_more_elements, nextElement.getError());
  EXPECT_EQ(InputElement::Kind::File, (*nextElement)->kind());
  FileNode *fileNode = dyn_cast<FileNode>(*nextElement);

  ErrorOr<File &> objfile = fileNode->getNextFile();
  EXPECT_NE(InputGraphError::no_more_files, objfile.getError());
  EXPECT_EQ("objfile1", (*objfile).path());

  objfile = fileNode->getNextFile();
  EXPECT_NE(InputGraphError::no_more_files, objfile.getError());
  EXPECT_EQ("objfile2", (*objfile).path());

  objfile = fileNode->getNextFile();
  EXPECT_EQ(InputGraphError::no_more_files, objfile.getError());

  fileNode->resetNextIndex();

  objfile = fileNode->getNextFile();
  EXPECT_NE(InputGraphError::no_more_files, objfile.getError());
  EXPECT_EQ("objfile1", (*objfile).path());

  nextElement = getInputGraph().getNextInputElement();
  EXPECT_EQ(InputGraphError::no_more_elements, nextElement.getError());
}

TEST_F(InputGraphTest, AddNodeWithFilesAndGroup) {
  _ctx.getInputGraph().addInputElement(createFile2("objfile1", "objfile2"));

  // Create a group node with two elements
  // an file node which looks like an archive and
  // two file nodes
  std::unique_ptr<Group> mygroup(new Group());
  mygroup->addFile(createFile2("objfile_1", "objfile_2"));
  mygroup->addFile(createFile1("group_objfile1"));
  mygroup->addFile(createFile1("group_objfile2"));
  getInputGraph().addInputElement(std::move(mygroup));

  EXPECT_EQ(2, inputFileCount());

  ErrorOr<InputElement *> nextElement = getInputGraph().getNextInputElement();
  EXPECT_NE(InputGraphError::no_more_elements, nextElement.getError());
  EXPECT_EQ(InputElement::Kind::File, (*nextElement)->kind());
  FileNode *fileNode = dyn_cast<FileNode>(*nextElement);

  ErrorOr<File &> objfile = fileNode->getNextFile();
  EXPECT_NE(InputGraphError::no_more_files, objfile.getError());
  EXPECT_EQ("objfile1", (*objfile).path());

  objfile = fileNode->getNextFile();
  EXPECT_NE(InputGraphError::no_more_files, objfile.getError());
  EXPECT_EQ("objfile2", (*objfile).path());

  objfile = fileNode->getNextFile();
  EXPECT_EQ(InputGraphError::no_more_files, objfile.getError());

  nextElement = getInputGraph().getNextInputElement();
  Group *group = dyn_cast<Group>(*nextElement);
  assert(group);

  objfile = group->getNextFile();
  EXPECT_NE(InputGraphError::no_more_files, objfile.getError());
  EXPECT_EQ("objfile_1", (*objfile).path());

  objfile = group->getNextFile();
  EXPECT_NE(InputGraphError::no_more_files, objfile.getError());
  EXPECT_EQ("objfile_2", (*objfile).path());

  objfile = group->getNextFile();
  EXPECT_NE(InputGraphError::no_more_files, objfile.getError());
  EXPECT_EQ("group_objfile1", (*objfile).path());

  objfile = group->getNextFile();
  EXPECT_NE(InputGraphError::no_more_files, objfile.getError());
  EXPECT_EQ("group_objfile2", (*objfile).path());

  nextElement = getInputGraph().getNextInputElement();
  EXPECT_EQ(InputGraphError::no_more_elements, nextElement.getError());
}

// Iterate through the group
TEST_F(InputGraphTest, AddNodeWithGroupIteration) {
  getInputGraph().addInputElement(createFile2("objfile1", "objfile2"));

  // Create a group node with two elements
  // an file node which looks like an archive and
  // two file nodes
  std::unique_ptr<Group> mygroup(new Group());
  mygroup->addFile(createFile2("objfile_1", "objfile_2"));
  mygroup->addFile(createFile1("group_objfile1"));
  mygroup->addFile(createFile1("group_objfile2"));
  getInputGraph().addInputElement(std::move(mygroup));

  EXPECT_EQ(2, inputFileCount());

  ErrorOr<InputElement *> nextElement = getInputGraph().getNextInputElement();
  EXPECT_NE(InputGraphError::no_more_elements, nextElement.getError());
  EXPECT_EQ(InputElement::Kind::File, (*nextElement)->kind());
  FileNode *fileNode = dyn_cast<FileNode>(*nextElement);

  ErrorOr<File &> objfile = fileNode->getNextFile();
  EXPECT_NE(InputGraphError::no_more_files, objfile.getError());
  EXPECT_EQ("objfile1", (*objfile).path());

  objfile = fileNode->getNextFile();
  EXPECT_NE(InputGraphError::no_more_files, objfile.getError());
  EXPECT_EQ("objfile2", (*objfile).path());

  objfile = fileNode->getNextFile();
  EXPECT_EQ(InputGraphError::no_more_files, objfile.getError());

  nextElement = getInputGraph().getNextInputElement();
  Group *group = dyn_cast<Group>(*nextElement);
  assert(group);

  objfile = group->getNextFile();
  EXPECT_NE(InputGraphError::no_more_files, objfile.getError());
  EXPECT_EQ("objfile_1", (*objfile).path());

  objfile = group->getNextFile();
  EXPECT_NE(InputGraphError::no_more_files, objfile.getError());
  EXPECT_EQ("objfile_2", (*objfile).path());

  objfile = group->getNextFile();
  EXPECT_NE(InputGraphError::no_more_files, objfile.getError());
  EXPECT_EQ("group_objfile1", (*objfile).path());

  objfile = group->getNextFile();
  EXPECT_NE(InputGraphError::no_more_files, objfile.getError());
  EXPECT_EQ("group_objfile2", (*objfile).path());

  group->notifyProgress();

  objfile = group->getNextFile();
  EXPECT_NE(InputGraphError::no_more_files, objfile.getError());
  EXPECT_EQ("objfile_1", (*objfile).path());

  objfile = group->getNextFile();
  EXPECT_NE(InputGraphError::no_more_files, objfile.getError());
  EXPECT_EQ("objfile_2", (*objfile).path());

  objfile = group->getNextFile();
  EXPECT_NE(InputGraphError::no_more_files, objfile.getError());
  EXPECT_EQ("group_objfile1", (*objfile).path());

  objfile = group->getNextFile();
  EXPECT_NE(InputGraphError::no_more_files, objfile.getError());
  EXPECT_EQ("group_objfile2", (*objfile).path());
}

// Node expansion tests.
TEST_F(InputGraphTest, ExpandAndReplaceInputGraphNode) {
  std::vector<std::unique_ptr<File>> objfiles;
  getInputGraph().addInputElement(createFile2("objfile1", "objfile2"));

  std::unique_ptr<MyExpandFileNode> expandFile(
      new MyExpandFileNode("expand_node"));
  expandFile->addElement(createFile1("objfile3"));
  expandFile->addElement(createFile1("objfile4"));
  getInputGraph().addInputElement(std::move(expandFile));

  // Add an extra obj after the expand node
  getInputGraph().addInputElement(createFile2("objfile5", "objfile6"));
  getInputGraph().normalize();

  ErrorOr<InputElement *> nextElement = getInputGraph().getNextInputElement();
  EXPECT_NE(InputGraphError::no_more_elements, nextElement.getError());
  EXPECT_EQ(InputElement::Kind::File, (*nextElement)->kind());
  FileNode *fileNode = dyn_cast<FileNode>(*nextElement);

  nextElement = getInputGraph().getNextInputElement();
  EXPECT_NE(InputGraphError::no_more_elements, nextElement.getError());
  EXPECT_EQ(InputElement::Kind::File, (*nextElement)->kind());
  fileNode = dyn_cast<FileNode>(*nextElement);

  nextElement = getInputGraph().getNextInputElement();
  EXPECT_NE(InputGraphError::no_more_elements, nextElement.getError());
  EXPECT_EQ(InputElement::Kind::File, (*nextElement)->kind());
  fileNode = dyn_cast<FileNode>(*nextElement);

  nextElement = getInputGraph().getNextInputElement();
  EXPECT_NE(InputGraphError::no_more_elements, nextElement.getError());
  EXPECT_EQ(InputElement::Kind::File, (*nextElement)->kind());
  fileNode = dyn_cast<FileNode>(*nextElement);

  nextElement = getInputGraph().getNextInputElement();
  EXPECT_EQ(InputGraphError::no_more_elements, nextElement.getError());
}
