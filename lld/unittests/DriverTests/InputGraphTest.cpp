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

#include <stdarg.h>

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
  MyFileNode(StringRef path, int64_t ordinal) : SimpleFileNode(path, ordinal) {}

  void resetNextIndex() override { FileNode::resetNextIndex(); }
};

class MyExpandFileNode : public SimpleFileNode {
public:
  MyExpandFileNode(StringRef path, int64_t ordinal)
      : SimpleFileNode(path, ordinal) {}

  /// \brief How do we want to expand the current node ?
  bool shouldExpand() const override { return true; }

  /// \brief Get the elements that we want to expand with.
  range<InputGraph::InputElementIterT> expandElements() override {
    return make_range(_expandElements.begin(), _expandElements.end());
  }

  /// Process the input Elemenet
  virtual bool addElement(std::unique_ptr<InputElement> element) {
    _expandElements.push_back(std::move(element));
    return true;
  }

private:
  InputGraph::InputElementVectorT _expandElements;
};

class InputGraphTest : public testing::Test {
public:
  InputGraphTest() {
    _inputGraph.reset(new InputGraph());
    _context.setInputGraph(std::move(_inputGraph));
  }

  virtual LinkingContext &linkingContext() { return _context; }

  InputElement &inputElement(unsigned index) {
    return linkingContext().inputGraph()[index];
  }

  virtual InputGraph &inputGraph() { return linkingContext().inputGraph(); }

  int inputFileCount() { return linkingContext().inputGraph().size(); }

protected:
  MyLinkingContext _context;
  std::unique_ptr<InputGraph> _inputGraph;
};

} // end anonymous namespace

TEST_F(InputGraphTest, Basic) {
  EXPECT_EQ(0, inputFileCount());
  ErrorOr<InputElement *> nextElement = inputGraph().getNextInputElement();
  EXPECT_EQ(InputGraphError::no_more_elements, nextElement.getError());
}

TEST_F(InputGraphTest, AddAFile) {
  std::unique_ptr<MyFileNode> myfile(new MyFileNode("file1", 0));
  EXPECT_EQ(true, inputGraph().addInputElement(std::move(myfile)));
  EXPECT_EQ(1, inputFileCount());
  ErrorOr<InputElement *> nextElement = inputGraph().getNextInputElement();
  EXPECT_NE(InputGraphError::no_more_elements, nextElement.getError());
  EXPECT_EQ(InputElement::Kind::File, (*nextElement)->kind());
  FileNode *fileNode = dyn_cast<FileNode>(*nextElement);
  EXPECT_EQ("file1", fileNode->getUserPath());
  nextElement = inputGraph().getNextInputElement();
  EXPECT_EQ(InputGraphError::no_more_elements, nextElement.getError());
}

TEST_F(InputGraphTest, AddAFileWithLLDFiles) {
  std::unique_ptr<MyFileNode> myfile(new MyFileNode("multi_files", 0));
  std::vector<std::unique_ptr<File> > objfiles;
  std::unique_ptr<SimpleFile> obj1(new SimpleFile("objfile1"));
  std::unique_ptr<SimpleFile> obj2(new SimpleFile("objfile2"));
  objfiles.push_back(std::move(obj1));
  objfiles.push_back(std::move(obj2));
  myfile->addFiles(std::move(objfiles));
  EXPECT_EQ(true, inputGraph().addInputElement(std::move(myfile)));
  EXPECT_EQ(1, inputFileCount());
  ErrorOr<InputElement *> nextElement = inputGraph().getNextInputElement();
  EXPECT_NE(InputGraphError::no_more_elements, nextElement.getError());
  EXPECT_EQ(InputElement::Kind::File, (*nextElement)->kind());
  FileNode *fileNode = dyn_cast<FileNode>(*nextElement);

  EXPECT_EQ("multi_files", fileNode->getUserPath());

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

  nextElement = inputGraph().getNextInputElement();
  EXPECT_EQ(InputGraphError::no_more_elements, nextElement.getError());
}

TEST_F(InputGraphTest, AddNodeWithFilesAndGroup) {
  std::unique_ptr<MyFileNode> myfile(new MyFileNode("multi_files1", 0));
  std::vector<std::unique_ptr<File> > objfiles;
  std::unique_ptr<SimpleFile> obj1(new SimpleFile("objfile1"));
  std::unique_ptr<SimpleFile> obj2(new SimpleFile("objfile2"));
  objfiles.push_back(std::move(obj1));
  objfiles.push_back(std::move(obj2));
  myfile->addFiles(std::move(objfiles));
  EXPECT_EQ(true, inputGraph().addInputElement(std::move(myfile)));

  // Create a group node with two elements
  // an file node which looks like an archive and
  // two file nodes
  std::unique_ptr<Group> mygroup(new Group(1));
  std::unique_ptr<MyFileNode> myarchive(new MyFileNode("archive_file", 2));
  std::vector<std::unique_ptr<File> > objfiles_group;
  std::unique_ptr<SimpleFile> obj_1(new SimpleFile("objfile_1"));
  std::unique_ptr<SimpleFile> obj_2(new SimpleFile("objfile_2"));
  objfiles_group.push_back(std::move(obj_1));
  objfiles_group.push_back(std::move(obj_2));
  myarchive->addFiles(std::move(objfiles_group));
  EXPECT_EQ(true, mygroup->addFile(std::move(myarchive)));

  std::unique_ptr<MyFileNode> mygroupobjfile_1(
      new MyFileNode("group_objfile1", 3));
  std::vector<std::unique_ptr<File> > objfiles_group1;
  std::unique_ptr<SimpleFile> mygroupobj1(
      new SimpleFile("group_objfile1"));
  objfiles_group1.push_back(std::move(mygroupobj1));
  mygroupobjfile_1->addFiles(std::move(objfiles_group1));
  EXPECT_EQ(true, mygroup->addFile(std::move(mygroupobjfile_1)));

  std::unique_ptr<MyFileNode> mygroupobjfile_2(
      new MyFileNode("group_objfile2", 4));
  std::vector<std::unique_ptr<File> > objfiles_group2;
  std::unique_ptr<SimpleFile> mygroupobj2(
      new SimpleFile("group_objfile2"));
  objfiles_group2.push_back(std::move(mygroupobj2));
  mygroupobjfile_2->addFiles(std::move(objfiles_group2));
  EXPECT_EQ(true, mygroup->addFile(std::move(mygroupobjfile_2)));

  // Add the group to the InputGraph.
  EXPECT_EQ(true, inputGraph().addInputElement(std::move(mygroup)));

  EXPECT_EQ(2, inputFileCount());

  ErrorOr<InputElement *> nextElement = inputGraph().getNextInputElement();
  EXPECT_NE(InputGraphError::no_more_elements, nextElement.getError());
  EXPECT_EQ(InputElement::Kind::File, (*nextElement)->kind());
  FileNode *fileNode = dyn_cast<FileNode>(*nextElement);

  EXPECT_EQ("multi_files1", fileNode->getUserPath());

  ErrorOr<File &> objfile = fileNode->getNextFile();
  EXPECT_NE(InputGraphError::no_more_files, objfile.getError());
  EXPECT_EQ("objfile1", (*objfile).path());

  objfile = fileNode->getNextFile();
  EXPECT_NE(InputGraphError::no_more_files, objfile.getError());
  EXPECT_EQ("objfile2", (*objfile).path());

  objfile = fileNode->getNextFile();
  EXPECT_EQ(InputGraphError::no_more_files, objfile.getError());

  nextElement = inputGraph().getNextInputElement();
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

  nextElement = inputGraph().getNextInputElement();
  EXPECT_EQ(InputGraphError::no_more_elements, nextElement.getError());
}

// Iterate through the group
TEST_F(InputGraphTest, AddNodeWithGroupIteration) {
  std::unique_ptr<MyFileNode> myfile(new MyFileNode("multi_files1", 0));
  std::vector<std::unique_ptr<File> > objfiles;
  std::unique_ptr<SimpleFile> obj1(new SimpleFile("objfile1"));
  std::unique_ptr<SimpleFile> obj2(new SimpleFile("objfile2"));
  objfiles.push_back(std::move(obj1));
  objfiles.push_back(std::move(obj2));
  myfile->addFiles(std::move(objfiles));
  EXPECT_EQ(true, inputGraph().addInputElement(std::move(myfile)));

  // Create a group node with two elements
  // an file node which looks like an archive and
  // two file nodes
  std::unique_ptr<Group> mygroup(new Group(1));
  std::unique_ptr<MyFileNode> myarchive(new MyFileNode("archive_file", 2));
  std::vector<std::unique_ptr<File> > objfiles_group;
  std::unique_ptr<SimpleFile> obj_1(new SimpleFile("objfile_1"));
  std::unique_ptr<SimpleFile> obj_2(new SimpleFile("objfile_2"));
  objfiles_group.push_back(std::move(obj_1));
  objfiles_group.push_back(std::move(obj_2));
  myarchive->addFiles(std::move(objfiles_group));
  EXPECT_EQ(true, mygroup->addFile(std::move(myarchive)));

  std::unique_ptr<MyFileNode> mygroupobjfile_1(
      new MyFileNode("group_objfile1", 3));
  std::vector<std::unique_ptr<File> > objfiles_group1;
  std::unique_ptr<SimpleFile> mygroupobj1(
      new SimpleFile("group_objfile1"));
  objfiles_group1.push_back(std::move(mygroupobj1));
  mygroupobjfile_1->addFiles(std::move(objfiles_group1));
  EXPECT_EQ(true, mygroup->addFile(std::move(mygroupobjfile_1)));

  std::unique_ptr<MyFileNode> mygroupobjfile_2(
      new MyFileNode("group_objfile2", 4));
  std::vector<std::unique_ptr<File> > objfiles_group2;
  std::unique_ptr<SimpleFile> mygroupobj2(
      new SimpleFile("group_objfile2"));
  objfiles_group2.push_back(std::move(mygroupobj2));
  mygroupobjfile_2->addFiles(std::move(objfiles_group2));
  EXPECT_EQ(true, mygroup->addFile(std::move(mygroupobjfile_2)));

  // Add the group to the InputGraph.
  EXPECT_EQ(true, inputGraph().addInputElement(std::move(mygroup)));

  EXPECT_EQ(2, inputFileCount());

  ErrorOr<InputElement *> nextElement = inputGraph().getNextInputElement();
  EXPECT_NE(InputGraphError::no_more_elements, nextElement.getError());
  EXPECT_EQ(InputElement::Kind::File, (*nextElement)->kind());
  FileNode *fileNode = dyn_cast<FileNode>(*nextElement);

  EXPECT_EQ("multi_files1", fileNode->getUserPath());

  ErrorOr<File &> objfile = fileNode->getNextFile();
  EXPECT_NE(InputGraphError::no_more_files, objfile.getError());
  EXPECT_EQ("objfile1", (*objfile).path());

  objfile = fileNode->getNextFile();
  EXPECT_NE(InputGraphError::no_more_files, objfile.getError());
  EXPECT_EQ("objfile2", (*objfile).path());

  objfile = fileNode->getNextFile();
  EXPECT_EQ(InputGraphError::no_more_files, objfile.getError());

  nextElement = inputGraph().getNextInputElement();
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
  std::unique_ptr<MyFileNode> myfile(new MyFileNode("multi_files1", 0));
  std::vector<std::unique_ptr<File> > objfiles;
  std::unique_ptr<SimpleFile> obj1(new SimpleFile("objfile1"));
  std::unique_ptr<SimpleFile> obj2(new SimpleFile("objfile2"));
  objfiles.push_back(std::move(obj1));
  objfiles.push_back(std::move(obj2));
  myfile->addFiles(std::move(objfiles));
  EXPECT_EQ(true, inputGraph().addInputElement(std::move(myfile)));
  objfiles.clear();

  std::unique_ptr<MyExpandFileNode> expandFile(new MyExpandFileNode(
      "expand_node", 1));

  std::unique_ptr<MyFileNode> filenode1(new MyFileNode("expand_file1", 2));
  std::unique_ptr<SimpleFile> obj3(new SimpleFile("objfile3"));
  objfiles.push_back(std::move(obj3));
  filenode1->addFiles(std::move(objfiles));
  expandFile->addElement(std::move(filenode1));
  objfiles.clear();

  std::unique_ptr<MyFileNode> filenode2(new MyFileNode("expand_file2", 3));
  std::unique_ptr<SimpleFile> obj4(new SimpleFile("objfile4"));
  objfiles.push_back(std::move(obj4));
  filenode2->addFiles(std::move(objfiles));
  expandFile->addElement(std::move(filenode2));
  objfiles.clear();

  // Add expand file to InputGraph
  EXPECT_EQ(true, inputGraph().addInputElement(std::move(expandFile)));

  std::unique_ptr<MyFileNode> filenode3(new MyFileNode("obj_after_expand", 4));
  std::unique_ptr<SimpleFile> obj5(new SimpleFile("objfile5"));
  std::unique_ptr<SimpleFile> obj6(new SimpleFile("objfile6"));
  objfiles.push_back(std::move(obj5));
  objfiles.push_back(std::move(obj6));
  filenode3->addFiles(std::move(objfiles));

  // Add an extra obj after the expand node
  EXPECT_EQ(true, inputGraph().addInputElement(std::move(filenode3)));

  inputGraph().normalize();

  ErrorOr<InputElement *> nextElement = inputGraph().getNextInputElement();
  EXPECT_NE(InputGraphError::no_more_elements, nextElement.getError());
  EXPECT_EQ(InputElement::Kind::File, (*nextElement)->kind());
  FileNode *fileNode = dyn_cast<FileNode>(*nextElement);
  EXPECT_EQ("multi_files1", (*fileNode).getUserPath());

  nextElement = inputGraph().getNextInputElement();
  EXPECT_NE(InputGraphError::no_more_elements, nextElement.getError());
  EXPECT_EQ(InputElement::Kind::File, (*nextElement)->kind());
  fileNode = dyn_cast<FileNode>(*nextElement);
  EXPECT_EQ("expand_file1", (*fileNode).getUserPath());

  nextElement = inputGraph().getNextInputElement();
  EXPECT_NE(InputGraphError::no_more_elements, nextElement.getError());
  EXPECT_EQ(InputElement::Kind::File, (*nextElement)->kind());
  fileNode = dyn_cast<FileNode>(*nextElement);
  EXPECT_EQ("expand_file2", (*fileNode).getUserPath());

  nextElement = inputGraph().getNextInputElement();
  EXPECT_NE(InputGraphError::no_more_elements, nextElement.getError());
  EXPECT_EQ(InputElement::Kind::File, (*nextElement)->kind());
  fileNode = dyn_cast<FileNode>(*nextElement);
  EXPECT_EQ("obj_after_expand", (*fileNode).getUserPath());

  nextElement = inputGraph().getNextInputElement();
  EXPECT_EQ(InputGraphError::no_more_elements, nextElement.getError());
}
