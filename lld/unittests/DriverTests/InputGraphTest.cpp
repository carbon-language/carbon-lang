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
  virtual Reader &getDefaultReader() const { return *_yamlReader; }

  virtual ErrorOr<Reference::Kind> relocKindFromString(StringRef str) const {
    return make_error_code(YamlReaderError::illegal_value);
  }

  virtual ErrorOr<std::string> stringFromRelocKind(Reference::Kind k) const {
    return make_error_code(YamlReaderError::illegal_value);
  }

  virtual Writer &writer() const { llvm_unreachable("no writer!"); }

  virtual bool validateImpl(raw_ostream &) { return true; }
};

class MyInputGraph : public InputGraph {
public:
  MyInputGraph() : InputGraph() {};
};

class MyFileNode : public FileNode {
public:
  MyFileNode(StringRef path, int64_t ordinal) : FileNode(path, ordinal) {}

  bool validate() { return true; }

  bool dump(raw_ostream &) { return true; }

  virtual error_code parse(const LinkingContext &, raw_ostream &) {
    return error_code::success();
  }

  virtual ErrorOr<File &> getNextFile() {
    if (_nextFileIndex == _files.size())
      return make_error_code(InputGraphError::no_more_files);
    return *_files[_nextFileIndex++];
  }
};

class MyGroupNode : public Group {
public:
  MyGroupNode(int64_t ordinal) : Group(ordinal) {}

  bool validate() { return true; }

  bool dump(raw_ostream &) { return true; }

  virtual error_code parse(const LinkingContext &, raw_ostream &) {
    return error_code::success();
  }
};

class MyExpandFileNode : public FileNode {
public:
  MyExpandFileNode(StringRef path, int64_t ordinal, ExpandType expandType)
      : FileNode(path, ordinal), _expandType(expandType) {}

  bool validate() { return true; }

  bool dump(raw_ostream &) { return true; }

  virtual error_code parse(const LinkingContext &, raw_ostream &) {
    return error_code::success();
  }

  virtual ErrorOr<File &> getNextFile() {
    if (_nextFileIndex == _files.size())
      return make_error_code(InputGraphError::no_more_files);
    return *_files[_nextFileIndex++];
  }

  /// \brief How do we want to expand the current node ?
  virtual ExpandType expandType() const { return _expandType; }

  /// \brief Get the elements that we want to expand with.
  virtual range<InputGraph::InputElementIterT> expandElements() {
    return make_range(_expandElements.begin(), _expandElements.end());
  }

  /// Process the input Elemenet
  virtual bool addElement(std::unique_ptr<InputElement> element) {
    _expandElements.push_back(std::move(element));
    return true;
  }

private:
  InputGraph::InputElementVectorT _expandElements;
  ExpandType _expandType;
};

class MyObjFile : public SimpleFile {
public:
  MyObjFile(LinkingContext &context, StringRef path)
      : SimpleFile(context, path) {}
};

class InputGraphTest : public testing::Test {
public:
  InputGraphTest() {
    _inputGraph.reset(new MyInputGraph());
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

TEST_F(InputGraphTest, Basic) {
  EXPECT_EQ(0, inputFileCount());
  ErrorOr<InputElement *> nextElement = inputGraph().getNextInputElement();
  EXPECT_EQ(InputGraphError::no_more_elements, error_code(nextElement));
}

TEST_F(InputGraphTest, AddAFile) {
  std::unique_ptr<MyFileNode> myfile(new MyFileNode("file1", 0));
  EXPECT_EQ(true, inputGraph().addInputElement(std::move(myfile)));
  EXPECT_EQ(1, inputFileCount());
  ErrorOr<InputElement *> nextElement = inputGraph().getNextInputElement();
  EXPECT_NE(InputGraphError::no_more_elements, error_code(nextElement));
  EXPECT_EQ(InputElement::Kind::File, (*nextElement)->kind());
  FileNode *fileNode = llvm::dyn_cast<FileNode>(*nextElement);
  StringRef path = fileNode->getUserPath();
  EXPECT_EQ(0, path.compare("file1"));
  nextElement = inputGraph().getNextInputElement();
  EXPECT_EQ(InputGraphError::no_more_elements, error_code(nextElement));
}

TEST_F(InputGraphTest, AddAFileWithLLDFiles) {
  std::unique_ptr<MyFileNode> myfile(new MyFileNode("multi_files", 0));
  std::vector<std::unique_ptr<File> > objfiles;
  std::unique_ptr<MyObjFile> obj1(new MyObjFile(_context, "objfile1"));
  std::unique_ptr<MyObjFile> obj2(new MyObjFile(_context, "objfile2"));
  objfiles.push_back(std::move(obj1));
  objfiles.push_back(std::move(obj2));
  myfile->addFiles(std::move(objfiles));
  EXPECT_EQ(true, inputGraph().addInputElement(std::move(myfile)));
  EXPECT_EQ(1, inputFileCount());
  ErrorOr<InputElement *> nextElement = inputGraph().getNextInputElement();
  EXPECT_NE(InputGraphError::no_more_elements, error_code(nextElement));
  EXPECT_EQ(InputElement::Kind::File, (*nextElement)->kind());
  FileNode *fileNode = llvm::dyn_cast<FileNode>(*nextElement);

  StringRef path = fileNode->getUserPath();
  EXPECT_EQ(0, path.compare("multi_files"));

  ErrorOr<File &> objfile = fileNode->getNextFile();
  EXPECT_NE(InputGraphError::no_more_files, error_code(objfile));
  EXPECT_EQ("objfile1", (*objfile).path());

  objfile = fileNode->getNextFile();
  EXPECT_NE(InputGraphError::no_more_files, error_code(objfile));
  EXPECT_EQ("objfile2", (*objfile).path());

  objfile = fileNode->getNextFile();
  EXPECT_EQ(InputGraphError::no_more_files, error_code(objfile));

  fileNode->resetNextIndex();

  objfile = fileNode->getNextFile();
  EXPECT_NE(InputGraphError::no_more_files, error_code(objfile));
  EXPECT_EQ("objfile1", (*objfile).path());

  nextElement = inputGraph().getNextInputElement();
  EXPECT_EQ(InputGraphError::no_more_elements, error_code(nextElement));
}

TEST_F(InputGraphTest, AddNodeWithFilesAndGroup) {
  std::unique_ptr<MyFileNode> myfile(new MyFileNode("multi_files1", 0));
  std::vector<std::unique_ptr<File> > objfiles;
  std::unique_ptr<MyObjFile> obj1(new MyObjFile(_context, "objfile1"));
  std::unique_ptr<MyObjFile> obj2(new MyObjFile(_context, "objfile2"));
  objfiles.push_back(std::move(obj1));
  objfiles.push_back(std::move(obj2));
  myfile->addFiles(std::move(objfiles));
  EXPECT_EQ(true, inputGraph().addInputElement(std::move(myfile)));

  // Create a group node with two elements
  // an file node which looks like an archive and
  // two file nodes
  std::unique_ptr<MyGroupNode> mygroup(new MyGroupNode(1));
  std::unique_ptr<MyFileNode> myarchive(new MyFileNode("archive_file", 2));
  std::vector<std::unique_ptr<File> > objfiles_group;
  std::unique_ptr<MyObjFile> obj_1(new MyObjFile(_context, "objfile_1"));
  std::unique_ptr<MyObjFile> obj_2(new MyObjFile(_context, "objfile_2"));
  objfiles_group.push_back(std::move(obj_1));
  objfiles_group.push_back(std::move(obj_2));
  myarchive->addFiles(std::move(objfiles_group));
  EXPECT_EQ(true, mygroup->processInputElement(std::move(myarchive)));

  std::unique_ptr<MyFileNode> mygroupobjfile_1(
      new MyFileNode("group_objfile1", 3));
  std::vector<std::unique_ptr<File> > objfiles_group1;
  std::unique_ptr<MyObjFile> mygroupobj1(
      new MyObjFile(_context, "group_objfile1"));
  objfiles_group1.push_back(std::move(mygroupobj1));
  mygroupobjfile_1->addFiles(std::move(objfiles_group1));
  EXPECT_EQ(true, mygroup->processInputElement(std::move(mygroupobjfile_1)));

  std::unique_ptr<MyFileNode> mygroupobjfile_2(
      new MyFileNode("group_objfile2", 4));
  std::vector<std::unique_ptr<File> > objfiles_group2;
  std::unique_ptr<MyObjFile> mygroupobj2(
      new MyObjFile(_context, "group_objfile2"));
  objfiles_group2.push_back(std::move(mygroupobj2));
  mygroupobjfile_2->addFiles(std::move(objfiles_group2));
  EXPECT_EQ(true, mygroup->processInputElement(std::move(mygroupobjfile_2)));

  // Add the group to the InputGraph.
  EXPECT_EQ(true, inputGraph().addInputElement(std::move(mygroup)));

  EXPECT_EQ(2, inputFileCount());

  ErrorOr<InputElement *> nextElement = inputGraph().getNextInputElement();
  EXPECT_NE(InputGraphError::no_more_elements, error_code(nextElement));
  EXPECT_EQ(InputElement::Kind::File, (*nextElement)->kind());
  FileNode *fileNode = llvm::dyn_cast<FileNode>(*nextElement);

  StringRef path = fileNode->getUserPath();
  EXPECT_EQ(0, path.compare("multi_files1"));

  ErrorOr<File &> objfile = fileNode->getNextFile();
  EXPECT_NE(InputGraphError::no_more_files, error_code(objfile));
  EXPECT_EQ("objfile1", (*objfile).path());

  objfile = fileNode->getNextFile();
  EXPECT_NE(InputGraphError::no_more_files, error_code(objfile));
  EXPECT_EQ("objfile2", (*objfile).path());

  objfile = fileNode->getNextFile();
  EXPECT_EQ(InputGraphError::no_more_files, error_code(objfile));

  nextElement = inputGraph().getNextInputElement();
  EXPECT_EQ(InputElement::Kind::Control, (*nextElement)->kind());
  ControlNode *controlNode = llvm::dyn_cast<ControlNode>(*nextElement);

  EXPECT_EQ(ControlNode::ControlKind::Group, controlNode->controlKind());

  objfile = controlNode->getNextFile();
  EXPECT_NE(InputGraphError::no_more_files, error_code(objfile));
  EXPECT_EQ("objfile_1", (*objfile).path());

  objfile = controlNode->getNextFile();
  EXPECT_NE(InputGraphError::no_more_files, error_code(objfile));
  EXPECT_EQ("objfile_2", (*objfile).path());

  objfile = controlNode->getNextFile();
  EXPECT_NE(InputGraphError::no_more_files, error_code(objfile));
  EXPECT_EQ("group_objfile1", (*objfile).path());

  objfile = controlNode->getNextFile();
  EXPECT_NE(InputGraphError::no_more_files, error_code(objfile));
  EXPECT_EQ("group_objfile2", (*objfile).path());

  nextElement = inputGraph().getNextInputElement();
  EXPECT_EQ(InputGraphError::no_more_elements, error_code(nextElement));
}

// Iterate through the group
TEST_F(InputGraphTest, AddNodeWithGroupIteration) {
  std::unique_ptr<MyFileNode> myfile(new MyFileNode("multi_files1", 0));
  std::vector<std::unique_ptr<File> > objfiles;
  std::unique_ptr<MyObjFile> obj1(new MyObjFile(_context, "objfile1"));
  std::unique_ptr<MyObjFile> obj2(new MyObjFile(_context, "objfile2"));
  objfiles.push_back(std::move(obj1));
  objfiles.push_back(std::move(obj2));
  myfile->addFiles(std::move(objfiles));
  EXPECT_EQ(true, inputGraph().addInputElement(std::move(myfile)));

  // Create a group node with two elements
  // an file node which looks like an archive and
  // two file nodes
  std::unique_ptr<MyGroupNode> mygroup(new MyGroupNode(1));
  std::unique_ptr<MyFileNode> myarchive(new MyFileNode("archive_file", 2));
  std::vector<std::unique_ptr<File> > objfiles_group;
  std::unique_ptr<MyObjFile> obj_1(new MyObjFile(_context, "objfile_1"));
  std::unique_ptr<MyObjFile> obj_2(new MyObjFile(_context, "objfile_2"));
  objfiles_group.push_back(std::move(obj_1));
  objfiles_group.push_back(std::move(obj_2));
  myarchive->addFiles(std::move(objfiles_group));
  EXPECT_EQ(true, mygroup->processInputElement(std::move(myarchive)));

  std::unique_ptr<MyFileNode> mygroupobjfile_1(
      new MyFileNode("group_objfile1", 3));
  std::vector<std::unique_ptr<File> > objfiles_group1;
  std::unique_ptr<MyObjFile> mygroupobj1(
      new MyObjFile(_context, "group_objfile1"));
  objfiles_group1.push_back(std::move(mygroupobj1));
  mygroupobjfile_1->addFiles(std::move(objfiles_group1));
  EXPECT_EQ(true, mygroup->processInputElement(std::move(mygroupobjfile_1)));

  std::unique_ptr<MyFileNode> mygroupobjfile_2(
      new MyFileNode("group_objfile2", 4));
  std::vector<std::unique_ptr<File> > objfiles_group2;
  std::unique_ptr<MyObjFile> mygroupobj2(
      new MyObjFile(_context, "group_objfile2"));
  objfiles_group2.push_back(std::move(mygroupobj2));
  mygroupobjfile_2->addFiles(std::move(objfiles_group2));
  EXPECT_EQ(true, mygroup->processInputElement(std::move(mygroupobjfile_2)));

  // Add the group to the InputGraph.
  EXPECT_EQ(true, inputGraph().addInputElement(std::move(mygroup)));

  EXPECT_EQ(2, inputFileCount());

  ErrorOr<InputElement *> nextElement = inputGraph().getNextInputElement();
  EXPECT_NE(InputGraphError::no_more_elements, error_code(nextElement));
  EXPECT_EQ(InputElement::Kind::File, (*nextElement)->kind());
  FileNode *fileNode = llvm::dyn_cast<FileNode>(*nextElement);

  StringRef path = fileNode->getUserPath();
  EXPECT_EQ(0, path.compare("multi_files1"));

  ErrorOr<File &> objfile = fileNode->getNextFile();
  EXPECT_NE(InputGraphError::no_more_files, error_code(objfile));
  EXPECT_EQ("objfile1", (*objfile).path());

  objfile = fileNode->getNextFile();
  EXPECT_NE(InputGraphError::no_more_files, error_code(objfile));
  EXPECT_EQ("objfile2", (*objfile).path());

  objfile = fileNode->getNextFile();
  EXPECT_EQ(InputGraphError::no_more_files, error_code(objfile));

  nextElement = inputGraph().getNextInputElement();
  EXPECT_EQ(InputElement::Kind::Control, (*nextElement)->kind());
  ControlNode *controlNode = llvm::dyn_cast<ControlNode>(*nextElement);

  EXPECT_EQ(ControlNode::ControlKind::Group, controlNode->controlKind());

  objfile = controlNode->getNextFile();
  EXPECT_NE(InputGraphError::no_more_files, error_code(objfile));
  EXPECT_EQ("objfile_1", (*objfile).path());

  objfile = controlNode->getNextFile();
  EXPECT_NE(InputGraphError::no_more_files, error_code(objfile));
  EXPECT_EQ("objfile_2", (*objfile).path());

  objfile = controlNode->getNextFile();
  EXPECT_NE(InputGraphError::no_more_files, error_code(objfile));
  EXPECT_EQ("group_objfile1", (*objfile).path());

  objfile = controlNode->getNextFile();
  EXPECT_NE(InputGraphError::no_more_files, error_code(objfile));
  EXPECT_EQ("group_objfile2", (*objfile).path());

  controlNode->setResolveState(Resolver::StateNewDefinedAtoms);

  objfile = controlNode->getNextFile();
  EXPECT_NE(InputGraphError::no_more_files, error_code(objfile));
  EXPECT_EQ("objfile_1", (*objfile).path());

  objfile = controlNode->getNextFile();
  EXPECT_NE(InputGraphError::no_more_files, error_code(objfile));
  EXPECT_EQ("objfile_2", (*objfile).path());

  objfile = controlNode->getNextFile();
  EXPECT_NE(InputGraphError::no_more_files, error_code(objfile));
  EXPECT_EQ("group_objfile1", (*objfile).path());

  objfile = controlNode->getNextFile();
  EXPECT_NE(InputGraphError::no_more_files, error_code(objfile));
  EXPECT_EQ("group_objfile2", (*objfile).path());
}

// Node expansion tests.
TEST_F(InputGraphTest, ExpandInputGraphNode) {
  std::unique_ptr<MyFileNode> myfile(new MyFileNode("multi_files1", 0));
  std::vector<std::unique_ptr<File> > objfiles;
  std::unique_ptr<MyObjFile> obj1(new MyObjFile(_context, "objfile1"));
  std::unique_ptr<MyObjFile> obj2(new MyObjFile(_context, "objfile2"));
  objfiles.push_back(std::move(obj1));
  objfiles.push_back(std::move(obj2));
  myfile->addFiles(std::move(objfiles));
  EXPECT_EQ(true, inputGraph().addInputElement(std::move(myfile)));
  objfiles.clear();

  std::unique_ptr<MyExpandFileNode> expandFile(new MyExpandFileNode(
      "expand_node", 1, InputElement::ExpandType::ExpandOnly));

  std::unique_ptr<MyFileNode> filenode1(new MyFileNode("expand_file1", 2));
  std::unique_ptr<MyObjFile> obj3(new MyObjFile(_context, "objfile3"));
  objfiles.push_back(std::move(obj3));
  filenode1->addFiles(std::move(objfiles));
  expandFile->addElement(std::move(filenode1));
  objfiles.clear();

  std::unique_ptr<MyFileNode> filenode2(new MyFileNode("expand_file2", 3));
  std::unique_ptr<MyObjFile> obj4(new MyObjFile(_context, "objfile4"));
  objfiles.push_back(std::move(obj4));
  filenode2->addFiles(std::move(objfiles));
  expandFile->addElement(std::move(filenode2));
  objfiles.clear();

  // Add expand file to InputGraph
  EXPECT_EQ(true, inputGraph().addInputElement(std::move(expandFile)));

  std::unique_ptr<MyFileNode> filenode3(new MyFileNode("obj_after_expand", 4));
  std::unique_ptr<MyObjFile> obj5(new MyObjFile(_context, "objfile5"));
  std::unique_ptr<MyObjFile> obj6(new MyObjFile(_context, "objfile6"));
  objfiles.push_back(std::move(obj5));
  objfiles.push_back(std::move(obj6));
  filenode3->addFiles(std::move(objfiles));

  // Add an extra obj after the expand node
  EXPECT_EQ(true, inputGraph().addInputElement(std::move(filenode3)));

  inputGraph().normalize();

  ErrorOr<InputElement *> nextElement = inputGraph().getNextInputElement();
  EXPECT_NE(InputGraphError::no_more_elements, error_code(nextElement));
  EXPECT_EQ(InputElement::Kind::File, (*nextElement)->kind());
  FileNode *fileNode = llvm::dyn_cast<FileNode>(*nextElement);
  EXPECT_EQ("multi_files1", (*fileNode).getUserPath());

  nextElement = inputGraph().getNextInputElement();
  EXPECT_NE(InputGraphError::no_more_elements, error_code(nextElement));
  EXPECT_EQ(InputElement::Kind::File, (*nextElement)->kind());
  fileNode = llvm::dyn_cast<FileNode>(*nextElement);
  EXPECT_EQ("expand_file1", (*fileNode).getUserPath());

  nextElement = inputGraph().getNextInputElement();
  EXPECT_NE(InputGraphError::no_more_elements, error_code(nextElement));
  EXPECT_EQ(InputElement::Kind::File, (*nextElement)->kind());
  fileNode = llvm::dyn_cast<FileNode>(*nextElement);
  EXPECT_EQ("expand_file2", (*fileNode).getUserPath());

  nextElement = inputGraph().getNextInputElement();
  EXPECT_NE(InputGraphError::no_more_elements, error_code(nextElement));
  EXPECT_EQ(InputElement::Kind::File, (*nextElement)->kind());
  fileNode = llvm::dyn_cast<FileNode>(*nextElement);
  EXPECT_EQ("expand_node", (*fileNode).getUserPath());

  nextElement = inputGraph().getNextInputElement();
  EXPECT_NE(InputGraphError::no_more_elements, error_code(nextElement));
  EXPECT_EQ(InputElement::Kind::File, (*nextElement)->kind());
  fileNode = llvm::dyn_cast<FileNode>(*nextElement);
  EXPECT_EQ("obj_after_expand", (*fileNode).getUserPath());

  nextElement = inputGraph().getNextInputElement();
  EXPECT_EQ(InputGraphError::no_more_elements, error_code(nextElement));
}

// Node expansion tests.
TEST_F(InputGraphTest, ExpandAndReplaceInputGraphNode) {
  std::unique_ptr<MyFileNode> myfile(new MyFileNode("multi_files1", 0));
  std::vector<std::unique_ptr<File> > objfiles;
  std::unique_ptr<MyObjFile> obj1(new MyObjFile(_context, "objfile1"));
  std::unique_ptr<MyObjFile> obj2(new MyObjFile(_context, "objfile2"));
  objfiles.push_back(std::move(obj1));
  objfiles.push_back(std::move(obj2));
  myfile->addFiles(std::move(objfiles));
  EXPECT_EQ(true, inputGraph().addInputElement(std::move(myfile)));
  objfiles.clear();

  std::unique_ptr<MyExpandFileNode> expandFile(new MyExpandFileNode(
      "expand_node", 1, InputElement::ExpandType::ReplaceAndExpand));

  std::unique_ptr<MyFileNode> filenode1(new MyFileNode("expand_file1", 2));
  std::unique_ptr<MyObjFile> obj3(new MyObjFile(_context, "objfile3"));
  objfiles.push_back(std::move(obj3));
  filenode1->addFiles(std::move(objfiles));
  expandFile->addElement(std::move(filenode1));
  objfiles.clear();

  std::unique_ptr<MyFileNode> filenode2(new MyFileNode("expand_file2", 3));
  std::unique_ptr<MyObjFile> obj4(new MyObjFile(_context, "objfile4"));
  objfiles.push_back(std::move(obj4));
  filenode2->addFiles(std::move(objfiles));
  expandFile->addElement(std::move(filenode2));
  objfiles.clear();

  // Add expand file to InputGraph
  EXPECT_EQ(true, inputGraph().addInputElement(std::move(expandFile)));

  std::unique_ptr<MyFileNode> filenode3(new MyFileNode("obj_after_expand", 4));
  std::unique_ptr<MyObjFile> obj5(new MyObjFile(_context, "objfile5"));
  std::unique_ptr<MyObjFile> obj6(new MyObjFile(_context, "objfile6"));
  objfiles.push_back(std::move(obj5));
  objfiles.push_back(std::move(obj6));
  filenode3->addFiles(std::move(objfiles));

  // Add an extra obj after the expand node
  EXPECT_EQ(true, inputGraph().addInputElement(std::move(filenode3)));

  inputGraph().normalize();

  ErrorOr<InputElement *> nextElement = inputGraph().getNextInputElement();
  EXPECT_NE(InputGraphError::no_more_elements, error_code(nextElement));
  EXPECT_EQ(InputElement::Kind::File, (*nextElement)->kind());
  FileNode *fileNode = llvm::dyn_cast<FileNode>(*nextElement);
  EXPECT_EQ("multi_files1", (*fileNode).getUserPath());

  nextElement = inputGraph().getNextInputElement();
  EXPECT_NE(InputGraphError::no_more_elements, error_code(nextElement));
  EXPECT_EQ(InputElement::Kind::File, (*nextElement)->kind());
  fileNode = llvm::dyn_cast<FileNode>(*nextElement);
  EXPECT_EQ("expand_file1", (*fileNode).getUserPath());

  nextElement = inputGraph().getNextInputElement();
  EXPECT_NE(InputGraphError::no_more_elements, error_code(nextElement));
  EXPECT_EQ(InputElement::Kind::File, (*nextElement)->kind());
  fileNode = llvm::dyn_cast<FileNode>(*nextElement);
  EXPECT_EQ("expand_file2", (*fileNode).getUserPath());

  nextElement = inputGraph().getNextInputElement();
  EXPECT_NE(InputGraphError::no_more_elements, error_code(nextElement));
  EXPECT_EQ(InputElement::Kind::File, (*nextElement)->kind());
  fileNode = llvm::dyn_cast<FileNode>(*nextElement);
  EXPECT_EQ("obj_after_expand", (*fileNode).getUserPath());

  nextElement = inputGraph().getNextInputElement();
  EXPECT_EQ(InputGraphError::no_more_elements, error_code(nextElement));
}
}
