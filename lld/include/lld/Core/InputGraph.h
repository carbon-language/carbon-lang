//===- lld/Core/InputGraph.h - Input Graph --------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
///
/// Inputs to the linker in the form of a Graph.
///
//===----------------------------------------------------------------------===//

#ifndef LLD_CORE_INPUT_GRAPH_H
#define LLD_CORE_INPUT_GRAPH_H

#include "lld/Core/File.h"
#include "llvm/Option/ArgList.h"

#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <stack>
#include <vector>

namespace lld {

class InputElement;
class LinkingContext;

/// \brief The inputs to the linker are represented by an InputGraph. The
/// nodes in the input graph contains Input elements. The InputElements are
/// either Input Files or Control Options. The Input Files represent each Input
/// File to the linker and the control option specify what the linker needs
/// to do when it processes the option.
/// Each InputElement that is part of the Graph has an Ordinal value
/// associated with it. The ordinal value is needed for the Writer to figure out
/// the relative position of the arguments that appeared in the Command Line.
class InputGraph {
public:
  typedef std::vector<std::unique_ptr<InputElement> > InputElementVectorT;
  typedef InputElementVectorT::iterator InputElementIterT;
  typedef std::vector<std::unique_ptr<File> > FileVectorT;
  typedef FileVectorT::iterator FileIterT;

  /// Where do we want to insert the input element when calling the
  /// insertElementAt, insertOneElementAt API's.
  enum Position : uint8_t {
    ANY,
    BEGIN,
    END
  };

  /// \brief Initialize the inputgraph
  InputGraph() : _ordinal(0), _nextElementIndex(0) {}

  /// \brief Adds a node into the InputGraph
  virtual bool addInputElement(std::unique_ptr<InputElement>);

  /// \brief Set Ordinals for all the InputElements that form the InputGraph
  virtual bool assignOrdinals();

  /// Normalize the InputGraph. It visits all nodes in the tree to replace a
  /// node with its children if it's shouldExpand() returnst true.
  virtual void normalize();

  virtual ~InputGraph() {}

  /// \brief Do postprocessing of the InputGraph if there is a need for the
  /// to provide additional information to the user, also rearranges
  /// InputElements by their ordinals. If a user wants to place an input file
  /// at the desired position, the user can do that.
  virtual void doPostProcess();

  range<InputElementIterT> inputElements() {
    return make_range(_inputArgs.begin(), _inputArgs.end());
  }

  // \brief Does the inputGraph contain any elements
  size_t size() const { return _inputArgs.size(); }

  /// \brief Dump the input Graph
  bool dump(raw_ostream &diagnostics = llvm::errs());

  InputElement &operator[](size_t index) const {
    return (*_inputArgs[index]);
  }

  /// \brief Insert an element into the input graph at position.
  void insertOneElementAt(std::unique_ptr<InputElement>,
                          Position position, size_t pos = 0);

  /// \brief Helper functions for the resolver
  ErrorOr<InputElement *> getNextInputElement();

  /// \brief Reset the inputGraph for the inputGraph to start processing
  /// files from the beginning
  void reset() { _nextElementIndex = 0; }

protected:
  // Input arguments
  InputElementVectorT _inputArgs;
  // Ordinals
  int64_t _ordinal;
  // Index of the next element to be processed
  uint32_t _nextElementIndex;
};

/// \brief This describes each element in the InputGraph. The Kind
/// determines what the current node contains.
class InputElement {
public:
  /// Each input element in the graph can be a File or a control
  enum class Kind : uint8_t {
    Group,      // Represents a type associated with Group
    File        // Represents a type associated with File Nodes
  };

  /// \brief Initialize the Input Element, The ordinal value of an input Element
  /// is initially set to -1, if the user wants to override its ordinal,
  /// let the user do it
  InputElement(Kind type, int64_t ordinal = -1);

  virtual ~InputElement() {}

  /// Return the Element Type for an Input Element
  virtual Kind kind() const { return _kind; }

  virtual void setOrdinal(int64_t ordinal) {
    if (_ordinal != -1)
      _ordinal = ordinal;
  }

  virtual int64_t getOrdinal() const { return _ordinal; }

  /// \brief Dump the Input Element
  virtual bool dump(raw_ostream &diagnostics) { return true; }

  /// \brief parse the input element
  virtual error_code parse(const LinkingContext &, raw_ostream &) = 0;

  /// \brief functions for the resolver to use

  /// Get the next file to be processed by the resolver
  virtual ErrorOr<File &> getNextFile() = 0;

  /// \brief Set the resolve state for the element
  virtual void setResolveState(uint32_t state) = 0;

  /// \brief Get the resolve state for the element
  virtual uint32_t getResolveState() const = 0;

  /// \brief Reset the next index
  virtual void resetNextIndex() = 0;

  /// Normalize functions

  /// Returns true if we want to replace this node with children.
  virtual bool shouldExpand() const { return false; }

  /// \brief Get the elements that we want to expand with.
  virtual range<InputGraph::InputElementIterT> expandElements() {
    llvm_unreachable("no element to expand");
  }

protected:
  Kind _kind;              // The type of the Element
  int64_t _ordinal;        // The ordinal value
};

/// \brief A Control node which contains a group of InputElements
/// This affects the resolver so that it resolves undefined symbols
/// in the group completely before looking at other input files that
/// follow the group
class Group : public InputElement {
public:
  Group(int64_t ordinal)
      : InputElement(InputElement::Kind::Group, ordinal),
        _currentElementIndex(0), _nextElementIndex(0) {}

  static inline bool classof(const InputElement *a) {
    return a->kind() == InputElement::Kind::Group;
  }

  /// \brief Process input element and add it to the group
  bool addFile(std::unique_ptr<InputElement> element) {
    _elements.push_back(std::move(element));
    return true;
  }

  range<InputGraph::InputElementIterT> elements() {
    return make_range(_elements.begin(), _elements.end());
  }

  void resetNextIndex() override {
    _currentElementIndex = _nextElementIndex = 0;
    for (auto &elem : _elements)
      elem->resetNextIndex();
  }

  uint32_t getResolveState() const override;
  void setResolveState(uint32_t) override;
  ErrorOr<File &> getNextFile() override;

protected:
  InputGraph::InputElementVectorT _elements;
  uint32_t _currentElementIndex;
  uint32_t _nextElementIndex;
};

/// \brief Represents an Input file in the graph
///
/// This class represents an input to the linker. It create the MemoryBuffer
/// lazily when needed based on the file path. It can also take a MemoryBuffer
/// directly.
class FileNode : public InputElement {
public:
  FileNode(StringRef path, int64_t ordinal = -1);

  virtual ErrorOr<StringRef> getPath(const LinkingContext &) const {
    return _path;
  }

  // The saved input path thats used when a file is not found while
  // trying to parse a file
  StringRef getUserPath() const { return _path; }

  virtual ~FileNode() {}

  /// \brief Casting support
  static inline bool classof(const InputElement *a) {
    return a->kind() == InputElement::Kind::File;
  }

  /// \brief create an error string for printing purposes
  virtual std::string errStr(error_code errc) {
    std::string msg = errc.message();
    Twine twine = Twine("Cannot open ") + _path + ": " + msg;
    return twine.str();
  }

  /// \brief Get the list of files
  range<InputGraph::FileIterT> files() {
    return make_range(_files.begin(), _files.end());
  }

  /// \brief number of files.
  size_t numFiles() const { return _files.size(); }

  /// \brief add a file to the list of files
  virtual void addFiles(InputGraph::FileVectorT files) {
    for (auto &ai : files)
      _files.push_back(std::move(ai));
  }

  /// \brief Reset the file index if the resolver needs to process
  /// the node again.
  void resetNextIndex() override;

  /// \brief Set the resolve state for the FileNode.
  void setResolveState(uint32_t resolveState) override {
    _resolveState = resolveState;
  }

  /// \brief Retrieve the resolve state of the FileNode.
  uint32_t getResolveState() const override { return _resolveState; }

protected:
  /// \brief Read the file into _buffer.
  error_code getBuffer(StringRef filePath);

  StringRef _path;                             // The path of the Input file
  InputGraph::FileVectorT _files;              // A vector of lld File objects
  std::unique_ptr<MemoryBuffer> _buffer; // Memory buffer to actual
                                               // contents
  uint32_t _resolveState;                      // The resolve state of the file
  uint32_t _nextFileIndex; // The next file that would be processed by the
                           // resolver
};

/// \brief Represents Internal Input files
class SimpleFileNode : public FileNode {
public:
  SimpleFileNode(StringRef path, int64_t ordinal = -1)
      : FileNode(path, ordinal) {}

  virtual ~SimpleFileNode() {}

  /// \brief add a file to the list of files
  virtual void appendInputFile(std::unique_ptr<File> f) {
    _files.push_back(std::move(f));
  }

  /// \brief parse the input element
  error_code parse(const LinkingContext &, raw_ostream &) override {
    return error_code::success();
  }

  /// \brief Return the next File thats part of this node to the
  /// resolver.
  ErrorOr<File &> getNextFile() override {
    if (_nextFileIndex == _files.size())
      return make_error_code(InputGraphError::no_more_files);
    return *_files[_nextFileIndex++];
  }

  // Do nothing here.
  void resetNextIndex() override {}
};
} // namespace lld

#endif // LLD_CORE_INPUT_GRAPH_H
