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
#include "llvm/ADT/StringRef.h"
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
/// InputElements have a weight function that can be used to determine the
/// weight of the file, for statistical purposes.
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

  /// Normalize the InputGraph.
  virtual void normalize();

  /// Destructor
  virtual ~InputGraph() {}

  /// \brief Do postprocessing of the InputGraph if there is a need for the
  /// to provide additional information to the user, also rearranges
  /// InputElements by their ordinals. If an user wants to place an input file
  /// at the desired position, the user can do that
  virtual void doPostProcess();

  range<InputElementIterT> inputElements() {
    return make_range(_inputArgs.begin(), _inputArgs.end());
  }

  /// \brief Validate the input graph
  virtual bool validate();

  // \brief Does the inputGraph contain any elements
  size_t size() const { return _inputArgs.size(); }

  /// \brief Dump the input Graph
  virtual bool dump(raw_ostream &diagnostics = llvm::errs());

  InputElement &operator[](size_t index) const {
    return (*_inputArgs[index]);
  }

  /// \brief Insert a vector of elements into the input graph at position.
  virtual void insertElementsAt(std::vector<std::unique_ptr<InputElement> >,
                                Position position, size_t pos = 0);

  /// \brief Insert an element into the input graph at position.
  virtual void insertOneElementAt(std::unique_ptr<InputElement>,
                                  Position position, size_t pos = 0);

  /// \brief Helper functions for the resolver
  virtual ErrorOr<InputElement *> getNextInputElement();

  /// \brief Set the index on what inputElement has to be returned
  virtual error_code setNextElementIndex(uint32_t index = 0);

  /// \brief Reset the inputGraph for the inputGraph to start processing
  /// files from the beginning
  virtual error_code reset() { return setNextElementIndex(0); }

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
    Control,    // Represents a type associated with ControlNodes
    SimpleFile, // Represents a type reserved for internal files
    File        // Represents a type associated with File Nodes
  };

  /// How does the inputGraph expand the InputElement
  enum class ExpandType : uint8_t {
    None,             // Do nothing(Default)
    ReplaceAndExpand, // Replace current node and expand
    ExpandOnly        // Expand the current node
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

  virtual int64_t weight() const { return _weight; }

  virtual void setWeight(int64_t weight) { _weight = weight; }

  /// \brief validates the Input Element
  virtual bool validate() = 0;

  /// \brief Dump the Input Element
  virtual bool dump(raw_ostream &diagnostics) = 0;

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

  /// \brief Is this a hidden node, hidden nodes are not part of
  /// of the resolver.
  virtual bool isHidden() const { return false; }

  /// Normalize functions

  /// \brief How do we want to expand the current node ?
  virtual ExpandType expandType() const { return ExpandType::None; }

  /// \brief Get the elements that we want to expand with.
  virtual range<InputGraph::InputElementIterT> expandElements() {
    llvm_unreachable("no element to expand");
  }

protected:
  Kind _kind;              // The type of the Element
  int64_t _ordinal;        // The ordinal value
  int64_t _weight;         // Weight of the file
};

/// \brief The Control class represents a control node in the InputGraph
class ControlNode : public InputElement {
public:
  /// A control node could be of several types supported by InputGraph
  /// Future kinds of Control node could be added
  enum class ControlKind : uint8_t{
    Simple, // Represents a simple control node
    Group   // Represents a type associated with ControlNodes
  };

  ControlNode(ControlNode::ControlKind controlKind =
                  ControlNode::ControlKind::Simple,
              int64_t _ordinal = -1)
      : InputElement(InputElement::Kind::Control, _ordinal),
        _controlKind(controlKind), _currentElementIndex(0),
        _nextElementIndex(0) {}

  virtual ~ControlNode() {}

  /// \brief Return the kind of control node
  virtual ControlNode::ControlKind controlKind() { return _controlKind; }

  /// \brief Process control start/exit
  virtual bool processControlEnter() { return true; }

  /// \brief Process control start/exit
  virtual bool processControlExit() { return true; }

  /// Process the input Elemenet
  virtual bool processInputElement(std::unique_ptr<InputElement> element) = 0;

  /// \brief Casting support
  static inline bool classof(const InputElement *a) {
    return a->kind() == InputElement::Kind::Control;
  }

  range<InputGraph::InputElementIterT> elements() {
    return make_range(_elements.begin(), _elements.end());
  }

  virtual void resetNextIndex() {
    _currentElementIndex = _nextElementIndex = 0;
    for (auto &elem : _elements)
      elem->resetNextIndex();
  }

  virtual uint32_t getResolveState() const;

  virtual void setResolveState(uint32_t);

protected:
  ControlKind _controlKind;
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

  /// \brief  number of files.
  size_t numFiles() const { return _files.size(); }

  /// \brief add a file to the list of files
  virtual void addFiles(InputGraph::FileVectorT files) {
    for (auto &ai : files)
      _files.push_back(std::move(ai));
  }

  /// \brief Reset the file index if the resolver needs to process
  /// the node again.
  virtual void resetNextIndex();

  /// \brief Set the resolve state for the FileNode.
  virtual void setResolveState(uint32_t resolveState) {
    _resolveState = resolveState;
  }

  /// \brief Retrieve the resolve state of the FileNode.
  virtual uint32_t getResolveState() const { return _resolveState; }

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

/// \brief A Control node which contains a group of InputElements
/// This affects the resolver so that it resolves undefined symbols
/// in the group completely before looking at other input files that
/// follow the group
class Group : public ControlNode {
public:
  Group(int64_t ordinal)
      : ControlNode(ControlNode::ControlKind::Group, ordinal) {}

  static inline bool classof(const InputElement *a) {
    return a->kind() == InputElement::Kind::Control;
  }

  /// \brief Process input element and add it to the group
  virtual bool processInputElement(std::unique_ptr<InputElement> element) {
    _elements.push_back(std::move(element));
    return true;
  }

  virtual ErrorOr<File &> getNextFile();
};

/// \brief Represents Internal Input files
class SimpleFileNode : public InputElement {
public:
  SimpleFileNode(StringRef path, int64_t ordinal = -1);

  virtual ErrorOr<StringRef> path(const LinkingContext &) const {
    return _path;
  }

  // The saved input path thats used when a file is not found while
  // trying to parse a file
  StringRef getUserPath() const { return _path; }

  virtual ~SimpleFileNode() {}

  /// \brief Casting support
  static inline bool classof(const InputElement *a) {
    return a->kind() == InputElement::Kind::SimpleFile;
  }

  /// \brief Get the list of files
  range<InputGraph::FileIterT> files() {
    return make_range(_files.begin(), _files.end());
  }

  /// \brief  number of files.
  size_t numFiles() const { return _files.size(); }

  /// \brief add a file to the list of files
  virtual void appendInputFile(std::unique_ptr<File> f) {
    _files.push_back(std::move(f));
  }

  /// \brief add a file to the list of files
  virtual void appendInputFiles(InputGraph::FileVectorT files) {
    for (auto &ai : files)
      _files.push_back(std::move(ai));
  }

  /// \brief validates the Input Element
  virtual bool validate() { return true; }

  /// \brief Dump the Input Element
  virtual bool dump(raw_ostream &) { return true; }

  /// \brief parse the input element
  virtual error_code parse(const LinkingContext &, raw_ostream &) {
    return error_code::success();
  }

  /// \brief Return the next File thats part of this node to the
  /// resolver.
  virtual ErrorOr<File &> getNextFile() {
    if (_nextFileIndex == _files.size())
      return make_error_code(InputGraphError::no_more_files);
    return *_files[_nextFileIndex++];
  }

  /// \brief Set the resolver state.
  virtual void setResolveState(uint32_t resolveState) {
    _resolveState = resolveState;
  }

  /// \brief Retrieve the resolve state.
  virtual uint32_t getResolveState() const { return _resolveState; }

  // Do nothing here.
  virtual void resetNextIndex() {}

protected:
  StringRef _path;                // A string associated with this file.
  InputGraph::FileVectorT _files; // Vector of lld::File objects
  uint32_t _nextFileIndex; // The next file that would be processed by the
                           // resolver
  uint32_t _resolveState;  // The resolve state associated with this Node
};
} // namespace lld

#endif // LLD_DRIVER_INPUT_GRAPH_H
