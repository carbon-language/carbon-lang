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
#include "lld/Core/range.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
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

  /// \brief Initialize the inputgraph
  InputGraph() : _nextElementIndex(0), _currentInputElement(nullptr) {}
  virtual ~InputGraph();

  /// getNextFile returns the next file that needs to be processed by
  /// the resolver. When there are no more files to be processed, an
  /// appropriate InputGraphError is returned. Ordinals are assigned
  /// to files returned by getNextFile, which means ordinals would be
  /// assigned in the way files are resolved.
  virtual ErrorOr<File &> getNextFile();

  /// Adds an observer of getNextFile(). Each time a new file is about to be
  /// returned from getNextFile(), registered observers are called with the file
  /// being returned.
  void registerObserver(std::function<void(File *)>);

  /// \brief Adds a node into the InputGraph
  void addInputElement(std::unique_ptr<InputElement>);

  /// \brief Adds a node at the beginning of the InputGraph
  void addInputElementFront(std::unique_ptr<InputElement>);

  /// Normalize the InputGraph. It calls getReplacements() on each element.
  void normalize();

  InputElementVectorT &inputElements() {
    return _inputArgs;
  }

  // Returns the current group size if we are at an --end-group.
  // Otherwise returns 0.
  int getGroupSize();
  void skipGroup();

  // \brief Returns the number of input files.
  size_t size() const { return _inputArgs.size(); }

  /// \brief Dump the input Graph
  bool dump(raw_ostream &diagnostics = llvm::errs());

protected:
  // Input arguments
  InputElementVectorT _inputArgs;
  // Index of the next element to be processed
  uint32_t _nextElementIndex;
  InputElement *_currentInputElement;
  std::vector<std::function<void(File *)>> _observers;

private:
  ErrorOr<InputElement *> getNextInputElement();
};

/// \brief This describes each element in the InputGraph. The Kind
/// determines what the current node contains.
class InputElement {
public:
  /// Each input element in the graph can be a File or a control
  enum class Kind : uint8_t {
    File,       // Represents a type associated with File Nodes
    GroupEnd,
  };

  InputElement(Kind type) : _kind(type) {}
  virtual ~InputElement() {}

  /// Return the Element Type for an Input Element
  virtual Kind kind() const { return _kind; }

  /// \brief Dump the Input Element
  virtual bool dump(raw_ostream &diagnostics) { return true; }

  /// \brief parse the input element
  virtual std::error_code parse(const LinkingContext &, raw_ostream &) = 0;

  /// \brief functions for the resolver to use

  /// Get the next file to be processed by the resolver
  virtual ErrorOr<File &> getNextFile() = 0;

  /// \brief Reset the next index
  virtual void resetNextIndex() = 0;

  /// Get the elements that we want to expand with.
  virtual bool getReplacements(InputGraph::InputElementVectorT &) {
    return false;
  }

protected:
  Kind _kind; // The type of the Element
};

// This is a marker for --end-group. getSize() returns the number of
// files between the corresponding --start-group and this marker.
class GroupEnd : public InputElement {
public:
  GroupEnd(int size) : InputElement(Kind::GroupEnd), _size(size) {}

  int getSize() const { return _size; }

  static inline bool classof(const InputElement *a) {
    return a->kind() == Kind::GroupEnd;
  }

  /// \brief Parse the group members.
  std::error_code parse(const LinkingContext &ctx, raw_ostream &diag) override {
    return std::error_code();
  }

  ErrorOr<File &> getNextFile() override {
    llvm_unreachable("shouldn't be here.");
  }

  void resetNextIndex() override {}

private:
  int _size;
};

/// \brief Represents an Input file in the graph
///
/// This class represents an input to the linker. It create the MemoryBuffer
/// lazily when needed based on the file path. It can also take a MemoryBuffer
/// directly.
class FileNode : public InputElement {
public:
  FileNode(StringRef path)
      : InputElement(InputElement::Kind::File), _path(path), _nextFileIndex(0) {
  }

  virtual ErrorOr<StringRef> getPath(const LinkingContext &) const {
    return _path;
  }

  virtual ~FileNode() {}

  /// \brief Casting support
  static inline bool classof(const InputElement *a) {
    return a->kind() == InputElement::Kind::File;
  }

  /// \brief create an error string for printing purposes
  virtual std::string errStr(std::error_code errc) {
    std::string msg = errc.message();
    return (Twine("Cannot open ") + _path + ": " + msg).str();
  }

  /// \brief Get the list of files
  range<InputGraph::FileIterT> files() {
    return make_range(_files.begin(), _files.end());
  }

  /// \brief add a file to the list of files
  virtual void addFiles(InputGraph::FileVectorT files) {
    assert(files.size() == 1);
    assert(_files.empty());
    for (std::unique_ptr<File> &ai : files)
      _files.push_back(std::move(ai));
  }

  /// \brief Reset the file index if the resolver needs to process
  /// the node again.
  void resetNextIndex() override { _nextFileIndex = 0; }

  bool getReplacements(InputGraph::InputElementVectorT &result) override;

  /// \brief Return the next File thats part of this node to the
  /// resolver.
  ErrorOr<File &> getNextFile() override {
    if (_nextFileIndex == _files.size())
      return make_error_code(InputGraphError::no_more_files);
    return *_files[_nextFileIndex++];
  }

protected:
  StringRef _path;                       // The path of the Input file
  InputGraph::FileVectorT _files;        // A vector of lld File objects

  // The next file that would be processed by the resolver
  uint32_t _nextFileIndex;
};

/// \brief Represents Internal Input files
class SimpleFileNode : public FileNode {
public:
  SimpleFileNode(StringRef path) : FileNode(path) {}
  SimpleFileNode(StringRef path, std::unique_ptr<File> f)
      : FileNode(path) {
    _files.push_back(std::move(f));
  }

  virtual ~SimpleFileNode() {}

  /// \brief add a file to the list of files
  virtual void appendInputFile(std::unique_ptr<File> f) {
    _files.push_back(std::move(f));
  }

  /// \brief parse the input element
  std::error_code parse(const LinkingContext &, raw_ostream &) override {
    return std::error_code();
  }
};
} // namespace lld

#endif // LLD_CORE_INPUT_GRAPH_H
