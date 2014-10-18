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

  /// getNextFile returns the next file that needs to be processed by
  /// the resolver. When there are no more files to be processed, an
  /// appropriate InputGraphError is returned. Ordinals are assigned
  /// to files returned by getNextFile, which means ordinals would be
  /// assigned in the way files are resolved.
  ErrorOr<File &> getNextFile();

  /// Notifies the current input element of Resolver made some progress on
  /// resolving undefined symbols using the current file. Group (representing
  /// --start-group and --end-group) uses that notification to make a decision
  /// whether it should iterate over again or terminate or not.
  void notifyProgress();

  /// Adds an observer of getNextFile(). Each time a new file is about to be
  /// returned from getNextFile(), registered observers are called with the file
  /// being returned.
  void registerObserver(std::function<void(File *)>);

  /// \brief Adds a node into the InputGraph
  void addInputElement(std::unique_ptr<InputElement>);

  /// \brief Adds a node at the beginning of the InputGraph
  void addInputElementFront(std::unique_ptr<InputElement>);

  /// Normalize the InputGraph. It calls expand() on each node and then replace
  /// it with getReplacements() results.
  void normalize();

  range<InputElementIterT> inputElements() {
    return make_range(_inputArgs.begin(), _inputArgs.end());
  }

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
    Group,      // Represents a type associated with Group
    File        // Represents a type associated with File Nodes
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

  /// Refer InputGraph::notifyProgress(). By default, it does nothing. Only
  /// Group is interested in this message.
  virtual void notifyProgress() {};

  /// \brief Reset the next index
  virtual void resetNextIndex() = 0;

  /// Returns true if we want to replace this node with children.
  virtual void expand() {}

  /// Get the elements that we want to expand with.
  virtual bool getReplacements(InputGraph::InputElementVectorT &) {
    return false;
  }

protected:
  Kind _kind; // The type of the Element
};

/// \brief A Control node which contains a group of InputElements
/// This affects the resolver so that it resolves undefined symbols
/// in the group completely before looking at other input files that
/// follow the group
class Group : public InputElement {
public:
  Group()
      : InputElement(InputElement::Kind::Group), _currentElementIndex(0),
        _nextElementIndex(0), _madeProgress(false) {}

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
    _madeProgress = false;
    _currentElementIndex = 0;
    _nextElementIndex = 0;
    for (std::unique_ptr<InputElement> &elem : _elements)
      elem->resetNextIndex();
  }

  /// \brief Parse the group members.
  std::error_code parse(const LinkingContext &ctx, raw_ostream &diag) override {
    for (std::unique_ptr<InputElement> &ei : _elements)
      if (std::error_code ec = ei->parse(ctx, diag))
        return ec;
    return std::error_code();
  }

  /// If Resolver made a progress using the current file, it's ok to revisit
  /// files in this group in future.
  void notifyProgress() override {
    for (std::unique_ptr<InputElement> &elem : _elements)
      elem->notifyProgress();
    _madeProgress = true;
  }

  ErrorOr<File &> getNextFile() override;

  void expand() override {
    for (std::unique_ptr<InputElement> &elt : _elements)
      elt->expand();
    std::vector<std::unique_ptr<InputElement>> result;
    for (std::unique_ptr<InputElement> &elt : _elements) {
      if (elt->getReplacements(result))
        continue;
      result.push_back(std::move(elt));
    }
    _elements.swap(result);
  }

protected:
  InputGraph::InputElementVectorT _elements;
  uint32_t _currentElementIndex;
  uint32_t _nextElementIndex;
  bool _madeProgress;
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
    for (std::unique_ptr<File> &ai : files)
      _files.push_back(std::move(ai));
  }

  /// \brief Reset the file index if the resolver needs to process
  /// the node again.
  void resetNextIndex() override { _nextFileIndex = 0; }

protected:
  /// \brief Read the file into _buffer.
  std::error_code getBuffer(StringRef filePath);

  StringRef _path;                       // The path of the Input file
  InputGraph::FileVectorT _files;        // A vector of lld File objects
  std::unique_ptr<MemoryBuffer> _buffer; // Memory buffer to actual contents

  // The next file that would be processed by the resolver
  uint32_t _nextFileIndex;
};

/// \brief Represents Internal Input files
class SimpleFileNode : public FileNode {
public:
  SimpleFileNode(StringRef path) : FileNode(path) {}

  virtual ~SimpleFileNode() {}

  /// \brief add a file to the list of files
  virtual void appendInputFile(std::unique_ptr<File> f) {
    _files.push_back(std::move(f));
  }

  /// \brief parse the input element
  std::error_code parse(const LinkingContext &, raw_ostream &) override {
    return std::error_code();
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
