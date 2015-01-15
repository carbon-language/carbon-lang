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
  InputGraph() : _index(0) {}

  /// \brief Adds a node into the InputGraph
  void addInputElement(std::unique_ptr<InputElement> ie) {
    _members.push_back(std::move(ie));
  }

  /// \brief Adds a node at the beginning of the InputGraph
  void addInputElementFront(std::unique_ptr<InputElement> ie) {
    _members.insert(_members.begin(), std::move(ie));
  }

  InputElementVectorT &members() { return _members; }

protected:
  // Input arguments
  InputElementVectorT _members;
  // Index of the next element to be processed
  size_t _index;
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

  /// \brief parse the input element
  virtual std::error_code parse(const LinkingContext &, raw_ostream &) = 0;

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
      : InputElement(InputElement::Kind::File), _path(path), _done(false) {
  }

  FileNode(StringRef path, std::unique_ptr<File> f)
      : InputElement(InputElement::Kind::File), _path(path), _file(std::move(f)),
        _done(false) {}

  virtual ErrorOr<StringRef> getPath(const LinkingContext &) const {
    return _path;
  }

  virtual ~FileNode() {}

  /// \brief Casting support
  static inline bool classof(const InputElement *a) {
    return a->kind() == InputElement::Kind::File;
  }

  /// \brief Get the list of files
  File *getFile() { return _file.get(); }

  /// \brief add a file to the list of files
  virtual void addFiles(InputGraph::FileVectorT files) {
    assert(files.size() == 1);
    assert(!_file);
    _file = std::move(files[0]);
  }

  std::error_code parse(const LinkingContext &, raw_ostream &) override;

protected:
  StringRef _path;                       // The path of the Input file
  std::unique_ptr<File> _file;           // An lld File object

  // The next file that would be processed by the resolver
  bool _done;
};

} // namespace lld

#endif // LLD_CORE_INPUT_GRAPH_H
