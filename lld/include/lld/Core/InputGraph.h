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

class InputGraph {
public:
  /// \brief Adds a node into the InputGraph
  void addInputElement(std::unique_ptr<InputElement> ie) {
    _members.push_back(std::move(ie));
  }

  /// \brief Adds a node at the beginning of the InputGraph
  void addInputElementFront(std::unique_ptr<InputElement> ie) {
    _members.insert(_members.begin(), std::move(ie));
  }

  std::vector<std::unique_ptr<InputElement>> &members() {
    return _members;
  }

protected:
  std::vector<std::unique_ptr<InputElement>> _members;
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

private:
  int _size;
};

// A container of File.
class FileNode : public InputElement {
public:
  explicit FileNode(std::unique_ptr<File> f)
      : InputElement(InputElement::Kind::File), _file(std::move(f)) {}

  virtual ~FileNode() {}

  static inline bool classof(const InputElement *a) {
    return a->kind() == InputElement::Kind::File;
  }

  File *getFile() { return _file.get(); }

protected:
  std::unique_ptr<File> _file;
};

} // namespace lld

#endif // LLD_CORE_INPUT_GRAPH_H
