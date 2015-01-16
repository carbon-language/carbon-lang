//===- lld/Core/Node.h - Input file class ---------------------------------===//
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
/// The classes in this file represents inputs to the linker.
///
//===----------------------------------------------------------------------===//

#ifndef LLD_CORE_NODE_H
#define LLD_CORE_NODE_H

#include "lld/Core/File.h"
#include "llvm/Option/ArgList.h"
#include <memory>
#include <vector>

namespace lld {

// A Node represents a FileNode or other type of Node. In the latter case,
// the node contains meta information about the input file list.
// Currently only GroupEnd node is defined as a meta node.
class Node {
public:
  enum class Kind { File, GroupEnd };
  explicit Node(Kind type) : _kind(type) {}
  virtual ~Node() {}
  virtual Kind kind() const { return _kind; }

private:
  Kind _kind;
};

// This is a marker for --end-group. getSize() returns the number of
// files between the corresponding --start-group and this marker.
class GroupEnd : public Node {
public:
  explicit GroupEnd(int size) : Node(Kind::GroupEnd), _size(size) {}

  int getSize() const { return _size; }

  static inline bool classof(const Node *a) {
    return a->kind() == Kind::GroupEnd;
  }

private:
  int _size;
};

// A container of File.
class FileNode : public Node {
public:
  explicit FileNode(std::unique_ptr<File> f)
      : Node(Node::Kind::File), _file(std::move(f)), _asNeeded(false) {}

  static inline bool classof(const Node *a) {
    return a->kind() == Node::Kind::File;
  }

  File *getFile() { return _file.get(); }

  void setAsNeeded(bool val) { _asNeeded = val; }
  bool asNeeded() const { return _asNeeded; }

protected:
  std::unique_ptr<File> _file;
  bool _asNeeded;
};

} // namespace lld

#endif // LLD_CORE_NODE_H
