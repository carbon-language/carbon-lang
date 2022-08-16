// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_MIGRATE_CPP_CPP_REFACTORING_OUTPUT_SEGMENT_H_
#define CARBON_MIGRATE_CPP_CPP_REFACTORING_OUTPUT_SEGMENT_H_

#include <string>
#include <utility>
#include <variant>

#include "clang/AST/ASTTypeTraits.h"

namespace Carbon {

// Represents a segment of the output string. `OutputSegment`s come in two
// flavors: Text and Node. A text segment holds string text that should be used
// to be added to the output. A node segment holds a node in Clang's AST and
// indicates that the output associated to that node should be the output
// segment that the `RewriteBuilder` (defined below) has attached to that AST
// node.
//
// For example, the output for a binary operator node corresponding to the C++
// code snippet `f() + 3 * 5`, would be the sequence of three output segments:
//
//                  {Node(lhs), Text(" + "), Node(rhs)}
//
// The left-hand side and right-hand side can then be queried recursively to
// determine what their output should be.
//
class OutputSegment {
 public:
  // Each of these there overloads creates a text-based `OutputSegment`.
  static auto Text(std::string content) -> OutputSegment {
    return OutputSegment(std::move(content));
  }
  static auto Text(std::string_view content) -> OutputSegment {
    return Text(std::string(content));
  }
  static auto Text(const char* content) -> OutputSegment {
    return Text(std::string(content));
  }

  // Creates a node-based `OutputSegment` from `node`.
  static auto Node(const clang::DynTypedNode& node) -> OutputSegment {
    return OutputSegment(node);
  }
  template <typename T>
  static auto Node(const T* node) -> OutputSegment {
    assert(node != nullptr);
    return OutputSegment(clang::DynTypedNode::create(*node));
  }

  // Creates a TypeLoc-based `OutputSegment` from `type_loc`.
  static auto TypeLoc(clang::TypeLoc type_loc) -> OutputSegment {
    // Traversals for TypeLocs have some sharp corners. In particular,
    // QualifiedTypeLocs are silently passed through to their unqualified
    // part. This means that when constructing output segments we also need to
    // match this behavior.
    auto qtl = type_loc.getAs<clang::QualifiedTypeLoc>();
    return OutputSegment(qtl.isNull() ? type_loc : qtl.getUnqualifiedLoc());
  }

 private:
  friend class OutputWriter;
  explicit OutputSegment(std::string content) : content(std::move(content)) {}
  explicit OutputSegment(const clang::DynTypedNode& node) : content(node) {}
  explicit OutputSegment(clang::TypeLoc type_loc) : content(type_loc) {}

  std::variant<std::string, clang::DynTypedNode, clang::TypeLoc> content;
};

}  // namespace Carbon

#endif  // CARBON_MIGRATE_CPP_CPP_REFACTORING_OUTPUT_SEGMENT_H_
