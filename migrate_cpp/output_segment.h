// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_MIGRATE_CPP_OUTPUT_SEGMENT_H_
#define CARBON_MIGRATE_CPP_OUTPUT_SEGMENT_H_

#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include "clang/AST/ASTTypeTraits.h"
#include "common/check.h"

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
class OutputSegment {
 public:
  // Returns whether or not the type T is an acceptable node type from which an
  // OutputSegment can be constructed. We intentionally do not want to support
  // `clang::Type` because we support traversing through `clang::TypeLoc`
  // instead. However, most other types we intend to support as they become
  // necessary.
  template <typename T>
  static constexpr auto IsSupportedClangASTNodeType() -> bool {
    return std::is_convertible_v<T*, clang::Stmt*> ||
           std::is_convertible_v<T*, clang::Decl*>;
  }

  // Creates a text-based `OutputSegment`.
  explicit OutputSegment(std::string content) : content_(std::move(content)) {}
  explicit OutputSegment(llvm::StringRef content) : content_(content.str()) {}
  explicit OutputSegment(const char* content) : content_(content) {}

  // Creates a node-based `OutputSegment` from `node`.
  explicit OutputSegment(const clang::DynTypedNode& node) : content_(node) {}
  template <typename T,
            std::enable_if_t<OutputSegment::IsSupportedClangASTNodeType<T>(),
                             int> = 0>
  explicit OutputSegment(const T* node);

  // Creates a TypeLoc-based `OutputSegment` from `type_loc`.
  explicit OutputSegment(clang::TypeLoc type_loc)
      : content_(PassThroughQualifiedTypeLoc(type_loc)) {}

 private:
  friend struct OutputWriter;

  template <typename T>
  auto AssertNotNull(T* ptr) -> T& {
    CARBON_CHECK(ptr != nullptr);
    return *ptr;
  }

  // Traversals for TypeLocs have some sharp corners. In particular,
  // QualifiedTypeLocs are silently passed through to their unqualified part.
  // This means that when constructing output segments we also need to match
  // this behavior.
  static auto PassThroughQualifiedTypeLoc(clang::TypeLoc type_loc)
      -> clang::TypeLoc {
    auto qtl = type_loc.getAs<clang::QualifiedTypeLoc>();
    return qtl.isNull() ? type_loc : qtl.getUnqualifiedLoc();
  }

  std::variant<std::string, clang::DynTypedNode, clang::TypeLoc> content_;
};

template <typename T, std::enable_if_t<
                          OutputSegment::IsSupportedClangASTNodeType<T>(), int>>
OutputSegment::OutputSegment(const T* node)
    : content_(clang::DynTypedNode::create(AssertNotNull(node))) {}

}  // namespace Carbon

#endif  // CARBON_MIGRATE_CPP_OUTPUT_SEGMENT_H_
