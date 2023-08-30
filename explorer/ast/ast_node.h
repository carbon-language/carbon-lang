// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_AST_AST_NODE_H_
#define CARBON_EXPLORER_AST_AST_NODE_H_

#include "explorer/ast/ast_rtti.h"
#include "explorer/base/source_location.h"
#include "llvm/Support/Casting.h"

namespace Carbon {

class CloneContext;

// Base class for all nodes in the AST.
//
// Every class derived from this class must be listed in ast_kinds.h. As a
// result, every abstract class `Foo` will have a `FooKind` enumerated type,
// whose enumerators correspond to the subclasses of `Foo`.
//
// AstNode and its derived classes support LLVM-style RTTI, including
// llvm::isa, llvm::cast, and llvm::dyn_cast. To support this, every
// class derived from Declaration must provide a `classof` operation, with
// the following form, where `Foo` is the name of the derived class:
//
// static auto classof(const AstNode* node) -> bool {
//   return InheritsFromFoo(node->kind());
// }
//
// Furthermore, if the class is abstract, it must provide a `kind()` operation,
// with the following form:
//
// auto kind() const -> FooKind { return static_cast<FooKind>(root_kind()); }
//
// The definitions of `InheritsFromFoo` and `FooKind` are generated from
// ast_rtti.h, and are implicitly provided by this header.
//
// Every AST node is expected to provide a cloning constructor:
//
// explicit MyAstNode(CloneContext& context, const MyAstNode& other);
//
// The cloning constructor should behave like a copy constructor, but pointers
// to other AST nodes should be passed through context.Clone to clone the
// referenced object.
//
// TODO: To support generic traversal, add children() method, and ensure that
//   all AstNodes are reachable from a root AstNode.
class AstNode : public Printable<AstNode> {
 public:
  AstNode(AstNode&&) = delete;
  auto operator=(AstNode&&) -> AstNode& = delete;
  virtual ~AstNode() = 0;

  // Print the AST rooted at the node.
  virtual void Print(llvm::raw_ostream& out) const = 0;
  // Print identifying information about the node, such as its name.
  virtual void PrintID(llvm::raw_ostream& out) const = 0;

  // Returns an enumerator specifying the concrete type of this node.
  //
  // Abstract subclasses of AstNode will provide their own `kind()` method
  // which hides this one, and provides a narrower return type.
  auto kind() const -> AstNodeKind { return kind_; }

  // The location of the code described by this node.
  auto source_loc() const -> SourceLocation { return source_loc_; }

 protected:
  // Constructs an AstNode representing code at the given location. `kind`
  // must be the enumerator that exactly matches the concrete type being
  // constructed.
  explicit AstNode(AstNodeKind kind, SourceLocation source_loc)
      : kind_(kind), source_loc_(source_loc) {}

  // Clone this AstNode.
  explicit AstNode(CloneContext& /*context*/, const AstNode& other)
      : kind_(other.kind_), source_loc_(other.source_loc_) {}

  // Equivalent to kind(), but will not be hidden by `kind()` methods of
  // derived classes.
  auto root_kind() const -> AstNodeKind { return kind_; }

 private:
  AstNodeKind kind_;
  SourceLocation source_loc_;
};

}  // namespace Carbon

#endif  // CARBON_EXPLORER_AST_AST_NODE_H_
