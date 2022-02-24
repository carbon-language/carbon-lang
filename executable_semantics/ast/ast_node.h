// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_AST_NODE_H_
#define EXECUTABLE_SEMANTICS_AST_AST_NODE_H_

#include "executable_semantics/ast/ast_rtti.h"
#include "executable_semantics/ast/source_location.h"
#include "llvm/Support/Casting.h"

namespace Carbon {

// Base class for all nodes in the AST.
//
// Every class derived from this class must be listed in ast_rtti.txt. See
// the documentation of gen_rtti.py for details about the format. As a result,
// every abstract class `Foo` will have a `FooKind` enumerated type, whose
// enumerators correspond to the subclasses of `Foo`.
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
// ast_rtti.txt, and are implicitly provided by this header.
//
// TODO: To support generic traversal, add children() method, and ensure that
//   all AstNodes are reachable from a root AstNode.
class AstNode {
 public:
  AstNode(AstNode&&) = delete;
  auto operator=(AstNode&&) -> AstNode& = delete;
  virtual ~AstNode() = 0;

  virtual void Print(llvm::raw_ostream& out) const = 0;
  LLVM_DUMP_METHOD void Dump() const { Print(llvm::errs()); }

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

  // Equivalent to kind(), but will not be hidden by `kind()` methods of
  // derived classes.
  auto root_kind() const -> AstNodeKind { return kind_; }

 private:
  AstNodeKind kind_;
  SourceLocation source_loc_;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_AST_NODE_H_
