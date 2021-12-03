// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_STATIC_SCOPE_H_
#define EXECUTABLE_SEMANTICS_AST_STATIC_SCOPE_H_

#include <functional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "executable_semantics/ast/ast_node.h"
#include "executable_semantics/ast/source_location.h"
#include "executable_semantics/common/nonnull.h"

namespace Carbon {

class Value;

// Non-owning type-erased wrapper around a const NodeType* `node`, where
// NodeType models the NamedEntity interface. This means that:
//
// - node->static_type() is well-formed and has type const Value&.
// - ImplementsNamedEntity<NodeType> is true. In other words,
//   NodeType must directly or indirectly implement NamedEntity in ast_rtti.txt.
class NamedEntityView {
 public:
  template <typename NodeType,
            typename = std::enable_if_t<ImplementsNamedEntity<NodeType>>>
  NamedEntityView(Nonnull<const NodeType*> node)
      : base_(node),
        static_type_([node]() -> const Value& { return node->static_type(); }) {
  }

  NamedEntityView(const NamedEntityView&) = default;
  NamedEntityView(NamedEntityView&&) = default;
  auto operator=(const NamedEntityView&) -> NamedEntityView& = default;
  auto operator=(NamedEntityView&&) -> NamedEntityView& = default;

  // Returns `node` as an instance of the base class AstNode.
  auto base() const -> const AstNode& { return *base_; }

  auto kind() const -> NamedEntityKind {
    return static_cast<NamedEntityKind>(base_->kind());
  }

  // Returns node->static_type()
  auto static_type() const -> const Value& { return static_type_(); }

  friend auto operator==(const NamedEntityView& lhs,
                         const NamedEntityView& rhs) {
    return lhs.base_ == rhs.base_;
  }

  friend auto operator!=(const NamedEntityView& lhs,
                         const NamedEntityView& rhs) {
    return lhs.base_ != rhs.base_;
  }

 private:
  Nonnull<const AstNode*> base_;
  std::function<const Value&()> static_type_;
};

// Maps the names visible in a given scope to the entities they name.
// A scope may have parent scopes, whose names will also be visible in the
// child scope.
class StaticScope {
 public:
  // Defines `name` to be `entity` in this scope, or reports a compilation error
  // if `name` is already defined to be a different entity in this scope.
  void Add(std::string name, NamedEntityView entity);

  // Make `parent` a parent of this scope.
  // REQUIRES: `parent` is not already a parent of this scope.
  void AddParent(Nonnull<StaticScope*> parent) {
    parent_scopes_.push_back(parent);
  }

  // Returns the nearest definition of `name` in the ancestor graph of this
  // scope, or reports a compilation error at `source_loc` there isn't exactly
  // one such definition.
  auto Resolve(const std::string& name, SourceLocation source_loc) const
      -> NamedEntityView;

 private:
  // Equivalent to Resolve, but returns `nullopt` instead of raising an error
  // if no definition can be found. Still raises a compilation error if more
  // than one definition is found.
  auto TryResolve(const std::string& name, SourceLocation source_loc) const
      -> std::optional<NamedEntityView>;

  // Maps locally declared names to their entities.
  std::unordered_map<std::string, NamedEntityView> declared_names_;

  // A list of scopes used for name lookup within this scope.
  std::vector<Nonnull<StaticScope*>> parent_scopes_;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_STATIC_SCOPE_H_
