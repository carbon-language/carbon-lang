// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_AST_STATIC_SCOPE_H_
#define CARBON_EXPLORER_AST_STATIC_SCOPE_H_

#include <string>
#include <string_view>

#include "common/error.h"
#include "explorer/ast/value_node.h"
#include "explorer/common/nonnull.h"
#include "explorer/common/source_location.h"
#include "llvm/ADT/StringMap.h"

namespace Carbon {

// Maps the names visible in a given scope to the entities they name.
// A scope may have parent scopes, whose names will also be visible in the
// child scope.
class StaticScope {
 public:
  // The status of a name. Later enumerators with higher values correspond to
  // more completely declared names.
  enum class NameStatus {
    // The name is known to exist in this scope, and any lookups finding it
    // should be rejected because it's not declared yet.
    KnownButNotDeclared,
    // We've started processing a declaration of this name, but it's not yet
    // fully declared, so any lookups finding it should be rejected.
    DeclaredButNotUsable,
    // The name is usable in this context.
    Usable,
  };

  // Construct a root scope.
  StaticScope() = default;

  // Construct a scope that is nested within the given scope.
  explicit StaticScope(Nonnull<const StaticScope*> parent)
      : parent_scope_(parent) {}

  // Defines `name` to be `entity` in this scope, or reports a compilation error
  // if `name` is already defined to be a different entity in this scope.
  // If `usable` is `false`, `name` cannot yet be referenced and `Resolve()`
  // methods will fail for it.
  auto Add(std::string_view name, ValueNodeView entity,
           NameStatus status = NameStatus::Usable) -> ErrorOr<Success>;

  // Marks `name` as being past its point of declaration.
  void MarkDeclared(std::string_view name);
  // Marks `name` as being completely declared and hence usable.
  void MarkUsable(std::string_view name);

  // Returns the nearest definition of `name` in the ancestor graph of this
  // scope, or reports a compilation error at `source_loc` there isn't exactly
  // one such definition.
  auto Resolve(std::string_view name, SourceLocation source_loc) const
      -> ErrorOr<ValueNodeView>;

  // Returns the value node of the BindingPattern of the returned var definition
  // if it exists in the ancestor graph.
  auto ResolveReturned() const -> std::optional<ValueNodeView>;

  // Adds the value node of the BindingPattern of the returned var definition to
  // this scope. Returns a compilation error when there is an existing returned
  // var in the ancestor graph.
  auto AddReturnedVar(ValueNodeView returned_var_def_view) -> ErrorOr<Success>;

 private:
  // Equivalent to Resolve, but returns `nullopt` instead of raising an error
  // if no definition can be found. Still raises a compilation error if more
  // than one definition is found.
  auto TryResolve(std::string_view name, SourceLocation source_loc) const
      -> ErrorOr<std::optional<ValueNodeView>>;

  struct Entry {
    ValueNodeView entity;
    NameStatus status;
  };
  // Maps locally declared names to their entities.
  llvm::StringMap<Entry> declared_names_;

  // The parent scope of this scope, if it not the root scope.
  std::optional<Nonnull<const StaticScope*>> parent_scope_;

  // Stores the value node of the BindingPattern of the returned var definition.
  std::optional<ValueNodeView> returned_var_def_view_;
};

}  // namespace Carbon

#endif  // CARBON_EXPLORER_AST_STATIC_SCOPE_H_
