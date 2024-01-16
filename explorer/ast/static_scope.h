// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_AST_STATIC_SCOPE_H_
#define CARBON_EXPLORER_AST_STATIC_SCOPE_H_

#include <string>
#include <string_view>

#include "common/error.h"
#include "explorer/ast/value_node.h"
#include "explorer/base/nonnull.h"
#include "explorer/base/source_location.h"
#include "explorer/base/trace_stream.h"
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
  explicit StaticScope(Nonnull<TraceStream*> trace_stream)
      : ast_node_(std::nullopt), trace_stream_(trace_stream) {}

  // Construct a scope that is nested within the given scope.
  explicit StaticScope(Nonnull<const StaticScope*> parent,
                       std::optional<Nonnull<const AstNode*>> ast_node)
      : parent_scope_(parent),
        ast_node_(ast_node),
        trace_stream_(parent->trace_stream_) {}

  StaticScope() = default;

  // Defines `name` to be `entity` in this scope, or reports a compilation error
  // if `name` is already defined to be a different entity in this scope.
  // If `usable` is `false`, `name` cannot yet be referenced and `Resolve()`
  // methods will fail for it.
  auto Add(std::string_view name, ValueNodeView entity,
           NameStatus status = NameStatus::Usable) -> ErrorOr<Success>;

  template <typename Action>
  void PrintCommon(Action action) const;

  void Print(llvm::raw_ostream& out) const;

  void PrintID(llvm::raw_ostream& out) const;

  // Marks `name` as being past its point of declaration.
  void MarkDeclared(std::string_view name);
  // Marks `name` as being completely declared and hence usable.
  void MarkUsable(std::string_view name);

  // Returns the nearest declaration of `name` in the ancestor graph of this
  // scope, or reports a compilation error at `source_loc` there isn't such a
  // declaration.
  // TODO: This should also diagnose if there's a shadowed declaration of the
  // name in an enclosing scope, but does not do so yet.
  auto Resolve(std::string_view name, SourceLocation source_loc) const
      -> ErrorOr<ValueNodeView>;

  // Returns the declaration of `name` in this scope, or reports a compilation
  // error at `source_loc` if the name is not declared in this scope. If
  // `allow_undeclared` is `true`, names that have been added but not yet marked
  // declared or usable do not result in an error.
  auto ResolveHere(std::optional<ValueNodeView> this_scope,
                   std::string_view name, SourceLocation source_loc,
                   bool allow_undeclared) const -> ErrorOr<ValueNodeView>;

  // Returns the value node of the BindingPattern of the returned var definition
  // if it exists in the ancestor graph.
  auto ResolveReturned() const -> std::optional<ValueNodeView>;

  // Adds the value node of the BindingPattern of the returned var definition to
  // this scope. Returns a compilation error when there is an existing returned
  // var in the ancestor graph.
  auto AddReturnedVar(ValueNodeView returned_var_def_view) -> ErrorOr<Success>;

 private:
  // Equivalent to Resolve, but returns `nullopt` instead of raising an error
  // if no declaration can be found.
  auto TryResolve(std::string_view name, SourceLocation source_loc) const
      -> ErrorOr<std::optional<ValueNodeView>>;

  // Equivalent to ResolveHere, but returns `nullopt` if no definition can be
  // found. Raises an error if the name is found but is not usable yet.
  auto TryResolveHere(std::string_view name, SourceLocation source_loc,
                      bool allow_undeclared) const
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

  std::optional<Nonnull<const AstNode*>> ast_node_;

  Nonnull<TraceStream*> trace_stream_;
};

}  // namespace Carbon

#endif  // CARBON_EXPLORER_AST_STATIC_SCOPE_H_
