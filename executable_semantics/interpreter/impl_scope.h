// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_IMPL_SCOPE_H_
#define EXECUTABLE_SEMANTICS_AST_IMPL_SCOPE_H_

#include "executable_semantics/ast/declaration.h"

namespace Carbon {

class Value;

// The `Impl` struct is a key-value pair where the key is the
// combination of a type and an interface, e.g., `List` and `Container`,
// and the value is the result of statically resolving to the `impl`
// for `List` as `Container`, which is an `EntityView`. The generality
// of `EntityView` is needed (not just `ImplDeclaration`) because
// inside a generic, we need to map, e.g., `T` and `Container` to the
// witness table that is passed into the generic.
struct Impl {
  Nonnull<const Value*> interface;
  Nonnull<const Value*> type;
  EntityView impl;
};

// Maps a type and interface to the location of the witness table for
// the `impl` for that type and interface.
// A scope may have parent scopes, whose names will also be visible in
// the child scope.
class ImplScope {
 public:
  // Associates `iface` and `type` with the `impl` in this scope.
  void Add(Nonnull<const Value*> iface, Nonnull<const Value*> type,
           EntityView impl);

  // Make `parent` a parent of this scope.
  // REQUIRES: `parent` is not already a parent of this scope.
  void AddParent(Nonnull<const ImplScope*> parent);

  // Returns the associated impl for the given `iface` and `type` in
  // the ancestor graph of this scope, or reports a compilation error
  // at `source_loc` there isn't exactly one matching impl.
  auto Resolve(Nonnull<const Value*> iface, Nonnull<const Value*> type,
               SourceLocation source_loc) const -> EntityView;

 private:
  auto TryResolve(Nonnull<const Value*> iface_type, Nonnull<const Value*> type,
                  SourceLocation source_loc) const -> std::optional<EntityView>;
  auto ResolveHere(Nonnull<const Value*> iface_type,
                   Nonnull<const Value*> impl_type,
                   SourceLocation source_loc) const
      -> std::optional<EntityView>;

  std::vector<Impl> impls_;
  std::vector<Nonnull<const ImplScope*>> parent_scopes_;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_IMPL_SCOPE_H_
