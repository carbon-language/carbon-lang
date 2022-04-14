// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPLORER_AST_IMPL_SCOPE_H_
#define EXPLORER_AST_IMPL_SCOPE_H_

#include "explorer/ast/declaration.h"

namespace Carbon {

class Value;

// The `ImplScope` class is responsible for mapping a type and
// interface to the location of the witness table for the `impl` for
// that type and interface.  A scope may have parent scopes, whose
// impls will also be visible in the child scope.
//
// There is typically one instance of `ImplScope` class per scope
// because the impls that are visible for a given type and interface
// can vary from scope to scope. For example, consider the `bar` and
// `baz` methods in the following class C and nested class D.
//
//     class C(U:! Type, T:! Type)  {
//       class D(V:! Type where U is Fooable(T)) {
//         fn bar[me: Self](x: U, y : T) -> T{
//           return x.foo(y)
//         }
//       }
//       fn baz[me: Self](x: U, y : T) -> T {
//         return x.foo(y);
//       }
//     }
//
//  The call to `x.foo` in `bar` is valid because the `U is Fooable(T)`
//  impl is visible in the body of `bar`. In contrast, the call to
//  `x.foo` in `baz` is not valid because there is no visible impl for
//  `U` and `Fooable` in that scope.
class ImplScope {
 public:
  // Associates `iface` and `type` with the `impl` in this scope.
  void Add(Nonnull<const Value*> iface, Nonnull<const Value*> type,
           ValueNodeView impl);

  // Make `parent` a parent of this scope.
  // REQUIRES: `parent` is not already a parent of this scope.
  void AddParent(Nonnull<const ImplScope*> parent);

  // Returns the associated impl for the given `iface` and `type` in
  // the ancestor graph of this scope, or reports a compilation error
  // at `source_loc` there isn't exactly one matching impl.
  auto Resolve(Nonnull<const Value*> iface, Nonnull<const Value*> type,
               SourceLocation source_loc) const -> ErrorOr<ValueNodeView>;

  void Print(llvm::raw_ostream& out) const;

 private:
  auto TryResolve(Nonnull<const Value*> iface_type, Nonnull<const Value*> type,
                  SourceLocation source_loc) const
      -> ErrorOr<std::optional<ValueNodeView>>;
  auto ResolveHere(Nonnull<const Value*> iface_type,
                   Nonnull<const Value*> impl_type,
                   SourceLocation source_loc) const
      -> std::optional<ValueNodeView>;

  // The `Impl` struct is a key-value pair where the key is the
  // combination of a type and an interface, e.g., `List` and `Container`,
  // and the value is the result of statically resolving to the `impl`
  // for `List` as `Container`, which is an `ValueNodeView`. The generality
  // of `ValueNodeView` is needed (not just `ImplDeclaration`) because
  // inside a generic, we need to map, e.g., from `T` and `Container` to the
  // witness table that is passed into the generic.
  struct Impl {
    Nonnull<const Value*> interface;
    Nonnull<const Value*> type;
    ValueNodeView impl;
  };

  std::vector<Impl> impls_;
  std::vector<Nonnull<const ImplScope*>> parent_scopes_;
};

}  // namespace Carbon

#endif  // EXPLORER_AST_IMPL_SCOPE_H_
