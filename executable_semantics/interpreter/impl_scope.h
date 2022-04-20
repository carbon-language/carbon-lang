// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_IMPL_SCOPE_H_
#define EXECUTABLE_SEMANTICS_AST_IMPL_SCOPE_H_

#include "executable_semantics/ast/declaration.h"

namespace Carbon {

class Value;
class TypeChecker;
class InterfaceType;

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
           Nonnull<Expression*> impl);
  // For a parameterized impl, associates `iface` and `type`
  // with the `impl` in this scope.
  void Add(Nonnull<const Value*> iface,
           llvm::ArrayRef<Nonnull<const GenericBinding*>> deduced,
           Nonnull<const Value*> type,
           llvm::ArrayRef<Nonnull<const ImplBinding*>> impl_bindings,
           Nonnull<Expression*> impl);

  // Make `parent` a parent of this scope.
  // REQUIRES: `parent` is not already a parent of this scope.
  void AddParent(Nonnull<const ImplScope*> parent);

  // Returns the associated impl for the given `iface` and `type` in
  // the ancestor graph of this scope, or reports a compilation error
  // at `source_loc` there isn't exactly one matching impl.
  auto Resolve(Nonnull<const Value*> iface, Nonnull<const Value*> type,
               SourceLocation source_loc, const TypeChecker& type_checker) const
      -> ErrorOr<Nonnull<Expression*>>;

  void Print(llvm::raw_ostream& out) const;

  // The `Impl` struct is a key-value pair where the key is the
  // combination of a type and an interface, e.g., `List` and `Container`,
  // and the value is the result of statically resolving to the `impl`
  // for `List` as `Container`, which is an `Expression` that produces
  // the witness for that `impl`.
  // When the `impl` is parameterized, `deduced` and `impl_bindings`
  // are non-empty. The former contains the type parameters and the
  // later are impl bindings, that is, parameters for witnesses.
  struct Impl {
    Nonnull<const Value*> interface;
    std::vector<Nonnull<const GenericBinding*>> deduced;
    Nonnull<const Value*> type;
    std::vector<Nonnull<const ImplBinding*>> impl_bindings;
    Nonnull<Expression*> impl;
  };

 private:
  // Returns the associated impl for the given `iface` and `type` in
  // the ancestor graph of this scope, returns std::nullopt if there
  // is none, or reports a compilation error is there is not a most
  // specific impl for the given `iface` and `type`.
  // Use `original_scope` to satisfy requirements of any generic impl
  // that matches `iface` and `type`.
  auto TryResolve(Nonnull<const Value*> iface, Nonnull<const Value*> type,
                  SourceLocation source_loc, const ImplScope& original_scope,
                  const TypeChecker& type_checker) const
      -> ErrorOr<std::optional<Nonnull<Expression*>>>;

  // Returns the associated impl for the given `iface` and `type` in
  // this scope, returns std::nullopt if there is none, or reports
  // a compilation error is there is not a most specific impl for the
  // given `iface` and `type`.
  // Use `original_scope` to satisfy requirements of any generic impl
  // that matches `iface` and `type`.
  auto ResolveHere(Nonnull<const Value*> iface_type,
                   Nonnull<const Value*> impl_type, SourceLocation source_loc,
                   const ImplScope& original_scope,
                   const TypeChecker& type_checker) const
      -> ErrorOr<std::optional<Nonnull<Expression*>>>;

  std::vector<Impl> impls_;
  std::vector<Nonnull<const ImplScope*>> parent_scopes_;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_IMPL_SCOPE_H_
