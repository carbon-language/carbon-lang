// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_IMPL_SCOPE_H_
#define EXECUTABLE_SEMANTICS_AST_IMPL_SCOPE_H_

#include "executable_semantics/ast/declaration.h"

namespace Carbon {

class Value;

struct Impl {
  Nonnull<const Value*> interface;
  Nonnull<const Value*> type;
  NamedEntityView impl;
};

class ImplScope {
 public:
  void Add(Nonnull<const Value*> iface, Nonnull<const Value*> type,
           NamedEntityView impl);

  void AddParent(Nonnull<const ImplScope*> parent);

  auto Resolve(Nonnull<const Value*> iface_type, Nonnull<const Value*> type,
               SourceLocation source_loc) const -> NamedEntityView;

 private:
  auto TryResolve(Nonnull<const Value*> iface_type, Nonnull<const Value*> type,
                  SourceLocation source_loc) const
      -> std::optional<NamedEntityView>;
  auto ResolveHere(Nonnull<const Value*> iface_type,
                   Nonnull<const Value*> impl_type,
                   SourceLocation source_loc) const
      -> std::optional<NamedEntityView>;

  std::vector<Impl> impls_;
  std::vector<Nonnull<const ImplScope*>> parent_scopes_;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_IMPL_SCOPE_H_
