// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_STATIC_SCOPE_H_
#define EXECUTABLE_SEMANTICS_AST_STATIC_SCOPE_H_

#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "executable_semantics/ast/ast_node.h"
#include "executable_semantics/ast/source_location.h"
#include "executable_semantics/common/nonnull.h"

namespace Carbon {

class NamedEntityInterface : public virtual AstNode {
 public:
  virtual ~NamedEntityInterface() = 0;

  NamedEntityInterface() = default;

  // TODO: This is unused, but is intended for casts after lookup.
  auto kind() const -> NamedEntityInterfaceKind {
    return static_cast<NamedEntityInterfaceKind>(root_kind());
  }
};

// The set of declared names in a scope. This is not aware of child scopes, but
// does include directions to parent or related scopes for lookup purposes.
class StaticScope {
 public:
  void Add(std::string name, Nonnull<const NamedEntityInterface*> entity);

 private:
  // Maps locally declared names to their entities.
  std::unordered_map<std::string, Nonnull<const NamedEntityInterface*>>
      declared_names_;

  // A list of scopes used for name lookup within this scope.
  // TODO: This is unused, but is intended for name lookup cross-scope.
  std::vector<Nonnull<StaticScope*>> parent_scopes_;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_STATIC_SCOPE_H_
