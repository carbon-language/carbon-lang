// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_STATIC_SCOPE_H_
#define EXECUTABLE_SEMANTICS_AST_STATIC_SCOPE_H_

#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "executable_semantics/ast/source_location.h"
#include "executable_semantics/common/nonnull.h"

namespace Carbon {

class NamedEntityInterface {
 public:
  enum class NamedEntityKind {
    // Includes variable definitions and matching contexts.
    BindingPattern,
    // Used by entries in choices.
    ChoiceDeclarationAlternative,
    // Used by continuations.
    Continuation,
    // Includes choices, classes, and functions. Variables are handled through
    // BindingPattern.
    Declaration,
    // Used by functions.
    GenericBinding,
    // Used by entries in classes.
    Member,
  };

  NamedEntityInterface() = default;
  virtual ~NamedEntityInterface() = default;

  NamedEntityInterface(NamedEntityInterface&&) = delete;
  auto operator=(NamedEntityInterface&&) -> NamedEntityInterface& = delete;

  // TODO: This is unused, but is intended for casts after lookup.
  virtual auto named_entity_kind() const -> NamedEntityKind = 0;
  virtual auto source_loc() const -> SourceLocation = 0;
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
