// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_SCOPED_NAMES_H_
#define EXECUTABLE_SEMANTICS_AST_SCOPED_NAMES_H_

#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "executable_semantics/ast/declaration.h"
#include "executable_semantics/ast/source_location.h"
#include "executable_semantics/ast/statement.h"
#include "executable_semantics/common/nonnull.h"

namespace Carbon {

// NamedEntity includes:
// - BindingPattern, including variable definitions and matching contexts.
// - ChoiceDeclaration::Alternative, for entries in choices.
// - Declarations, including choices, classes, and functions.
//   - Variables are handled through BindingPattern.
// - GenericBinding, for functions.
// - Member, for entries in classes.
// - Statements, including continuations.
using NamedEntity =
    std::variant<Nonnull<const BindingPattern*>,
                 Nonnull<const ChoiceDeclaration::Alternative*>,
                 Nonnull<const Declaration*>, Nonnull<const GenericBinding*>,
                 Nonnull<const Member*>, Nonnull<const Statement*>>;

// The set of declared names in a scope. This is not aware of child scopes, but
// does include directions to parent or related scopes for lookup purposes.
class ScopedNames {
 public:
  void Add(std::string name, NamedEntity entity);

 private:
  std::unordered_map<std::string, NamedEntity> declared_names_;
  std::vector<Nonnull<ScopedNames*>> parent_scopes_;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_SCOPED_NAMES_H_
