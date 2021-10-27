// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_DECLARED_NAMES_H_
#define EXECUTABLE_SEMANTICS_AST_DECLARED_NAMES_H_

#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "executable_semantics/common/nonnull.h"

namespace Carbon {

class Declaration;
class BindingPattern;

// In NamedEntity, members include:
// - Declarations, including variables and functions.
// - BindingPattern, for matching contexts.
//
// May want to add ChoiceDeclaration::Alternative, although that would need to
// be refactored for forward declarations.
using NamedEntity =
    std::variant<Nonnull<Declaration*>, Nonnull<BindingPattern*>>;

// The set of declared names in a scope. This is not aware of child scopes, but
// does include directions to parent or related scopes for lookup purposes.
class ScopedNames {
 public:
 private:
  std::unordered_map<std::string, NamedEntity> declared_names_;
  std::vector<Nonnull<ScopedNames*>> parent_scopes_;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_CLASS_DEFINITION_H_
