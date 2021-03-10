// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_VARIABLE_DEFINITION_H_
#define EXECUTABLE_SEMANTICS_AST_VARIABLE_DEFINITION_H_

#include <list>
#include <string>

#include "executable_semantics/ast/expression.h"

namespace Carbon {

// AST node for a global variable.
// var <type> : <name> = <initializer>;
struct VariableDefinition {
  int sourceLocation;
  std::string name;
  Expression* type;
  Expression* initializer;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_VARIABLE_DEFINITION_H_
