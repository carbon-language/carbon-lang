// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_ABSTRACT_SYNTAX_TREE_H_
#define EXECUTABLE_SEMANTICS_AST_ABSTRACT_SYNTAX_TREE_H_

#include <variant>

#include "executable_semantics/ast/declaration.h"

namespace Carbon {
using AST = std::list<const Carbon::Declaration*>;
}

#endif  // EXECUTABLE_SEMANTICS_AST_ABSTRACT_SYNTAX_TREE_H_
