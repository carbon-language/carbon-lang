// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_FIELD_LIST_H_
#define EXECUTABLE_SEMANTICS_AST_FIELD_LIST_H_

#include <list>

#include "executable_semantics/ast/expression.h"

namespace Carbon {

// This is used in the parsing of tuples and parenthesized expressions.
struct FieldList {
  std::list<std::pair<std::string, Expression*>>* fields;
  bool has_explicit_comma = false;
};

auto MakeFieldList(std::list<std::pair<std::string, Expression*>>* fields)
    -> FieldList*;
auto MakeConsField(FieldList* e1, FieldList* e2) -> FieldList*;

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_FIELD_LIST_H_
