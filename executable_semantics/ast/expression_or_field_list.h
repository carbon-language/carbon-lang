// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_EXPRESSION_OR_FIELD_LIST_H_
#define EXECUTABLE_SEMANTICS_AST_EXPRESSION_OR_FIELD_LIST_H_

#include <list>

#include "executable_semantics/ast/expression.h"

namespace Carbon {

enum class ExpOrFieldListKind { Exp, FieldList };

// This is used in the parsing of tuples and parenthesized expressions.
struct ExpOrFieldList {
  ExpOrFieldListKind tag;
  union {
    Expression* exp;
    std::list<std::pair<std::string, Expression*>>* fields;
  } u;
};

auto MakeExp(Expression* exp) -> ExpOrFieldList*;
auto MakeFieldList(std::list<std::pair<std::string, Expression*>>* fields)
    -> ExpOrFieldList*;
auto MakeConsField(ExpOrFieldList* e1, ExpOrFieldList* e2) -> ExpOrFieldList*;

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_EXPRESSION_OR_FIELD_LIST_H_
