// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_FUNCTION_DEFINITION_H_
#define EXECUTABLE_SEMANTICS_AST_FUNCTION_DEFINITION_H_

#include "executable_semantics/ast/expression.h"
#include "executable_semantics/ast/statement.h"

namespace Carbon {

struct FunctionDefinition {
  int line_num;
  std::string name;
  Expression* param_pattern;
  Expression* return_type;
  Statement* body;
};

auto MakeFunDef(int line_num, std::string name, Expression* ret_type,
                Expression* param, Statement* body)
    -> struct FunctionDefinition*;
void PrintFunDef(struct FunctionDefinition*);
void PrintFunDefDepth(struct FunctionDefinition*, int);

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_FUNCTION_DEFINITION_H_
