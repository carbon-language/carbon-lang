// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_FUNCTION_DEFINITION_H_
#define EXECUTABLE_SEMANTICS_AST_FUNCTION_DEFINITION_H_

#include "executable_semantics/ast/expression.h"
#include "executable_semantics/ast/statement.h"

namespace Carbon {

// TODO: expand the kinds of things that can be deduced parameters.
//   For now, only generic parameters are supported.
struct GenericBinding {
  std::string name;
  const Expression* type;
};

struct FunctionDefinition {
  int line_num;
  std::string name;
  std::vector<GenericBinding> deduced_parameters;
  const Expression* param_pattern;
  const Expression* return_type;
  const Statement* body;
};

auto MakeFunDef(int line_num, std::string name, const Expression* ret_type,
                std::vector<GenericBinding> deduced_params,
                const Expression* param, const Statement* body)
    -> FunctionDefinition;
void PrintFunDef(const FunctionDefinition&);
void PrintFunDefDepth(const FunctionDefinition&, int);

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_FUNCTION_DEFINITION_H_
