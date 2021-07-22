// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_FUNCTION_DEFINITION_H_
#define EXECUTABLE_SEMANTICS_AST_FUNCTION_DEFINITION_H_

#include "common/ostream.h"
#include "executable_semantics/ast/expression.h"
#include "executable_semantics/ast/statement.h"

namespace Carbon {

struct FunctionDefinition {
  FunctionDefinition() = default;
  FunctionDefinition(int line_num, std::string name,
                     const Expression* param_pattern,
                     const Expression* return_type, const Statement* body)
      : line_num(line_num),
        name(std::move(name)),
        param_pattern(param_pattern),
        return_type(return_type),
        body(body) {}

  void Print(llvm::raw_ostream& out) const { PrintDepth(-1, out); }
  void PrintDepth(int depth, llvm::raw_ostream& out) const;

  int line_num;
  std::string name;
  const Expression* param_pattern;
  const Expression* return_type;
  const Statement* body;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_FUNCTION_DEFINITION_H_
