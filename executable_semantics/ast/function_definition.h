// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_FUNCTION_DEFINITION_H_
#define EXECUTABLE_SEMANTICS_AST_FUNCTION_DEFINITION_H_

#include "common/ostream.h"
#include "executable_semantics/ast/expression.h"
#include "executable_semantics/ast/pattern.h"
#include "executable_semantics/ast/statement.h"
#include "llvm/Support/Compiler.h"

namespace Carbon {

// TODO: expand the kinds of things that can be deduced parameters.
//   For now, only generic parameters are supported.
struct GenericBinding {
  std::string name;
  const Expression* type;
};

struct FunctionDefinition {
  FunctionDefinition() = default;
  FunctionDefinition(int line_num, std::string name,
                     std::vector<GenericBinding> deduced_params,
                     const TuplePattern* param_pattern,
                     const Pattern* return_type, bool is_omitted_return_type,
                     const Statement* body)
      : line_num(line_num),
        name(std::move(name)),
        deduced_parameters(deduced_params),
        param_pattern(param_pattern),
        return_type(return_type),
        is_omitted_return_type(is_omitted_return_type),
        body(body) {}

  void Print(llvm::raw_ostream& out) const { PrintDepth(-1, out); }
  void PrintDepth(int depth, llvm::raw_ostream& out) const;
  LLVM_DUMP_METHOD void Dump() const { Print(llvm::errs()); }

  int line_num;
  std::string name;
  std::vector<GenericBinding> deduced_parameters;
  const TuplePattern* param_pattern;
  const Pattern* return_type;
  bool is_omitted_return_type;
  const Statement* body;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_FUNCTION_DEFINITION_H_
