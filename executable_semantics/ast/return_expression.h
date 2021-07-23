// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_RETURN_EXPRESSION_
#define EXECUTABLE_SEMANTICS_AST_RETURN_EXPRESSION_

#include "executable_semantics/ast/expression.h"

namespace Carbon {

struct ReturnExpression {
  enum class Kind {
    // For example, `return 3;` explicitly returns `3`.
    Explicit,
    // For example, `return;` implicitly returns `()`.
    Implicit,
  };

  ReturnExpression(int line_num, const Expression* exp)
      : kind(exp != nullptr ? Kind::Explicit : Kind::Implicit),
        exp(exp != nullptr
                ? exp
                : Carbon::Expression::MakeTupleLiteral(line_num, {})) {}

  // Indicates the expression kind used by the return.
  Kind kind;
  // The expression for the return.
  const Expression* exp;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_FUNCTION_DEFINITION_H_
