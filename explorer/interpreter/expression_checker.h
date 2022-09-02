// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_INTERPRETER_EXPRESSION_CHECKER_H_
#define CARBON_EXPLORER_INTERPRETER_EXPRESSION_CHECKER_H_

#include "common/error.h"
#include "explorer/ast/expression.h"
#include "explorer/interpreter/impl_scope.h"

namespace Carbon {

class ExpressionChecker {
 public:
  explicit ExpressionChecker(
      Nonnull<Arena*> arena,
      std::optional<Nonnull<llvm::raw_ostream*>> trace_stream)
      : arena_(arena), trace_stream_(trace_stream) {}

  // Traverses the AST rooted at `e`, populating the static_type() of all nodes
  // and ensuring they follow Carbon's typing rules.
  //
  // `values` maps variable names to their compile-time values. It is not
  //    directly used in this function but is passed to InterpExp.
  auto TypeCheckExp(Nonnull<Expression*> e, const ImplScope& impl_scope)
      -> ErrorOr<Success>;

 private:
  Nonnull<Arena*> arena_;
  std::optional<Nonnull<llvm::raw_ostream*>> trace_stream_;
};

}  // namespace Carbon

#endif  // CARBON_EXPLORER_INTERPRETER_EXPRESSION_CHECKER_H_