// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/interpreter/action.h"

#include <iterator>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "executable_semantics/ast/expression.h"
#include "executable_semantics/ast/function_definition.h"
#include "executable_semantics/common/arena.h"
#include "executable_semantics/interpreter/stack.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"

namespace Carbon {

using llvm::cast;

auto Action::ast_node() const -> const void* {
  switch (kind()) {
    case Action::Kind::LValAction:
      return cast<LValAction>(*this).Exp();
    case Action::Kind::ExpressionAction:
      return cast<ExpressionAction>(*this).Exp();
    case Action::Kind::PatternAction:
      return cast<PatternAction>(*this).Pat();
    case Action::Kind::StatementAction:
      return cast<StatementAction>(*this).Stmt();
    case Action::Kind::ScopeAction:
      return nullptr;
  }
}

void Action::Print(llvm::raw_ostream& out) const {
  switch (kind()) {
    case Action::Kind::LValAction:
      out << *cast<LValAction>(*this).Exp();
      break;
    case Action::Kind::ExpressionAction:
      out << *cast<ExpressionAction>(*this).Exp();
      break;
    case Action::Kind::PatternAction:
      out << *cast<PatternAction>(*this).Pat();
      break;
    case Action::Kind::StatementAction:
      cast<StatementAction>(*this).Stmt()->PrintDepth(1, out);
      break;
    case Action::Kind::ScopeAction:
      out << "ScopeAction";
  }
  out << "<" << pos_ << ">";
  if (results_.size() > 0) {
    out << "(";
    llvm::ListSeparator sep;
    for (auto& result : results_) {
      out << sep << *result;
    }
    out << ")";
  }
}

void Action::PrintList(const Stack<Nonnull<Action*>>& ls,
                       llvm::raw_ostream& out) {
  llvm::ListSeparator sep(" :: ");
  for (const auto& action : ls) {
    out << sep << *action;
  }
}

}  // namespace Carbon
