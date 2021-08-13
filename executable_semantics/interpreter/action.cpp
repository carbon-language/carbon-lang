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

void Action::Print(llvm::raw_ostream& out) const {
  switch (Tag()) {
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
  }
  out << "<" << pos << ">";
  if (results.size() > 0) {
    out << "(";
    llvm::ListSeparator sep;
    for (auto& result : results) {
      out << sep;
      if (result) {
        out << *result;
      }
    }
    out << ")";
  }
}

void Action::PrintList(const Stack<Action*>& ls, llvm::raw_ostream& out) {
  llvm::ListSeparator sep(" :: ");
  for (const auto& action : ls) {
    out << sep << *action;
  }
}

}  // namespace Carbon
