// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/interpreter/action.h"

#include <iterator>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "executable_semantics/ast/declaration.h"
#include "executable_semantics/ast/expression.h"
#include "executable_semantics/common/arena.h"
#include "executable_semantics/interpreter/stack.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"

namespace Carbon {

using llvm::cast;

Scope::Scope(Scope&& other) noexcept
    : values_(other.values_),
      locals_(std::exchange(other.locals_, {})),
      heap_(other.heap_) {}

auto Scope::operator=(Scope&& rhs) noexcept -> Scope& {
  values_ = rhs.values_;
  locals_ = std::exchange(rhs.locals_, {});
  heap_ = rhs.heap_;
  return *this;
}

Scope::~Scope() {
  for (const auto& l : locals_) {
    std::optional<AllocationId> a = values_.Get(l);
    CHECK(a.has_value());
    heap_->Deallocate(*a);
  }
}

void Action::Print(llvm::raw_ostream& out) const {
  switch (kind()) {
    case Action::Kind::LValAction:
      out << cast<LValAction>(*this).expression();
      break;
    case Action::Kind::ExpressionAction:
      out << cast<ExpressionAction>(*this).expression();
      break;
    case Action::Kind::PatternAction:
      out << cast<PatternAction>(*this).pattern();
      break;
    case Action::Kind::StatementAction:
      cast<StatementAction>(*this).statement().PrintDepth(1, out);
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
