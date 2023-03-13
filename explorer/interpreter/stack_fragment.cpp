// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/interpreter/stack_fragment.h"

#include "common/check.h"
#include "llvm/ADT/StringExtras.h"

namespace Carbon {

StackFragment::~StackFragment() {
  CARBON_CHECK(reversed_todo_.empty())
      << "All StackFragments must be empty before the Carbon program ends.";
}

void StackFragment::StoreReversed(
    std::vector<std::unique_ptr<Action>> reversed_todo) {
  CARBON_CHECK(reversed_todo_.empty());
  reversed_todo_ = std::move(reversed_todo);
}

void StackFragment::RestoreTo(Stack<std::unique_ptr<Action>>& todo) {
  while (!reversed_todo_.empty()) {
    todo.Push(std::move(reversed_todo_.back()));
    reversed_todo_.pop_back();
  }
}

void StackFragment::Clear() {
  // We destroy the underlying Actions explicitly to ensure they're
  // destroyed in the correct order.
  for (auto& action : reversed_todo_) {
    action.reset();
  }
  reversed_todo_.clear();
}

void StackFragment::Print(llvm::raw_ostream& out) const {
  out << "{";
  llvm::ListSeparator sep(" :: ");
  for (const std::unique_ptr<Action>& action : reversed_todo_) {
    out << sep << *action;
  }
  out << "}";
}

}  // namespace Carbon
