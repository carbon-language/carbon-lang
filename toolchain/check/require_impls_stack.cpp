// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/require_impls_stack.h"

namespace Carbon::Check {

auto RequireImplsStack::Push(EnclosingScopeId scope_id) -> void {
  scope_ids_.push_back(scope_id);
  array_stack_.PushArray();
}

auto RequireImplsStack::Pop() -> void {
  scope_ids_.pop_back();
  array_stack_.PopArray();
}

auto RequireImplsStack::AppendToTop(SemIR::RequireImplsId id) -> void {
  array_stack_.AppendToTop(id);
}

auto RequireImplsStack::PeekTop() const
    -> llvm::ArrayRef<SemIR::RequireImplsId> {
  return array_stack_.PeekArray();
}

auto RequireImplsStack::PeekForScope(EnclosingScopeId scope_id)
    -> llvm::ArrayRef<SemIR::RequireImplsId> {
  auto it = std::find(scope_ids_.rbegin(), scope_ids_.rend(), scope_id);
  CARBON_CHECK(it != scope_ids_.rend());
  auto index = std::distance(it, scope_ids_.rend()) - 1;
  return array_stack_.PeekArrayAt(index);
}

}  // namespace Carbon::Check
