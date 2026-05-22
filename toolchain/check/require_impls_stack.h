// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_REQUIRE_IMPLS_STACK_H_
#define CARBON_TOOLCHAIN_CHECK_REQUIRE_IMPLS_STACK_H_

#include "common/array_stack.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// A stack where each frame holds an array of RequireImplsIds and is associated
// with an enclosing scope that can be searched for in the stack.
class RequireImplsStack {
 public:
  using EnclosingScopeId =
      std::variant<SemIR::InterfaceId, SemIR::NamedConstraintId>;

  // Push a new stack frame for an interface or named constraint to add
  // RequireImplsIds.
  auto Push(EnclosingScopeId scope_id) -> void;
  // Pop the top stack frame and all its RequireImplsIds.
  auto Pop() -> void;

  // Append to the top stack frame.
  auto AppendToTop(SemIR::RequireImplsId id) -> void;

  // Returns the RequireImplsIds in the top stack frame.
  auto PeekTop() const -> llvm::ArrayRef<SemIR::RequireImplsId>;

  // Finds the stack frame for a given scope and returns the RequireImplsIds in
  // that stack frame.
  auto PeekForScope(EnclosingScopeId scope_id)
      -> llvm::ArrayRef<SemIR::RequireImplsId>;

 private:
  llvm::SmallVector<EnclosingScopeId> scope_ids_;
  ArrayStack<SemIR::RequireImplsId> array_stack_;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_REQUIRE_IMPLS_STACK_H_
