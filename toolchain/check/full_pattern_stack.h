// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_FULL_PATTERN_STACK_H_
#define CARBON_TOOLCHAIN_CHECK_FULL_PATTERN_STACK_H_

#include "common/array_stack.h"
#include "common/check.h"
#include "common/set.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Stack of full-patterns currently being checked. When a pattern
// is followed by an explicit initializer, name bindings should not be used
// within that initializer, although they are usable before it (within the
// pattern) and after it. This class keeps track of those state transitions.
// It is structured as a stack to handle situations like a pattern that
// contains an initializer, or a pattern in a lambda in an expression pattern.
//
// TODO: Unify this with Context::pattern_block_stack, or differentiate them
// more clearly.
class FullPatternStack {
 public:
  // Marks the possible start of a new full-pattern (i.e. a pattern which occurs
  // in a non-pattern context).
  auto PushFullPattern() -> void { bind_name_stack_.PushArray(); }

  // Marks the start of the initializer for the full-pattern at the top of the
  // stack.
  auto StartPatternInitializer() -> void {
    for (SemIR::InstId bind_name_id : bind_name_stack_.PeekArray()) {
      CARBON_CHECK(unusable_bind_names_.Insert(bind_name_id).is_inserted());
    }
  }

  // Marks the end of the initializer for the full-pattern at the top of the
  // stack.
  auto EndPatternInitializer() -> void {
    for (SemIR::InstId bind_name_id : bind_name_stack_.PeekArray()) {
      CARBON_CHECK(unusable_bind_names_.Erase(bind_name_id));
    }
  }

  // Marks the end of checking for the full-pattern at the top of the stack.
  // This cannot be called while processing an initializer for the top
  // pattern.
  auto PopFullPattern() -> void { bind_name_stack_.PopArray(); }

  // Records that `bind_inst_id` was introduced by the full-pattern at the
  // top of the stack.
  auto AddBindName(SemIR::InstId bind_inst_id) -> void {
    bind_name_stack_.AppendToTop(bind_inst_id);
  }

  // Returns false if the pattern that introduced `bind_inst_id` is currently
  // being initialized.
  auto IsBindNameUsable(SemIR::InstId bind_inst_id) const -> bool {
    return !unusable_bind_names_.Contains(bind_inst_id);
  }

 private:
  ArrayStack<SemIR::InstId> bind_name_stack_;
  Set<SemIR::InstId> unusable_bind_names_;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_FULL_PATTERN_STACK_H_
