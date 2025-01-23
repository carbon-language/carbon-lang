// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_FULL_PATTERN_STACK_H_
#define CARBON_TOOLCHAIN_CHECK_FULL_PATTERN_STACK_H_

#include "common/array_stack.h"
#include "common/check.h"
#include "toolchain/check/lexical_lookup.h"
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
// more clearly (and consider unifying this with ScopeStack instead).
class FullPatternStack {
 public:
  explicit FullPatternStack(LexicalLookup* lookup) : lookup_(lookup) {}

  // Marks the possible start of a new full-pattern (i.e. a pattern which occurs
  // in a non-pattern context).
  auto PushFullPattern() -> void { bind_name_stack_.PushArray(); }

  // Marks the start of the initializer for the full-pattern at the top of the
  // stack.
  auto StartPatternInitializer() -> void {
    for (auto& [name_id, inst_id] : bind_name_stack_.PeekArray()) {
      CARBON_CHECK(inst_id == SemIR::InstId::InitTombstone);
      auto& lookup_result = lookup_->Get(name_id);
      if (!lookup_result.empty()) {
        // TODO: find a way to preserve location information, so that we can
        // provide good diagnostics for a redeclaration of `name_id` in
        // the initializer, if that becomes possible.
        std::swap(lookup_result.back().inst_id, inst_id);
      }
    }
  }

  // Marks the end of the initializer for the full-pattern at the top of the
  // stack.
  auto EndPatternInitializer() -> void {
    for (auto& [name_id, inst_id] : bind_name_stack_.PeekArray()) {
      auto& lookup_result = lookup_->Get(name_id);
      if (!lookup_result.empty()) {
        std::swap(lookup_result.back().inst_id, inst_id);
      }
      CARBON_CHECK(inst_id == SemIR::InstId::InitTombstone);
    }
  }

  // Marks the end of checking for the full-pattern at the top of the stack.
  // This cannot be called while processing an initializer for the top
  // pattern.
  auto PopFullPattern() -> void { bind_name_stack_.PopArray(); }

  // Records that `bind_inst_id` was introduced by the full-pattern at the
  // top of the stack.
  auto AddBindName(SemIR::NameId name_id) -> void {
    bind_name_stack_.AppendToTop(
        {.name_id = name_id, .inst_id = SemIR::InstId::InitTombstone});
  }

 private:
  LexicalLookup* lookup_;
  struct LookupEntry {
    SemIR::NameId name_id;
    SemIR::InstId inst_id;
  };
  ArrayStack<LookupEntry> bind_name_stack_;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_FULL_PATTERN_STACK_H_
