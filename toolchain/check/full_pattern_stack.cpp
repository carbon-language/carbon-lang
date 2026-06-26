// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/full_pattern_stack.h"

#include "toolchain/check/context.h"
#include "toolchain/check/pattern.h"

namespace Carbon::Check {

auto FullPatternStack::StartPatternInitializer() -> void {
  CARBON_CHECK(kind_stack_.back() == Kind::ClassScopeVarDecl ||
               kind_stack_.back() == Kind::NameBindingDecl);
  for (auto& [name_id, inst_id] : bind_name_stack_.PeekArray()) {
    CARBON_CHECK(
        inst_id == SemIR::InstId::InitTombstone,
        "stashing the lookup result would overwrite an existing stash {0}",
        inst_id);
    auto& lookup_result = lookup_->Get(name_id);
    if (!lookup_result.empty()) {
      // Temporarily overwrite the result of name lookup for this binding to be
      // `InitTombstone`, so that references to it in the initializer are
      // diagnosed as errors. The original result of name lookup is stashed in
      // `bind_name_stack_` so we can restore it later.
      //
      // TODO: find a way to preserve location information, so that we can
      // provide good diagnostics for a redeclaration of `name_id` in
      // the initializer, if that becomes possible.
      std::swap(lookup_result.back().inst_id, inst_id);
    }
  }
}

auto FullPatternStack::EndPatternInitializer() -> void {
  for (auto& [name_id, inst_id] : bind_name_stack_.PeekArray()) {
    auto& lookup_result = lookup_->Get(name_id);
    if (!lookup_result.empty()) {
      // Restore the original result of name lookup for this binding.
      std::swap(lookup_result.back().inst_id, inst_id);
    }
    CARBON_CHECK(inst_id == SemIR::InstId::InitTombstone,
                 "name_id resolved to a non-tombstone value {0} during "
                 "initializer handling",
                 inst_id);
  }
}

auto FullPatternStack::BuildLocalVarStorage(Context& context,
                                            bool is_returned_var) -> void {
  for (auto& var_info : var_pattern_stack_.PeekArray()) {
    var_info.storage_id =
        GetOrAddVarStorage(context, var_info.pattern_id, is_returned_var);
  }
  next_var_index_stack_.back() = 0;
}

}  // namespace Carbon::Check
