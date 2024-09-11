// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/scope_stack.h"

#include "common/check.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

auto ScopeStack::VerifyOnFinish() -> void {
  CARBON_CHECK(scope_stack_.empty(), "{0}", scope_stack_.size());
}

auto ScopeStack::Push(SemIR::InstId scope_inst_id, SemIR::NameScopeId scope_id,
                      SemIR::SpecificId specific_id,
                      bool lexical_lookup_has_load_error) -> void {
  // If this scope doesn't have a specific of its own, it lives in the enclosing
  // scope's specific, if any.
  auto enclosing_specific_id = specific_id;
  if (!specific_id.is_valid() && !scope_stack_.empty()) {
    enclosing_specific_id = PeekSpecificId();
  }

  compile_time_binding_stack_.PushArray();
  scope_stack_.push_back(
      {.index = next_scope_index_,
       .scope_inst_id = scope_inst_id,
       .scope_id = scope_id,
       .specific_id = enclosing_specific_id,
       .next_compile_time_bind_index = SemIR::CompileTimeBindIndex(
           compile_time_binding_stack_.all_values_size()),
       .lexical_lookup_has_load_error =
           LexicalLookupHasLoadError() || lexical_lookup_has_load_error});
  if (scope_id.is_valid()) {
    non_lexical_scope_stack_.push_back({.scope_index = next_scope_index_,
                                        .name_scope_id = scope_id,
                                        .specific_id = enclosing_specific_id});
  } else {
    // For lexical lookups, unqualified lookup doesn't know how to find the
    // associated specific, so if we start adding lexical scopes associated with
    // specifics, we'll need to somehow track them in lookup.
    CARBON_CHECK(!specific_id.is_valid(),
                 "Lexical scope should not have an associated specific.");
  }

  // TODO: Handle this case more gracefully.
  CARBON_CHECK(next_scope_index_.index != std::numeric_limits<int32_t>::max(),
               "Ran out of scopes");
  ++next_scope_index_.index;
}

auto ScopeStack::Pop() -> void {
  auto scope = scope_stack_.pop_back_val();

  scope.names.ForEach([&](SemIR::NameId str_id) {
    auto& lexical_results = lexical_lookup_.Get(str_id);
    CARBON_CHECK(lexical_results.back().scope_index == scope.index,
                 "Inconsistent scope index for name {0}", str_id);
    lexical_results.pop_back();
  });

  if (scope.scope_id.is_valid()) {
    CARBON_CHECK(non_lexical_scope_stack_.back().scope_index == scope.index);
    non_lexical_scope_stack_.pop_back();
  }

  if (scope.has_returned_var) {
    CARBON_CHECK(!return_scope_stack_.empty());
    CARBON_CHECK(return_scope_stack_.back().returned_var.is_valid());
    return_scope_stack_.back().returned_var = SemIR::InstId::Invalid;
  }

  CARBON_CHECK(
      scope.next_compile_time_bind_index.index ==
          static_cast<int32_t>(compile_time_binding_stack_.all_values_size()),
      "Wrong number of entries in compile-time binding stack, have {0}, "
      "expected {1}",
      compile_time_binding_stack_.all_values_size(),
      scope.next_compile_time_bind_index.index);
  compile_time_binding_stack_.PopArray();
}

auto ScopeStack::PopTo(ScopeIndex index) -> void {
  while (PeekIndex() > index) {
    Pop();
  }
  CARBON_CHECK(PeekIndex() == index,
               "Scope index {0} does not enclose the current scope {1}", index,
               PeekIndex());
}

auto ScopeStack::LookupInCurrentScope(SemIR::NameId name_id) -> SemIR::InstId {
  auto& lexical_results = lexical_lookup_.Get(name_id);
  if (lexical_results.empty()) {
    return SemIR::InstId::Invalid;
  }

  auto result = lexical_results.back();
  if (result.scope_index != PeekIndex()) {
    return SemIR::InstId::Invalid;
  }

  return result.inst_id;
}

auto ScopeStack::LookupInLexicalScopes(SemIR::NameId name_id)
    -> std::pair<SemIR::InstId, llvm::ArrayRef<NonLexicalScope>> {
  // Find the results from lexical scopes. These will be combined with results
  // from non-lexical scopes such as namespaces and classes.
  llvm::ArrayRef<LexicalLookup::Result> lexical_results =
      lexical_lookup_.Get(name_id);

  // If we have no lexical results, check all non-lexical scopes.
  if (lexical_results.empty()) {
    return {LexicalLookupHasLoadError() ? SemIR::InstId::BuiltinError
                                        : SemIR::InstId::Invalid,
            non_lexical_scope_stack_};
  }

  // Find the first non-lexical scope that is within the scope of the lexical
  // lookup result.
  auto* first_non_lexical_scope = std::lower_bound(
      non_lexical_scope_stack_.begin(), non_lexical_scope_stack_.end(),
      lexical_results.back().scope_index,
      [](const NonLexicalScope& scope, ScopeIndex index) {
        return scope.scope_index < index;
      });
  return {
      lexical_results.back().inst_id,
      llvm::ArrayRef(first_non_lexical_scope, non_lexical_scope_stack_.end())};
}

auto ScopeStack::LookupOrAddName(SemIR::NameId name_id, SemIR::InstId target_id)
    -> SemIR::InstId {
  if (!scope_stack_.back().names.Insert(name_id).is_inserted()) {
    auto existing = lexical_lookup_.Get(name_id).back().inst_id;
    CARBON_CHECK(existing.is_valid(),
                 "Name in scope but not in lexical lookups");
    return existing;
  }
  ++scope_stack_.back().num_names;

  // TODO: Reject if we previously performed a failed lookup for this name
  // in this scope or a scope nested within it.
  auto& lexical_results = lexical_lookup_.Get(name_id);
  CARBON_CHECK(
      lexical_results.empty() ||
          lexical_results.back().scope_index < PeekIndex(),
      "Failed to clean up after scope nested within the current scope");
  lexical_results.push_back({.inst_id = target_id, .scope_index = PeekIndex()});
  return SemIR::InstId::Invalid;
}

auto ScopeStack::SetReturnedVarOrGetExisting(SemIR::InstId inst_id)
    -> SemIR::InstId {
  CARBON_CHECK(!return_scope_stack_.empty(), "`returned var` in no function");
  auto& returned_var = return_scope_stack_.back().returned_var;
  if (returned_var.is_valid()) {
    return returned_var;
  }

  returned_var = inst_id;
  CARBON_CHECK(!scope_stack_.back().has_returned_var,
               "Scope has returned var but none is set");
  if (inst_id.is_valid()) {
    scope_stack_.back().has_returned_var = true;
  }
  return SemIR::InstId::Invalid;
}

auto ScopeStack::Suspend() -> SuspendedScope {
  CARBON_CHECK(!scope_stack_.empty(), "No scope to suspend");
  SuspendedScope result = {.entry = scope_stack_.pop_back_val(),
                           .suspended_items = {}};
  if (result.entry.scope_id.is_valid()) {
    non_lexical_scope_stack_.pop_back();
  }

  auto peek_compile_time_bindings = compile_time_binding_stack_.PeekArray();
  result.suspended_items.reserve(result.entry.num_names +
                                 peek_compile_time_bindings.size());

  result.entry.names.ForEach([&](SemIR::NameId name_id) {
    auto [index, inst_id] = lexical_lookup_.Suspend(name_id);
    CARBON_CHECK(index !=
                 SuspendedScope::ScopeItem::IndexForCompileTimeBinding);
    result.suspended_items.push_back({.index = index, .inst_id = inst_id});
  });
  CARBON_CHECK(static_cast<int>(result.suspended_items.size()) ==
               result.entry.num_names);

  // Move any compile-time bindings into the suspended scope.
  for (auto inst_id : peek_compile_time_bindings) {
    result.suspended_items.push_back(
        {.index = SuspendedScope::ScopeItem::IndexForCompileTimeBinding,
         .inst_id = inst_id});
  }
  compile_time_binding_stack_.PopArray();

  // This would be easy to support if we had a need, but currently we do not.
  CARBON_CHECK(!result.entry.has_returned_var,
               "Should not suspend a scope with a returned var.");
  return result;
}

auto ScopeStack::Restore(SuspendedScope scope) -> void {
  compile_time_binding_stack_.PushArray();
  for (auto [index, inst_id] : scope.suspended_items) {
    if (index == SuspendedScope::ScopeItem::IndexForCompileTimeBinding) {
      compile_time_binding_stack_.AppendToTop(inst_id);
    } else {
      lexical_lookup_.Restore({.index = index, .inst_id = inst_id},
                              scope.entry.index);
    }
  }

  CARBON_CHECK(
      scope.entry.next_compile_time_bind_index.index ==
          static_cast<int32_t>(compile_time_binding_stack_.all_values_size()),
      "Wrong number of entries in compile-time binding stack when restoring, "
      "have {0}, expected {1}",
      compile_time_binding_stack_.all_values_size(),
      scope.entry.next_compile_time_bind_index.index);

  if (scope.entry.scope_id.is_valid()) {
    non_lexical_scope_stack_.push_back(
        {.scope_index = scope.entry.index,
         .name_scope_id = scope.entry.scope_id,
         .specific_id = scope.entry.specific_id});
  }
  scope_stack_.push_back(std::move(scope.entry));
}

}  // namespace Carbon::Check
