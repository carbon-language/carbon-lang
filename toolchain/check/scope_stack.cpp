// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/scope_stack.h"

#include "common/check.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

auto ScopeStack::VerifyOnFinish() -> void {
  CARBON_CHECK(scope_stack_.empty()) << scope_stack_.size();
}

auto ScopeStack::Push(SemIR::InstId scope_inst_id, SemIR::NameScopeId scope_id,
                      bool lexical_lookup_has_load_error) -> void {
  scope_stack_.push_back(
      {.index = next_scope_index_,
       .scope_inst_id = scope_inst_id,
       .scope_id = scope_id,
       .prev_lexical_lookup_has_load_error = lexical_lookup_has_load_error_});
  if (scope_id.is_valid()) {
    non_lexical_scope_stack_.push_back({next_scope_index_, scope_id});
  }

  lexical_lookup_has_load_error_ |= lexical_lookup_has_load_error;

  // TODO: Handle this case more gracefully.
  CARBON_CHECK(next_scope_index_.index != std::numeric_limits<int32_t>::max())
      << "Ran out of scopes";
  ++next_scope_index_.index;
}

auto ScopeStack::Pop() -> void {
  auto scope = scope_stack_.pop_back_val();

  lexical_lookup_has_load_error_ = scope.prev_lexical_lookup_has_load_error;

  for (const auto& str_id : scope.names) {
    auto& lexical_results = lexical_lookup_.Get(str_id);
    CARBON_CHECK(lexical_results.back().scope_index == scope.index)
        << "Inconsistent scope index for name " << str_id;
    lexical_results.pop_back();
  }

  if (scope.scope_id.is_valid()) {
    CARBON_CHECK(non_lexical_scope_stack_.back().scope_index == scope.index);
    non_lexical_scope_stack_.pop_back();
  }

  if (scope.has_returned_var) {
    CARBON_CHECK(!return_scope_stack_.empty());
    CARBON_CHECK(return_scope_stack_.back().returned_var.is_valid());
    return_scope_stack_.back().returned_var = SemIR::InstId::Invalid;
  }
}

auto ScopeStack::PopTo(ScopeIndex index) -> void {
  while (PeekIndex() > index) {
    Pop();
  }
  CARBON_CHECK(PeekIndex() == index)
      << "Scope index " << index << " does not enclose the current scope "
      << PeekIndex();
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

auto ScopeStack::LookupInEnclosingScopes(SemIR::NameId name_id)
    -> std::pair<SemIR::InstId, llvm::ArrayRef<NonLexicalScope>> {
  // Find the results from enclosing lexical scopes. These will be combined with
  // results from non-lexical scopes such as namespaces and classes.
  llvm::ArrayRef<LexicalLookup::Result> lexical_results =
      lexical_lookup_.Get(name_id);

  // If we have no lexical results, check all non-lexical scopes.
  if (lexical_results.empty()) {
    return {lexical_lookup_has_load_error_ ? SemIR::InstId::BuiltinError
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
  if (!scope_stack_.back().names.insert(name_id).second) {
    auto existing = lexical_lookup_.Get(name_id).back().inst_id;
    CARBON_CHECK(existing.is_valid())
        << "Name in scope but not in lexical lookups";
    return existing;
  }

  // TODO: Reject if we previously performed a failed lookup for this name
  // in this scope or a scope nested within it.
  auto& lexical_results = lexical_lookup_.Get(name_id);
  CARBON_CHECK(lexical_results.empty() ||
               lexical_results.back().scope_index < PeekIndex())
      << "Failed to clean up after scope nested within the current scope";
  lexical_results.push_back({.inst_id = target_id, .scope_index = PeekIndex()});
  return SemIR::InstId::Invalid;
}

auto ScopeStack::SetReturnedVarOrGetExisting(SemIR::InstId inst_id)
    -> SemIR::InstId {
  CARBON_CHECK(!return_scope_stack_.empty()) << "`returned var` in no function";
  auto& returned_var = return_scope_stack_.back().returned_var;
  if (returned_var.is_valid()) {
    return returned_var;
  }

  returned_var = inst_id;
  CARBON_CHECK(!scope_stack_.back().has_returned_var)
      << "Scope has returned var but none is set";
  if (inst_id.is_valid()) {
    scope_stack_.back().has_returned_var = true;
  }
  return SemIR::InstId::Invalid;
}

}  // namespace Carbon::Check
