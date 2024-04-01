// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_SCOPE_STACK_H_
#define CARBON_TOOLCHAIN_CHECK_SCOPE_STACK_H_

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/check/lexical_lookup.h"
#include "toolchain/check/scope_index.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// A stack of lexical and semantic scopes that we are currently performing
// checking within.
class ScopeStack {
 public:
  explicit ScopeStack(const StringStoreWrapper<IdentifierId>& identifiers)
      : lexical_lookup_(identifiers) {}

  // A scope in which `break` and `continue` can be used.
  struct BreakContinueScope {
    SemIR::InstBlockId break_target;
    SemIR::InstBlockId continue_target;
  };

  // A scope in which `return` can be used.
  struct ReturnScope {
    // The declaration from which we can return. Inside a function, this will
    // be a `FunctionDecl`.
    SemIR::InstId decl_id;

    // The value corresponding to the current `returned var`, if any. Will be
    // set and unset as `returned var`s are declared and go out of scope.
    SemIR::InstId returned_var = SemIR::InstId::Invalid;
  };

  // A non-lexical scope in which unqualified lookup may be required.
  struct NonLexicalScope {
    // The index of the scope in the scope stack.
    ScopeIndex scope_index;

    // The corresponding name scope.
    SemIR::NameScopeId name_scope_id;
  };

  // Information about a scope that has been temporarily removed from the stack.
  struct SuspendedScope;

  // Pushes a scope onto scope_stack_. NameScopeId::Invalid is used for new
  // scopes. lexical_lookup_has_load_error is used to limit diagnostics when a
  // given namespace may contain a mix of both successful and failed name
  // imports.
  auto Push(SemIR::InstId scope_inst_id = SemIR::InstId::Invalid,
            SemIR::NameScopeId scope_id = SemIR::NameScopeId::Invalid,
            bool lexical_lookup_has_load_error = false) -> void;

  // Pops the top scope from scope_stack_, cleaning up names from
  // lexical_lookup_.
  auto Pop() -> void;

  // Pops the top scope from scope_stack_ if it contains no names.
  auto PopIfEmpty() -> void {
    if (scope_stack_.back().names.empty()) {
      Pop();
    }
  }

  // Pops scopes until we return to the specified scope index.
  auto PopTo(ScopeIndex index) -> void;

  // Returns the scope index associated with the current scope.
  auto PeekIndex() const -> ScopeIndex { return Peek().index; }

  // Returns the name scope associated with the current lexical scope, if any.
  auto PeekNameScopeId() const -> SemIR::NameScopeId { return Peek().scope_id; }

  // Returns the instruction associated with the current scope, or Invalid if
  // there is no such instruction, such as for a block scope.
  auto PeekInstId() const -> SemIR::InstId { return Peek().scope_inst_id; }

  // Returns the current scope, if it is of the specified kind. Otherwise,
  // returns nullopt.
  template <typename InstT>
  auto GetCurrentScopeAs(const SemIR::File& sem_ir) -> std::optional<InstT> {
    auto inst_id = PeekInstId();
    if (!inst_id.is_valid()) {
      return std::nullopt;
    }
    return sem_ir.insts().TryGetAs<InstT>(inst_id);
  }

  // If there is no `returned var` in scope, sets the given instruction to be
  // the current `returned var` and returns an invalid instruction ID. If there
  // is already a `returned var`, returns it instead.
  auto SetReturnedVarOrGetExisting(SemIR::InstId inst_id) -> SemIR::InstId;

  // Looks up the name `name_id` in the current scope. Returns the existing
  // lookup result, if any.
  auto LookupInCurrentScope(SemIR::NameId name_id) -> SemIR::InstId;

  // Looks up the name `name_id` in the current scope and its enclosing scopes.
  // Returns the innermost lexical lookup result, if any, along with a list of
  // non-lexical scopes in which lookup should also be performed, ordered from
  // outermost to innermost.
  auto LookupInEnclosingScopes(SemIR::NameId name_id)
      -> std::pair<SemIR::InstId, llvm::ArrayRef<NonLexicalScope>>;

  // Looks up the name `name_id` in the current scope. Returns the existing
  // instruction if any, and otherwise adds the name with the value `target_id`
  // and returns Invalid.
  auto LookupOrAddName(SemIR::NameId name_id, SemIR::InstId target_id)
      -> SemIR::InstId;

  // Temporarily removes the top of the stack and its lexical lookup results.
  auto Suspend() -> SuspendedScope;

  // Restores a suspended scope stack entry.
  auto Restore(SuspendedScope scope) -> void;

  // Runs verification that the processing cleanly finished.
  auto VerifyOnFinish() -> void;

  auto return_scope_stack() -> llvm::SmallVector<ReturnScope>& {
    return return_scope_stack_;
  }

  auto break_continue_stack() -> llvm::SmallVector<BreakContinueScope>& {
    return break_continue_stack_;
  }

 private:
  // An entry in scope_stack_.
  struct ScopeStackEntry {
    // The sequential index of this scope entry within the file.
    ScopeIndex index;

    // The instruction associated with this entry, if any. This can be one of:
    //
    // - A `ClassDecl`, for a class definition scope.
    // - A `FunctionDecl`, for the outermost scope in a function
    //   definition.
    // - Invalid, for any other scope.
    SemIR::InstId scope_inst_id;

    // The name scope associated with this entry, if any.
    SemIR::NameScopeId scope_id;

    // Whether lexical_lookup_ has load errors from this scope or an enclosing
    // scope.
    bool lexical_lookup_has_load_error;

    // Whether a `returned var` was introduced in this scope, and needs to be
    // unregistered when the scope ends.
    bool has_returned_var = false;

    // Names which are registered with lexical_lookup_, and will need to be
    // unregistered when the scope ends.
    llvm::DenseSet<SemIR::NameId> names;

    // TODO: This likely needs to track things which need to be destructed.
  };

  auto Peek() const -> const ScopeStackEntry& { return scope_stack_.back(); }

  // Returns whether lexical lookup currently has any load errors.
  auto LexicalLookupHasLoadError() const -> bool {
    return !scope_stack_.empty() &&
           scope_stack_.back().lexical_lookup_has_load_error;
  }

  // A stack of scopes from which we can `return`.
  llvm::SmallVector<ReturnScope> return_scope_stack_;

  // A stack of `break` and `continue` targets.
  llvm::SmallVector<BreakContinueScope> break_continue_stack_;

  // A stack for scope context.
  llvm::SmallVector<ScopeStackEntry> scope_stack_;

  // Information about non-lexical scopes. This is a subset of the entries and
  // the information in scope_stack_.
  llvm::SmallVector<NonLexicalScope> non_lexical_scope_stack_;

  // The index of the next scope that will be pushed onto scope_stack_. The
  // first is always the package scope.
  ScopeIndex next_scope_index_ = ScopeIndex::Package;

  // Tracks lexical lookup results.
  LexicalLookup lexical_lookup_;
};

struct ScopeStack::SuspendedScope {
  // The suspended scope stack entry.
  ScopeStackEntry entry;
  // The lexical lookups for the suspended entry. The inline size is an attempt
  // to keep the size of a `SuspendedFunction` reasonable while avoiding heap
  // allocations most of the time.
  llvm::SmallVector<LexicalLookup::SuspendedResult, 8> suspended_lookups;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_SCOPE_STACK_H_
