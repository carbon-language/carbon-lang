// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_SCOPE_STACK_H_
#define CARBON_TOOLCHAIN_CHECK_SCOPE_STACK_H_

#include "common/array_stack.h"
#include "common/set.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/check/full_pattern_stack.h"
#include "toolchain/check/lexical_lookup.h"
#include "toolchain/check/scope_index.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// A stack of lexical and semantic scopes that we are currently performing
// checking within.
class ScopeStack {
 public:
  explicit ScopeStack(const SemIR::File* sem_ir)
      : sem_ir_(sem_ir),
        lexical_lookup_(sem_ir->identifiers()),
        full_pattern_stack_(&lexical_lookup_) {}

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
    SemIR::InstId returned_var = SemIR::InstId::None;
  };

  // A non-lexical scope in which unqualified lookup may be required.
  struct NonLexicalScope {
    // The index of the scope in the scope stack.
    ScopeIndex scope_index;

    // The corresponding name scope.
    SemIR::NameScopeId name_scope_id;

    // The corresponding specific.
    SemIR::SpecificId specific_id;
  };

  // Information about a scope that has been temporarily removed from the stack.
  struct SuspendedScope;

  // Pushes a scope onto scope_stack_. NameScopeId::None is used for new
  // scopes. lexical_lookup_has_load_error is used to limit diagnostics when a
  // given namespace may contain a mix of both successful and failed name
  // imports.
  auto Push(SemIR::InstId scope_inst_id = SemIR::InstId::None,
            SemIR::NameScopeId scope_id = SemIR::NameScopeId::None,
            SemIR::SpecificId specific_id = SemIR::SpecificId::None,
            bool lexical_lookup_has_load_error = false) -> void;

  // Pops the top scope from scope_stack_, cleaning up names from
  // lexical_lookup_.
  auto Pop() -> void;

  // Pops the top scope from scope_stack_ if it contains no names.
  auto PopIfEmpty() -> void {
    if (scope_stack_.back().num_names == 0) {
      Pop();
    }
  }

  // Pops scopes until we return to the specified scope index.
  auto PopTo(ScopeIndex index) -> void;

  // Returns the scope index associated with the current scope.
  auto PeekIndex() const -> ScopeIndex { return Peek().index; }

  // Returns the name scope associated with the current lexical scope, if any.
  auto PeekNameScopeId() const -> SemIR::NameScopeId { return Peek().scope_id; }

  // Returns the instruction associated with the current scope, or `None` if
  // there is no such instruction, such as for a block scope.
  auto PeekInstId() const -> SemIR::InstId { return Peek().scope_inst_id; }

  // Returns the specific associated with the innermost enclosing scope that is
  // associated with a specific. This will generally be the self specific of the
  // innermost enclosing generic, as there is no way to enter any other specific
  // scope.
  auto PeekSpecificId() const -> SemIR::SpecificId {
    return Peek().specific_id;
  }

  // Returns true if current scope is lexical.
  auto PeekIsLexicalScope() const -> bool { return Peek().is_lexical_scope(); }

  // Returns the current scope, if it is of the specified kind. Otherwise,
  // returns nullopt.
  template <typename InstT>
  auto GetCurrentScopeAs() -> std::optional<InstT> {
    auto inst_id = PeekInstId();
    if (!inst_id.has_value()) {
      return std::nullopt;
    }
    return sem_ir_->insts().TryGetAs<InstT>(inst_id);
  }

  // If there is no `returned var` in scope, sets the given instruction to be
  // the current `returned var` and returns an `None`. If there
  // is already a `returned var`, returns it instead.
  auto SetReturnedVarOrGetExisting(SemIR::InstId inst_id) -> SemIR::InstId;

  // Looks up the name `name_id` in the current scope and enclosing scopes, but
  // do not look past `scope_index`. Returns the existing lookup result, if any.
  auto LookupInLexicalScopesWithin(SemIR::NameId name_id,
                                   ScopeIndex scope_index) -> SemIR::InstId;

  // Looks up the name `name_id` in the current scope and related lexical
  // scopes. Returns the innermost lexical lookup result, if any, along with a
  // list of non-lexical scopes in which lookup should also be performed,
  // ordered from outermost to innermost.
  auto LookupInLexicalScopes(SemIR::NameId name_id)
      -> std::pair<SemIR::InstId, llvm::ArrayRef<NonLexicalScope>>;

  // Looks up the name `name_id` in the current scope, or in `scope_index` if
  // specified. Returns the existing instruction if the name is already declared
  // in that scope or any unfinished scope within it, and otherwise adds the
  // name with the value `target_id` and returns `None`.
  auto LookupOrAddName(SemIR::NameId name_id, SemIR::InstId target_id,
                       ScopeIndex scope_index = ScopeIndex::None)
      -> SemIR::InstId;

  // Prepares to add a compile-time binding in the current scope, and returns
  // its index. The added binding must then be pushed using
  // `PushCompileTimeBinding`.
  auto AddCompileTimeBinding() -> SemIR::CompileTimeBindIndex {
    auto index = scope_stack_.back().next_compile_time_bind_index;
    ++scope_stack_.back().next_compile_time_bind_index.index;
    return index;
  }

  // Pushes a compile-time binding into the current scope.
  auto PushCompileTimeBinding(SemIR::InstId bind_id) -> void {
    compile_time_binding_stack_.AppendToTop(bind_id);
  }

  // Temporarily removes the top of the stack and its lexical lookup results.
  auto Suspend() -> SuspendedScope;

  // Restores a suspended scope stack entry.
  auto Restore(SuspendedScope scope) -> void;

  // Runs verification that the processing cleanly finished.
  auto VerifyOnFinish() const -> void;

  auto return_scope_stack() -> llvm::SmallVector<ReturnScope>& {
    return return_scope_stack_;
  }

  auto break_continue_stack() -> llvm::SmallVector<BreakContinueScope>& {
    return break_continue_stack_;
  }

  auto destroy_id_stack() -> ArrayStack<SemIR::InstId>& {
    return destroy_id_stack_;
  }

  auto compile_time_bindings_stack() -> ArrayStack<SemIR::InstId>& {
    return compile_time_binding_stack_;
  }

  auto full_pattern_stack() -> FullPatternStack& { return full_pattern_stack_; }

 private:
  // An entry in scope_stack_.
  struct ScopeStackEntry {
    auto is_lexical_scope() const -> bool { return !scope_id.has_value(); }

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

    // The specific associated with this entry, if any.
    SemIR::SpecificId specific_id;

    // The next compile-time binding index to allocate in this scope.
    SemIR::CompileTimeBindIndex next_compile_time_bind_index;

    // Whether lexical_lookup_ has load errors from this scope or an ancestor
    // scope.
    bool lexical_lookup_has_load_error;

    // Whether a `returned var` was introduced in this scope, and needs to be
    // unregistered when the scope ends.
    bool has_returned_var = false;

    // Whether there are any ids in the `names` set.
    int num_names = 0;

    // Names which are registered with lexical_lookup_, and will need to be
    // unregistered when the scope ends.
    Set<SemIR::NameId> names = {};
  };

  auto Peek() const -> const ScopeStackEntry& { return scope_stack_.back(); }

  // Returns whether lexical lookup currently has any load errors.
  auto LexicalLookupHasLoadError() const -> bool {
    return !scope_stack_.empty() &&
           scope_stack_.back().lexical_lookup_has_load_error;
  }

  // Checks that the provided scope's `next_compile_time_bind_index` matches the
  // full size of the current `compile_time_binding_stack_`. The values should
  // always match, and this is used to validate the correspondence during
  // significant changes.
  auto VerifyNextCompileTimeBindIndex(llvm::StringLiteral label,
                                      const ScopeStackEntry& scope) -> void;

  // The current file.
  const SemIR::File* sem_ir_;

  // A stack of scopes from which we can `return`.
  llvm::SmallVector<ReturnScope> return_scope_stack_;

  // A stack of `break` and `continue` targets.
  llvm::SmallVector<BreakContinueScope> break_continue_stack_;

  // A stack for scope context.
  llvm::SmallVector<ScopeStackEntry> scope_stack_;

  // A stack of `destroy` functions to call. This only has entries for lexical
  // scopes, because non-lexical scopes don't have destruction on scope exit.
  ArrayStack<SemIR::InstId> destroy_id_stack_;

  // Information about non-lexical scopes. This is a subset of the entries and
  // the information in scope_stack_.
  llvm::SmallVector<NonLexicalScope> non_lexical_scope_stack_;

  // A stack of the current compile time bindings.
  ArrayStack<SemIR::InstId> compile_time_binding_stack_;

  // The index of the next scope that will be pushed onto scope_stack_. The
  // first is always the package scope.
  ScopeIndex next_scope_index_ = ScopeIndex::Package;

  // Tracks lexical lookup results.
  LexicalLookup lexical_lookup_;

  // Stack of full-patterns currently being checked.
  FullPatternStack full_pattern_stack_;
};

struct ScopeStack::SuspendedScope {
  // An item that was suspended within this scope. This represents either a
  // lexical lookup entry in this scope, or a compile time binding entry in this
  // scope.
  //
  // TODO: For compile-time bindings, the common case is that they will both
  // have a suspended lexical lookup entry and a suspended compile time binding
  // entry. We should be able to store that as a single ScopeItem rather than
  // two.
  struct ScopeItem {
    static constexpr uint32_t IndexForCompileTimeBinding = -1;

    // The scope index for a LexicalLookup::SuspendedResult, or
    // CompileTimeBindingIndex for a suspended compile time binding.
    uint32_t index;
    // The instruction within the scope.
    SemIR::InstId inst_id;
  };

  // The suspended scope stack entry.
  ScopeStackEntry entry;
  // The list of items that were within this scope when it was suspended. The
  // inline size is an attempt to keep the size of a `SuspendedFunction`
  // reasonable while avoiding heap allocations most of the time.
  llvm::SmallVector<ScopeItem, 8> suspended_items;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_SCOPE_STACK_H_
