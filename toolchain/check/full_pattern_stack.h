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

// Stack of full-patterns currently being checked (a full-pattern is a pattern
// that is not part of an enclosing pattern). It is structured as a stack to
// handle situations like a pattern that contains an initializer, or a pattern
// in a lambda in an expression pattern.
//
// When a pattern is followed by an explicit initializer, name bindings should
// not be used within that initializer, although they are usable before it
// (within the pattern) and after it. This class keeps track of those state
// transitions, as well as the kind of full-pattern (e.g. parameter list or name
// binding pattern).
//
// TODO: Unify this with Context::pattern_block_stack, or differentiate them
// more clearly (and consider unifying this with ScopeStack instead).
class FullPatternStack {
 public:
  explicit FullPatternStack(LexicalLookup* lookup) : lookup_(lookup) {}

  // The kind of a full-pattern. Note that an implicit parameter list and
  // adjacent explicit parameter list together form a single full-pattern,
  // but we separate them here in order to represent the state transition
  // between them.
  enum class Kind {
    NameBindingDecl,
    ImplicitParamList,
    ExplicitParamList,
  };

  // The kind of the current full-pattern.
  auto CurrentKind() const -> Kind { return kind_stack_.back(); }

  // Marks the start of a new full-pattern of the specified kind.
  auto PushFullPattern(Kind kind) -> void {
    kind_stack_.push_back(kind);
    bind_name_stack_.PushArray();
  }

  // Marks the end of an implicit parameter list, and the presumptive start
  // of the corresponding explicit parameter list.
  auto EndImplicitParamList() -> void {
    CARBON_CHECK(kind_stack_.back() == Kind::ImplicitParamList, "{0}",
                 kind_stack_.back());
    kind_stack_.back() = Kind::ExplicitParamList;
  }

  // Marks the start of the initializer for the current name binding decl.
  auto StartPatternInitializer() -> void {
    CARBON_CHECK(kind_stack_.back() == Kind::NameBindingDecl);
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

  // Marks the end of the initializer for the current name-binding decl.
  auto EndPatternInitializer() -> void {
    for (auto& [name_id, inst_id] : bind_name_stack_.PeekArray()) {
      auto& lookup_result = lookup_->Get(name_id);
      if (!lookup_result.empty()) {
        std::swap(lookup_result.back().inst_id, inst_id);
      }
      CARBON_CHECK(inst_id == SemIR::InstId::InitTombstone);
    }
  }

  // Marks the end of checking for the current full-pattern. This cannot be
  // called while processing an initializer for the top pattern.
  auto PopFullPattern() -> void {
    kind_stack_.pop_back();
    bind_name_stack_.PopArray();
  }

  // Records that `name_id` was introduced by the current full-pattern.
  auto AddBindName(SemIR::NameId name_id) -> void {
    bind_name_stack_.AppendToTop(
        {.name_id = name_id, .inst_id = SemIR::InstId::InitTombstone});
  }

 private:
  LexicalLookup* lookup_;

  llvm::SmallVector<Kind> kind_stack_;

  struct LookupEntry {
    SemIR::NameId name_id;
    SemIR::InstId inst_id;
  };
  ArrayStack<LookupEntry> bind_name_stack_;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_FULL_PATTERN_STACK_H_
