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

  // The kind of a full-pattern. There are two primary kinds: name binding
  // declarations and parameterized entity declarations. However, for efficiency
  // we also use this enum to track state transitions within a parameterized
  // entity declaration. A parameterized entity declaration always starts and
  // finishes in the `NotInEitherParamList` state, and can transition to either
  // the `ImplicitParamList` or `ExplicitParamList` state, and then back to the
  // `NotInEitherParamList` state.
  enum class Kind {
    // A name-binding declaration, such as a `let` or `var` statement.
    NameBindingDecl,

    // The implicit parameter list of a function or impl declaration.
    ImplicitParamList,

    // The explicit parameter list of a function declaration.
    ExplicitParamList,

    // This kind indicates that we're within the declaration of a parameterized
    // entity (such as a function or impl), but not within an explicit or
    // implicit parameter list. This is primarily useful for the return part of
    // a function declaration, which doesn't contain pattern syntax, but can
    // implicitly introduce output parameter patterns. However, the parse tree
    // doesn't let us reliably distinguish the return part from the part before
    // the parameter lists (or between them), particularly in the case where
    // there is no explicit parameter list.
    NotInEitherParamList
  };

  auto empty() const -> bool { return kind_stack_.empty(); }

  // The kind of the current full-pattern.
  auto CurrentKind() const -> Kind { return kind_stack_.back(); }

  // Marks the start of a new full-pattern for a parameterized entity
  // declaration, such as a function or impl. The kind is initially
  // NotInEitherParamList.
  auto PushParameterizedDecl() -> void {
    kind_stack_.push_back(Kind::NotInEitherParamList);
    bind_name_stack_.PushArray();
  }

  // Marks the start of a new full-pattern for a name binding declaration.
  auto PushNameBindingDecl() -> void {
    kind_stack_.push_back(Kind::NameBindingDecl);
    bind_name_stack_.PushArray();
  }

  // Marks the start of the current parameterized entity's implicit parameter
  // list.
  auto StartImplicitParamList() -> void {
    CARBON_CHECK(kind_stack_.back() == Kind::NotInEitherParamList, "{0}",
                 kind_stack_.back());
    kind_stack_.back() = Kind::ImplicitParamList;
  }

  // Marks the end of the current parameterized entity's implicit parameter
  // list.
  auto EndImplicitParamList() -> void {
    CARBON_CHECK(kind_stack_.back() == Kind::ImplicitParamList, "{0}",
                 kind_stack_.back());
    kind_stack_.back() = Kind::NotInEitherParamList;
  }

  // Marks the start of the current parameterized entity's explicit parameter
  // list.
  auto StartExplicitParamList() -> void {
    CARBON_CHECK(kind_stack_.back() == Kind::NotInEitherParamList, "{0}",
                 kind_stack_.back());
    kind_stack_.back() = Kind::ExplicitParamList;
  }

  // Marks the end of the current parameterized entity's explicit parameter
  // list.
  auto EndExplicitParamList() -> void {
    CARBON_CHECK(kind_stack_.back() == Kind::ExplicitParamList, "{0}",
                 kind_stack_.back());
    kind_stack_.back() = Kind::NotInEitherParamList;
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

  // Runs verification that the processing cleanly finished.
  auto VerifyOnFinish() const -> void {
    CARBON_CHECK(kind_stack_.empty(),
                 "full_pattern_stack still has {0} entries",
                 kind_stack_.size());
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
