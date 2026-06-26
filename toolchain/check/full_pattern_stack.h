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

class Context;

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

    // A `var` field declaration inside a class.
    ClassScopeVarDecl,

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

  // Whether the kind of the current full-pattern is a class `var` decl.
  auto IsCurrentKindClassScopeVarDecl() -> bool {
    return !empty() && CurrentKind() == Kind::ClassScopeVarDecl;
  }

  // Marks the start of a new full-pattern for a parameterized entity
  // declaration, such as a function or impl. The kind is initially
  // NotInEitherParamList.
  auto PushParameterizedDecl() -> void {
    kind_stack_.push_back(Kind::NotInEitherParamList);
    bind_name_stack_.PushArray();
    var_pattern_stack_.PushArray();
    next_var_index_stack_.push_back(-1);
  }

  // Marks the start of a new full-pattern for a name binding declaration.
  auto PushNameBindingDecl() -> void {
    kind_stack_.push_back(Kind::NameBindingDecl);
    bind_name_stack_.PushArray();
    var_pattern_stack_.PushArray();
    next_var_index_stack_.push_back(-1);
  }

  // Marks the start of a new full-pattern for a class `var` declaration.
  auto PushClassScopeVarDecl() -> void {
    kind_stack_.push_back(Kind::ClassScopeVarDecl);
    bind_name_stack_.PushArray();
    var_pattern_stack_.PushArray();
    next_var_index_stack_.push_back(-1);
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
  auto StartPatternInitializer() -> void;

  // Marks the end of the initializer for the current name-binding decl.
  auto EndPatternInitializer() -> void;

  // Marks the end of checking and pattern matching for the current
  // full-pattern.
  auto PopFullPattern() -> void {
    kind_stack_.pop_back();
    bind_name_stack_.PopArray();
    int index = next_var_index_stack_.pop_back_val();
    CARBON_CHECK(index < 0 || static_cast<size_t>(index) ==
                                  var_pattern_stack_.PeekArray().size(),
                 "`GetLocalVarStorage` not called for all var patterns");
    var_pattern_stack_.PopArray();
  }

  // Records that `name_id` was introduced by the current full-pattern.
  auto AddBindName(SemIR::NameId name_id) -> void {
    bind_name_stack_.AppendToTop(
        {.name_id = name_id, .inst_id = SemIR::InstId::InitTombstone});
  }

  // Records a `VarPattern` inst as part of the current full pattern, so that
  // its `VarStorage` can be allocated and tracked. This should only be called
  // when the current full-pattern is a kind that can have an initializer;
  // otherwise the `VarStorage` should be allocated on demand during pattern
  // matching.
  auto AddLocalVarPattern(SemIR::InstId var_pattern_id) -> void {
    CARBON_CHECK(kind_stack_.back() == Kind::ClassScopeVarDecl ||
                 kind_stack_.back() == Kind::NameBindingDecl);
    CARBON_CHECK(next_var_index_stack_.back() < 0);
    var_pattern_stack_.AppendToTop(
        {.pattern_id = var_pattern_id, .storage_id = SemIR::InstId::None});
  }

  // Creates `VarStorage` insts for all `VarPattern` insts recorded by
  // `AddLocalVarPattern` for the current full-pattern. This must typically
  // be called before handling the initializer (if any) for the current full-
  // pattern, in order to preserve the dominance ordering (see the comments
  // on `Check::Initialize` for details).
  auto BuildLocalVarStorage(Context& context, bool is_returned_var) -> void;

  // Returns the `VarStorage` inst that was allocated for `pattern_id` by
  // `BuildLocalVarStorage`.
  //
  // As an optimization, this assumes (and enforces) that it will be called
  // exactly once for each inst passed to `AddLocalVarPattern`, and in the same
  // order.
  auto GetLocalVarStorage(SemIR::InstId var_pattern_id) -> SemIR::InstId {
    CARBON_CHECK(kind_stack_.back() == Kind::ClassScopeVarDecl ||
                 kind_stack_.back() == Kind::NameBindingDecl);
    auto& index = next_var_index_stack_.back();
    CARBON_CHECK(index >= 0);
    auto var_info = var_pattern_stack_.PeekArray()[index];
    CARBON_CHECK(var_info.pattern_id == var_pattern_id,
                 "var patterns visited in unexpected order");
    ++index;
    return var_info.storage_id;
  }

  // Runs verification that the processing cleanly finished.
  auto VerifyOnFinish() const -> void {
    CARBON_CHECK(kind_stack_.empty(),
                 "full_pattern_stack still has {0} entries",
                 kind_stack_.size());
  }

 private:
  LexicalLookup* lookup_;

  // The stack of pending full-patterns is organized as a struct of arrays, with
  // separate stacks for separate properties of a full-pattern.

  // The kinds of the currently pending full patterns.
  llvm::SmallVector<Kind> kind_stack_;

  // Locally stashed name-lookup information about a binding.
  struct BindingInfo {
    // The name of the binding.
    SemIR::NameId name_id;
    // While handling the initializer, name lookup for `name_id` in `lookup_`
    // temporarily resolves to `InitTombstone`. During that time, this records
    // the inst that it resolved to before the initializer, so that it can
    // be restored afterward. This is `InitTombstone` while not handling the
    // initializer, or if `name_id` doesn't resolve in `lookup_`.
    SemIR::InstId inst_id;
  };

  // The name bindings introduced by the currently pending full-patterns.
  ArrayStack<BindingInfo> bind_name_stack_;

  struct VarInfo {
    SemIR::InstId pattern_id;
    SemIR::InstId storage_id;
  };

  // The `var` patterns introduced by the currently pending full-patterns.
  ArrayStack<VarInfo> var_pattern_stack_;

  // For each full pattern, the index of the first un-consumed `VarInfo` in
  // the corresponding frame of `var_pattern_stack_`, or -1 if the contents
  // of that frame are not ready for consumption.
  llvm::SmallVector<int> next_var_index_stack_;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_FULL_PATTERN_STACK_H_
