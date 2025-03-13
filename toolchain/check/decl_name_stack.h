// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_DECL_NAME_STACK_H_
#define CARBON_TOOLCHAIN_CHECK_DECL_NAME_STACK_H_

#include "llvm/ADT/SmallVector.h"
#include "toolchain/check/name_component.h"
#include "toolchain/check/scope_index.h"
#include "toolchain/check/scope_stack.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

class Context;

// Provides support and stacking for qualified declaration name handling.
//
// A qualified declaration name will consist of entries, which are `Name`s
// optionally followed by generic parameter lists, such as `Vector(T:! type)`
// in `fn Vector(T:! type).Clear();`, but parameter lists aren't supported yet.
// Identifiers such as `Clear` will be resolved to a name if possible, for
// example when declaring things that are in a non-generic type or namespace,
// and are otherwise marked as an unresolved identifier.
//
// Unresolved identifiers are valid if and only if they are the last step of a
// qualified name; all resolved qualifiers must resolve to an entity with
// members, such as a namespace. Resolved identifiers in the last step will
// occur for both out-of-line definitions and new declarations, depending on
// context.
//
// For each name component that is processed and denotes a scope, the
// corresponding scope is also entered. This is important for unqualified name
// lookup both in the definition of the entity being declared, and for names
// appearing later in the declaration name itself. For example, in:
//
// ```
// fn ClassA.ClassB(T:! U).Fn() { var x: V; }
// ```
//
// the lookup for `U` looks in `ClassA`; the lookup for `V` looks first in
// `ClassA.ClassB`, then its parent scope `ClassA`. Scopes entered as part of
// processing the name are exited when the name is popped from the stack.
//
// Example state transitions:
//
// ```
// // Empty -> Unresolved, because `MyNamespace` is newly declared.
// namespace MyNamespace;
//
// // Empty -> Resolved -> Unresolved, because `MyType` is newly declared.
// class MyNamespace.MyType;
//
// // Empty -> Resolved -> Resolved, because `MyType` was forward declared.
// class MyNamespace.MyType {
//   // Empty -> Unresolved, because `DoSomething` is newly declared.
//   fn DoSomething();
// }
//
// // Empty -> Resolved -> Resolved -> ResolvedNonScope, because `DoSomething`
// // is forward declared in `MyType`, but is not a scope itself.
// fn MyNamespace.MyType.DoSomething() { ... }
// ```
class DeclNameStack {
 public:
  // Context for declaration name construction.
  // TODO: Add a helper for class, function, and interface to turn a NameContext
  // into an EntityWithParamsBase.
  struct NameContext {
    enum class State : int8_t {
      // A context that has not processed any parts of the qualifier.
      Empty,

      // The name has been resolved to an instruction ID.
      Resolved,

      // An identifier didn't resolve.
      Unresolved,

      // An identifier was poisoned in this scope.
      Poisoned,

      // The name has already been finished. This is not set in the name
      // returned by `FinishName`, but is used internally to track that
      // `FinishName` has already been called.
      Finished,

      // An error has occurred, such as an additional qualifier past an
      // unresolved name. No new diagnostics should be emitted.
      Error,
    };

    // Combines name information to produce a base struct for entity
    // construction.
    auto MakeEntityWithParamsBase(const NameComponent& name,
                                  SemIR::InstId decl_id, bool is_extern,
                                  SemIR::LibraryNameId extern_library) const
        -> SemIR::EntityWithParamsBase {
      return {
          .name_id = name_id_for_new_inst(),
          .parent_scope_id = parent_scope_id,
          .generic_id = SemIR::GenericId::None,
          .first_param_node_id = name.first_param_node_id,
          .last_param_node_id = name.last_param_node_id,
          .pattern_block_id = name.pattern_block_id,
          .implicit_param_patterns_id = name.implicit_param_patterns_id,
          .param_patterns_id = name.param_patterns_id,
          .call_params_id = name.call_params_id,
          .is_extern = is_extern,
          .extern_library_id = extern_library,
          .non_owning_decl_id =
              extern_library.has_value() ? decl_id : SemIR::InstId::None,
          .first_owning_decl_id =
              extern_library.has_value() ? SemIR::InstId::None : decl_id,
      };
    }

    // Returns any name collision found, or `None`. Requires a non-poisoned
    // value.
    auto prev_inst_id() const -> SemIR::InstId;

    // Returns the name_id for a new instruction. This is `None` when the name
    // resolved.
    auto name_id_for_new_inst() const -> SemIR::NameId {
      switch (state) {
        case State::Unresolved:
        case State::Poisoned:
          return name_id;
        default:
          return SemIR::NameId::None;
      }
    }

    // The current scope when this name began. This is the scope that we will
    // return to at the end of the declaration.
    ScopeIndex initial_scope_index;

    State state = State::Empty;

    // Whether there have been qualifiers in the name.
    bool has_qualifiers = false;

    // The scope which qualified names are added to. For unqualified names in
    // an unnamed scope, this will be `None` to indicate the current scope
    // should be used.
    SemIR::NameScopeId parent_scope_id;

    // The location of the final name component.
    SemIR::LocId loc_id = SemIR::LocId::None;

    // The name of the final name component.
    SemIR::NameId name_id = SemIR::NameId::None;

    union {
      // The ID of a resolved qualifier, including both identifiers and
      // expressions. `None` indicates resolution failed.
      SemIR::InstId resolved_inst_id;

      // When `state` is `Poisoned` (name is unresolved due to name poisoning),
      // the poisoning location.
      SemIR::LocId poisoning_loc_id = SemIR::LocId::None;
    };
  };

  // Information about a declaration name that has been temporarily removed from
  // the stack and will later be restored. Names can only be suspended once they
  // are finished.
  struct SuspendedName {
    // The declaration name information.
    NameContext name_context;

    // Suspended scopes. We only preallocate space for two of these, because
    // suspended names are usually used for classes and functions with
    // unqualified names, which only need at most two scopes -- one scope for
    // the parameter and one scope for the entity itself, and we can store quite
    // a few of these when processing a large class definition.
    llvm::SmallVector<ScopeStack::SuspendedScope, 2> scopes;
  };

  explicit DeclNameStack(Context* context) : context_(context) {}

  // Pushes processing of a new declaration name, which will be used
  // contextually, and prepares to enter scopes for that name. To pop this
  // state, `FinishName` and `PopScope` must be called, in that order.
  auto PushScopeAndStartName() -> void;

  // Creates and returns a name context corresponding to declaring an
  // unqualified name in the current context. This is suitable for adding to
  // name lookup in situations where a qualified name is not permitted, such as
  // a pattern binding.
  auto MakeUnqualifiedName(SemIR::LocId loc_id, SemIR::NameId name_id)
      -> NameContext;

  // Applies a name component as a qualifier for the current name. This will
  // enter the scope corresponding to the name if the name describes an existing
  // scope, such as a namespace or a defined class.
  auto ApplyNameQualifier(const NameComponent& name) -> void;

  // Finishes the current declaration name processing, returning the final
  // context for adding the name to lookup. The final name component should be
  // popped and passed to this function, and will be added to the declaration
  // name.
  auto FinishName(const NameComponent& name) -> NameContext;

  // Finishes the current declaration name processing for an `impl`, returning
  // the final context for adding the name to lookup.
  //
  // `impl`s don't actually have names, but want the rest of the name processing
  // logic such as building parameter scopes, so are a special case.
  auto FinishImplName() -> NameContext;

  // Pops the declaration name from the declaration name stack, and pops all
  // scopes that were entered as part of handling the declaration name. These
  // are the scopes corresponding to name qualifiers in the name, for example
  // the `A.B` in `fn A.B.F()`.
  //
  // This should be called at the end of the declaration.
  auto PopScope() -> void;

  // Peeks the current parent scope of the name on top of the stack. Note
  // that if we're still processing the name qualifiers, this can change before
  // the name is completed. Also, if the name up to this point was already
  // declared and is a scope, this will be that scope, rather than the scope
  // containing it.
  auto PeekParentScopeId() const -> SemIR::NameScopeId {
    return decl_name_stack_.back().parent_scope_id;
  }

  // Peeks the resolution scope index of the name on top of the stack.
  auto PeekInitialScopeIndex() const -> ScopeIndex {
    return decl_name_stack_.back().initial_scope_index;
  }

  // Temporarily remove the current declaration name and its associated scopes
  // from the stack. Can only be called once the name is finished.
  auto Suspend() -> SuspendedName;

  // Restore a previously suspended name.
  auto Restore(SuspendedName sus) -> void;

  // Adds a name to name lookup. Assumes duplicates are already handled.
  auto AddName(NameContext name_context, SemIR::InstId target_id,
               SemIR::AccessKind access_kind) -> void;

  // Adds a name to name lookup. Prints a diagnostic for name conflicts.
  auto AddNameOrDiagnose(NameContext name_context, SemIR::InstId target_id,
                         SemIR::AccessKind access_kind) -> void;

  // Adds a name to name lookup if neither already declared nor poisoned in this
  // scope.
  auto LookupOrAddName(NameContext name_context, SemIR::InstId target_id,
                       SemIR::AccessKind access_kind)
      -> SemIR::ScopeLookupResult;

  // Runs verification that the processing cleanly finished.
  auto VerifyOnFinish() const -> void {
    CARBON_CHECK(decl_name_stack_.empty(),
                 "decl_name_stack still has {0} entries",
                 decl_name_stack_.size());
  }

 private:
  // Returns a name context corresponding to an empty name.
  auto MakeEmptyNameContext() -> NameContext;

  // Appends a name to the given name context, and performs a lookup to find
  // what, if anything, the name refers to.
  auto ApplyAndLookupName(NameContext& name_context, SemIR::LocId loc_id,
                          SemIR::NameId name_id) -> void;

  // Attempts to resolve the given name context as a scope, and returns the
  // corresponding scope. Issues a suitable diagnostic and returns `None` if
  // the name doesn't resolve to a scope.
  auto ResolveAsScope(const NameContext& name_context,
                      const NameComponent& name) const
      -> std::pair<SemIR::NameScopeId, SemIR::SpecificId>;

  // The linked context.
  Context* context_;

  // Provides nesting for construction.
  llvm::SmallVector<NameContext> decl_name_stack_;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_DECL_NAME_STACK_H_
