// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_DECL_NAME_STACK_H_
#define CARBON_TOOLCHAIN_CHECK_DECL_NAME_STACK_H_

#include "llvm/ADT/SmallVector.h"
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
// the lookup for `U` looks in `ClassA`, and the lookup for `V` looks in
// `ClassA.ClassB` then in its enclosing scope `ClassA`. Scopes entered as part
// of processing the name are exited when the name is popped from the stack.
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
  struct NameContext {
    enum class State : int8_t {
      // A context that has not processed any parts of the qualifier.
      Empty,

      // An instruction ID has been resolved, whether through an identifier or
      // expression. This provided a new scope, such as a type.
      Resolved,

      // An instruction ID has been resolved, whether through an identifier or
      // expression. It did not provide a new scope, so must be the final part,
      // such as an out-of-line function definition.
      ResolvedNonScope,

      // An identifier didn't resolve.
      Unresolved,

      // The name has already been finished. This is not set in the name
      // returned by `FinishName`, but is used internally to track that
      // `FinishName` has already been called.
      Finished,

      // An error has occurred, such as an additional qualifier past an
      // unresolved name. No new diagnostics should be emitted.
      Error,
    };

    // Returns whether the name resolved to an existing entity.
    auto is_resolved() -> bool {
      return state != State::Unresolved && state != State::Empty;
    }

    // Returns the name_id for a new instruction. This is invalid when the name
    // resolved.
    auto name_id_for_new_inst() -> SemIR::NameId {
      return !is_resolved() ? unresolved_name_id : SemIR::NameId::Invalid;
    }

    // Returns the enclosing_scope_id for a new instruction. This is invalid
    // when the name resolved. Note this is distinct from the enclosing_scope of
    // the NameContext, which refers to the scope of the introducer rather than
    // the scope of the name.
    auto enclosing_scope_id_for_new_inst() -> SemIR::NameScopeId {
      return !is_resolved() ? target_scope_id : SemIR::NameScopeId::Invalid;
    }

    // The current scope when this name began. This is the scope that we will
    // return to at the end of the declaration.
    ScopeIndex enclosing_scope;

    State state = State::Empty;

    // Whether there have been qualifiers in the name.
    bool has_qualifiers = false;

    // The scope which qualified names are added to. For unqualified names in
    // an unnamed scope, this will be Invalid to indicate the current scope
    // should be used.
    SemIR::NameScopeId target_scope_id;

    // The last location ID used.
    SemIR::LocId loc_id = SemIR::LocId::Invalid;

    union {
      // The ID of a resolved qualifier, including both identifiers and
      // expressions. Invalid indicates resolution failed.
      SemIR::InstId resolved_inst_id = SemIR::InstId::Invalid;

      // The ID of an unresolved identifier.
      SemIR::NameId unresolved_name_id;
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

  // Peeks the current target scope of the name on top of the stack. Note that
  // if we're still processing the name qualifiers, this can change before the
  // name is completed. Also, if the name up to this point was already declared
  // and is a scope, this will be that scope, rather than the scope enclosing
  // it.
  auto PeekTargetScope() const -> SemIR::NameScopeId {
    return decl_name_stack_.back().target_scope_id;
  }

  // Peeks the enclosing scope index of the name on top of the stack.
  auto PeekEnclosingScope() const -> ScopeIndex {
    return decl_name_stack_.back().enclosing_scope;
  }

  // Finishes the current declaration name processing, returning the final
  // context for adding the name to lookup.
  //
  // This also pops the final name instruction from the instruction stack,
  // which will be applied to the declaration name if appropriate.
  auto FinishName() -> NameContext;

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

  // Temporarily remove the current declaration name and its associated scopes
  // from the stack. Can only be called once the name is finished.
  auto Suspend() -> SuspendedName;

  // Restore a previously suspended name.
  auto Restore(SuspendedName sus) -> void;

  // Creates and returns a name context corresponding to declaring an
  // unqualified name in the current context. This is suitable for adding to
  // name lookup in situations where a qualified name is not permitted, such as
  // a pattern binding.
  auto MakeUnqualifiedName(SemIR::LocId loc_id, SemIR::NameId name_id)
      -> NameContext;

  // Applies a Name from the name stack to the top of the declaration name
  // stack. This will enter the scope corresponding to the name if the name
  // describes an existing scope, such as a namespace or a defined class.
  auto ApplyNameQualifier(SemIR::LocId loc_id, SemIR::NameId name_id) -> void;

  // Adds a name to name lookup. Prints a diagnostic for name conflicts.
  auto AddNameToLookup(NameContext name_context, SemIR::InstId target_id)
      -> void;

  // Adds a name to name lookup, or returns the existing instruction if this
  // name has already been declared in this scope.
  auto LookupOrAddName(NameContext name_context, SemIR::InstId target_id)
      -> SemIR::InstId;

 private:
  // Returns a name context corresponding to an empty name.
  auto MakeEmptyNameContext() -> NameContext;

  // Applies a Name from the name stack to given name context.
  auto ApplyNameQualifierTo(NameContext& name_context, SemIR::LocId loc_id,
                            SemIR::NameId name_id, bool is_unqualified) -> void;

  // Returns true if the context is in a state where it can resolve qualifiers.
  // Updates name_context as needed.
  auto TryResolveQualifier(NameContext& name_context, SemIR::LocId loc_id)
      -> bool;

  // Updates the scope on name_context as needed. This is called after
  // resolution is complete, whether for Name or expression. When updating for
  // an unqualified name, the resolution is noted without pushing scopes; it's
  // instead expected this will become a name conflict.
  auto UpdateScopeIfNeeded(NameContext& name_context, bool is_unqualified)
      -> void;

  // The linked context.
  Context* context_;

  // Provides nesting for construction.
  llvm::SmallVector<NameContext> decl_name_stack_;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_DECL_NAME_STACK_H_
