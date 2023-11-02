// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_DECLARATION_NAME_STACK_H_
#define CARBON_TOOLCHAIN_CHECK_DECLARATION_NAME_STACK_H_

#include "llvm/ADT/SmallVector.h"
#include "toolchain/parse/tree.h"
#include "toolchain/sem_ir/inst.h"

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
class DeclarationNameStack {
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

      // An error has occurred, such as an additional qualifier past an
      // unresolved name. No new diagnostics should be emitted.
      Error,
    };

    State state = State::Empty;

    // The scope which qualified names are added to. For unqualified names in
    // an unnamed scope, this will be Invalid to indicate the current scope
    // should be used.
    SemIR::NameScopeId target_scope_id;

    // The last parse node used.
    Parse::Node parse_node = Parse::Node::Invalid;

    union {
      // The ID of a resolved qualifier, including both identifiers and
      // expressions. Invalid indicates resolution failed.
      SemIR::InstId resolved_inst_id = SemIR::InstId::Invalid;

      // The ID of an unresolved identifier.
      IdentifierId unresolved_name_id;
    };
  };

  explicit DeclarationNameStack(Context* context) : context_(context) {}

  // Pushes processing of a new declaration name, which will be used
  // contextually.
  auto Push() -> void;

  // Pops the current declaration name processing, returning the final context
  // for adding the name to lookup. This also pops the final name node from the
  // name stack, which will be applied to the declaration name if appropriate.
  auto Pop() -> NameContext;

  // Creates and returns a name context corresponding to declaring an
  // unqualified name in the current context. This is suitable for adding to
  // name lookup in situations where a qualified name is not permitted, such as
  // a pattern binding.
  auto MakeUnqualifiedName(Parse::Node parse_node, IdentifierId name_id)
      -> NameContext;

  // Applies a Name from the name stack to the top of the declaration name
  // stack.
  auto ApplyNameQualifier(Parse::Node parse_node, IdentifierId name_id) -> void;

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
  auto ApplyNameQualifierTo(NameContext& name_context, Parse::Node parse_node,
                            IdentifierId name_id) -> void;

  // Returns true if the context is in a state where it can resolve qualifiers.
  // Updates name_context as needed.
  auto CanResolveQualifier(NameContext& name_context, Parse::Node parse_node)
      -> bool;

  // Updates the scope on name_context as needed. This is called after
  // resolution is complete, whether for Name or expression.
  auto UpdateScopeIfNeeded(NameContext& name_context) -> void;

  // The linked context.
  Context* context_;

  // Provides nesting for construction.
  llvm::SmallVector<NameContext> declaration_name_stack_;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_DECLARATION_NAME_STACK_H_
