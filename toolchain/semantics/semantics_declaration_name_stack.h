// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_DECLARATION_NAME_STACK_H_
#define CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_DECLARATION_NAME_STACK_H_

#include "llvm/ADT/SmallVector.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/semantics/semantics_node.h"

namespace Carbon {

class SemanticsContext;

// Provides support and stacking for qualified declaration name handling.
//
// A qualified declaration name will consist of entries which are either
// Identifiers or full expressions. Expressions are expected to resolve to
// types, such as how `fn Vector(i32).Clear() { ... }` uses the expression
// `Vector(i32)` to indicate the type whose member is being declared.
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
// // New -> Unresolved, because `MyNamespace` is newly declared.
// namespace MyNamespace;
//
// // New -> Resolved -> Unresolved, because `MyType` is newly declared.
// class MyNamespace.MyType;
//
// // New -> Resolved -> Resolved, because `MyType` was forward declared.
// class MyNamespace.MyType {
//   // New -> Unresolved, because `DoSomething` is newly declared.
//   fn DoSomething();
// }
//
// // New -> Resolved -> Resolved -> ResolvedNonScope, because `DoSomething`
// // is forward declared in `MyType`, but is not a scope itself.
// fn MyNamespace.MyType.DoSomething() { ... }
// ```
class SemanticsDeclarationNameStack {
 public:
  // Context for declaration name construction.
  struct Context {
    enum class State {
      // A new context which has not processed any parts of the qualifier.
      New,

      // A node ID has been resolved, whether through an identifier or
      // expression. This provided a new scope, such as a type.
      Resolved,

      // A node ID has been resolved, whether through an identifier or
      // expression. It did not provide a new scope, so must be the final part,
      // such as an out-of-line function definition.
      ResolvedNonScope,

      // An identifier didn't resolve.
      Unresolved,

      // An error has occurred, such as an additional qualifier past an
      // unresolved name. No new diagnostics should be emitted.
      Error,
    };

    State state = State::New;

    // The scope which qualified names are added to. For unqualified names,
    // this will be Invalid to indicate the current scope should be used.
    SemanticsNameScopeId target_scope_id = SemanticsNameScopeId::Invalid;

    // The last parse node used.
    ParseTree::Node parse_node = ParseTree::Node::Invalid;

    union {
      // The ID of a resolved qualifier, including both identifiers and
      // expressions. Invalid indicates resolution failed.
      SemanticsNodeId resolved_node_id = SemanticsNodeId::Invalid;

      // The ID of an unresolved identifier.
      SemanticsStringId unresolved_name_id;
    };
  };

  explicit SemanticsDeclarationNameStack(SemanticsContext* context)
      : context_(context) {}

  // Pushes processing of a new declaration name, which will be used
  // contextually.
  auto Push() -> void;

  // Pops the current declaration name processing, returning the final context
  // for adding the name to lookup. This also pops the final name node from the
  // node stack, which will be applied to the declaration name if appropriate.
  auto Pop() -> Context;

  // Applies an expression from the node stack to the top of the declaration
  // name stack.
  auto ApplyExpressionQualifier(ParseTree::Node parse_node,
                                SemanticsNodeId node_id) -> void;

  // Applies a Name from the node stack to the top of the declaration name
  // stack.
  auto ApplyNameQualifier(ParseTree::Node parse_node, SemanticsStringId name_id)
      -> void;

  // Adds a name to name lookup. Prints a diagnostic for name conflicts.
  auto AddNameToLookup(Context name_context, SemanticsNodeId target_id) -> void;

 private:
  // Returns true if the context is in a state where it can resolve qualifiers.
  // Updates name_context as needed.
  auto CanResolveQualifier(Context& name_context, ParseTree::Node parse_node)
      -> bool;

  // Updates the scope on name_context as needed. This is called after
  // resolution is complete, whether for Name or expression.
  auto UpdateScopeIfNeeded(Context& name_context) -> void;

  // The linked context.
  SemanticsContext* context_;

  // Provides nesting for construction.
  llvm::SmallVector<Context> declaration_name_stack_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_DECLARATION_NAME_STACK_H_
