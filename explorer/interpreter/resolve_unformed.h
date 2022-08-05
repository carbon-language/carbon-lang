// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_INTERPRETER_RESOLVE_UNFORMED_H_
#define CARBON_EXPLORER_INTERPRETER_RESOLVE_UNFORMED_H_

#include <optional>

#include "explorer/ast/ast.h"
#include "explorer/common/nonnull.h"

namespace Carbon {

// Maps AST nodes to flow facts within a function.
class FlowFacts {
 public:
  enum class ActionType {
    // Used at `VariableDefinition` with initialization.
    AddInit,
    // Used at `VariableDefinition` without initialization.
    AddUninit,
    // Used at AST nodes that potentially initializes a variable.
    Form,
    // Used at AST nodes that uses a variable.
    Check,
    // Used in traversing children nodes without an acion to take.
    None,
  };
  // Take action on flow facts based on `ActionType`.
  auto TakeAction(Nonnull<const AstNode*> node, ActionType action,
                  SourceLocation source_loc, const std::string& name)
      -> ErrorOr<Success>;

 private:
  enum class FormedState {
    MustBeFormed,
    MayBeFormed,
    Unformed,
  };
  // Aggregate information about a AstNode being analyzed.
  struct Fact {
    FormedState formed_state;
  };

  auto get_fact(Nonnull<const AstNode*> node) const
      -> std::optional<Nonnull<const Fact*>> {
    auto entry = facts_.find(node);
    if (entry != facts_.end()) {
      return &entry->second;
    }
    return std::nullopt;
  }

  void add_fact(Nonnull<const AstNode*> node, const FormedState state) {
    CARBON_CHECK(!get_fact(node).has_value());
    facts_.insert({node, {state}});
  }

  // Adds a must-be-formed flow fact .
  void AddInitFact(Nonnull<const AstNode*> node) {
    add_fact(node, FormedState::MustBeFormed);
  }
  // Adds an unformed flow fact.
  void AddUninitFact(Nonnull<const AstNode*> node) {
    add_fact(node, FormedState::Unformed);
  }
  // Marks an unformed flow fact as may-be-formed.
  void FormFact(Nonnull<const AstNode*> node) {
    // TODO: Use CARBON_CHECK when we are able to handle global variables.
    if (get_fact(node).has_value() &&
        facts_[node].formed_state == FormedState::Unformed) {
      facts_[node].formed_state = FormedState::MayBeFormed;
    }
  }
  // Returns compilation error if the AST node is impossible to be formed.
  auto CheckFact(Nonnull<const AstNode*> node, SourceLocation source_loc,
                 const std::string& name) const -> ErrorOr<Success> {
    // TODO: @slaterlatiao add all available value nodes to flow facts and use
    // CARBON_CHECK on the following line.
    std::optional<Nonnull<const Fact*>> fact = get_fact(node);
    if (fact.has_value() && (*fact)->formed_state == FormedState::Unformed) {
      return CompilationError(source_loc)
             << "use of uninitialized variable " << name;
    }
    return Success();
  }

  std::unordered_map<Nonnull<const AstNode*>, Fact> facts_;
};

// An intraprocedural forward analysis that checks the may-be-formed states on
// local variables. Returns compilation error on usage of must-be-unformed
// variables.
auto ResolveUnformed(const AST& ast) -> ErrorOr<Success>;

}  // namespace Carbon

#endif  // CARBON_EXPLORER_INTERPRETER_RESOLVE_UNFORMED_H_
