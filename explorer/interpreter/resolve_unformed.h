// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_INTERPRETER_RESOLVE_UNFORMED_H_
#define CARBON_EXPLORER_INTERPRETER_RESOLVE_UNFORMED_H_

#include "explorer/ast/ast.h"
#include "explorer/common/nonnull.h"

namespace Carbon {

// Maps AST nodes to flow facts within a function.
class FlowFacts {
 public:
  enum class ActionType {
    // Adds a must-be-formed flow fact.
    // Used at `VariableDefinition` with initialization.
    AddInit,
    // Adds an unformed flow fact.
    // Used at `VariableDefinition` without initialization.
    AddUninit,
    // Marks an unformed flow fact as may-be-formed.
    // Used at AST nodes that potentially initializes a variable.
    Form,
    // Returns compilation error if the AST node is impossible to be formed.
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

  void AddFact(Nonnull<const AstNode*> node, const FormedState state) {
    CARBON_CHECK(facts_.find(node) == facts_.end());
    facts_.insert({node, {state}});
  }

  std::unordered_map<Nonnull<const AstNode*>, Fact> facts_;
};

// An intraprocedural forward analysis that checks the may-be-formed states on
// local variables. Returns compilation error on usage of must-be-unformed
// variables.
auto ResolveUnformed(const AST& ast) -> ErrorOr<Success>;

}  // namespace Carbon

#endif  // CARBON_EXPLORER_INTERPRETER_RESOLVE_UNFORMED_H_
