// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TOOLCHAIN_SEMANTICS_SEMANTICS_H_
#define TOOLCHAIN_SEMANTICS_SEMANTICS_H_

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "toolchain/parser/parse_tree.h"

namespace Carbon {

// Provides semantic analysis on a ParseTree.
class Semantics {
 public:
  // Semantic information for a function.
  struct Function {
    // The node corresponding to the function name.
    ParseTree::Node name_node;
  };

  // Provides a link back to an entity in a name scope.
  struct NamedEntity {
    // The kind of entity. There should be one entry per list of entities that
    // needs to be indexed into.
    enum class Kind {
      Function,
    };

    Kind kind;

    // The index of the named entity within its list.
    int32_t index;
  };

 private:
  friend class SemanticAnalyzer;

  explicit Semantics(const ParseTree& parse_tree) : parse_tree_(&parse_tree) {}

  // Creates a function, adds it to the enclosing scope, and returns a reference
  // for further mutations. On a name collision, it will not be added to the
  // scope, but will still be returned.
  auto AddFunction(DiagnosticEmitter<ParseTree::Node> emitter,
                   llvm::StringMap<NamedEntity>& enclosing_name_scope,
                   llvm::StringRef name) -> Function&;

  // All functions from the parse tree.
  llvm::SmallVector<Function, 0> functions_;

  // Names declared in the root scope.
  llvm::StringMap<NamedEntity> root_name_scope_;

  const ParseTree* parse_tree_;
};

}  // namespace Carbon

#endif  // TOOLCHAIN_SEMANTICS_SEMANTICS_H_
