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

  // Analyzes a parse tree and returns the constructed semantic information.
  static auto Analyze(const ParseTree& parse_tree, DiagnosticConsumer& consumer)
      -> Semantics;

 private:
  class Analyzer;
  friend class Analyzer;

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


  // All functions from the parse tree.
  llvm::SmallVector<Function, 0> functions_;

  // Names declared in the root scope.
  llvm::StringMap<NamedEntity> root_name_scope_;

  Semantics() = default;
};

}  // namespace Carbon

#endif  // TOOLCHAIN_SEMANTICS_SEMANTICS_H_
