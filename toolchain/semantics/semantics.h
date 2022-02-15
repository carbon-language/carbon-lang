// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TOOLCHAIN_SEMANTICS_SEMANTICS_H_
#define TOOLCHAIN_SEMANTICS_SEMANTICS_H_

#include "toolchain/parser/parse_tree.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

namespace Carbon {

// Provides semantic analysis on a ParseTree.
class Semantics {
 public:
  struct Function {
    ParseTree::Node name_node;
  };

  struct Entity {
    enum class Category {
      Function,
    };

    Category category_;
    int32_t index_;
  };

  // Analyzes a parse tree and returns the constructed semantic information.
  static auto Analyze(const ParseTree& parse_tree, DiagnosticConsumer& consumer)
      -> Semantics;

 private:
  class Analyzer;
  friend class Analyzer;
  
  llvm::SmallVector<Function, 0> functions_;
  // Names declared in the root scope.
  llvm::StringMap<Entity> root_name_scope_;

  Semantics() = default;
};

}  // namespace Carbon

#endif  // TOOLCHAIN_SEMANTICS_SEMANTICS_H_
