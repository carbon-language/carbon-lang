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
  class Function {
   public:
    explicit Function(ParseTree::Node decl_node, ParseTree::Node name_node)
        : decl_node_(decl_node), name_node_(name_node) {}

    auto decl_node() const -> ParseTree::Node { return decl_node_; }
    auto name_node() const -> ParseTree::Node { return name_node_; }

   private:
    // The FunctionDeclaration node.
    ParseTree::Node decl_node_;

    // The function's DeclaredName node.
    ParseTree::Node name_node_;
  };

  // Provides a link back to an entity in a name scope.
  class NamedEntity {
   public:
    NamedEntity() : NamedEntity(Kind::Invalid, -1) {}

   private:
    friend class Semantics;

    // The kind of entity. There should be one entry per list of entities that
    // needs to be indexed into.
    enum class Kind {
      Invalid,
      Function,
    };

    NamedEntity(Kind kind, int32_t index) : kind_(kind), index_(index) {}

    Kind kind_;

    // The index of the named entity within its list.
    int32_t index_;
  };

 private:
  friend class SemanticAnalyzer;

  explicit Semantics(const ParseTree& parse_tree) : parse_tree_(&parse_tree) {}

  // Creates a function, adds it to the enclosing scope, and returns a reference
  // for further mutations. On a name collision, it will not be added to the
  // scope, but will still be returned.
  auto AddFunction(DiagnosticEmitter<ParseTree::Node>& emitter,
                   llvm::StringMap<NamedEntity>& enclosing_name_scope,
                   ParseTree::Node decl_node, ParseTree::Node name_node)
      -> Function&;

  // Returns the location of an entity. This assists diagnostic output where
  // supplemental locations are provided in formatting.
  auto GetEntityLocation(NamedEntity entity) -> Diagnostic::Location;

  // All functions from the parse tree.
  llvm::SmallVector<Function, 0> functions_;

  // Names declared in the root scope.
  llvm::StringMap<NamedEntity> root_name_scope_;

  const ParseTree* parse_tree_;
};

}  // namespace Carbon

#endif  // TOOLCHAIN_SEMANTICS_SEMANTICS_H_
