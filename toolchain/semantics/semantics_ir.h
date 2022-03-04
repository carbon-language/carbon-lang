// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TOOLCHAIN_SEMANTICS_SEMANTICS_IR_H_
#define TOOLCHAIN_SEMANTICS_SEMANTICS_IR_H_

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/semantics/function.h"

namespace Carbon {

// Provides semantic analysis on a ParseTree.
class SemanticsIR {
 public:
  // Provides a link back to an entity in a name scope.
  class NamedEntity {
   public:
    NamedEntity() : NamedEntity(Kind::Invalid, -1) {}

   private:
    friend class SemanticsIR;

    // The kind of entity. There should be one entry per list of entities that
    // needs to be indexed into.
    enum class Kind {
      Invalid,
      Function,
    };

    NamedEntity(Kind kind, int32_t index) : kind_(kind), index_(index) {
      // TODO: kind_ and index_ are currently unused, this suppresses the
      // warning.
      kind_ = kind;
      index_ = index;
    }

    Kind kind_;

    // The index of the named entity within its list.
    int32_t index_;
  };

  struct Block {
   public:
    void Add(llvm::StringRef name, NamedEntity entity);

   private:
    llvm::SmallVector<NamedEntity> ordering_;
    llvm::StringMap<NamedEntity> name_lookup_;
  };

 private:
  friend class SemanticsIRFactory;

  explicit SemanticsIR(const ParseTree& parse_tree)
      : parse_tree_(&parse_tree) {}

  // Creates a function, adds it to the enclosing scope, and returns a reference
  // for further mutations. On a name collision, it will not be added to the
  // scope, but will still be returned.
  auto AddFunction(Block& block, ParseTree::Node decl_node,
                   ParseTree::Node name_node) -> Semantics::Function&;

  // All functions from the parse tree.
  llvm::SmallVector<Semantics::Function, 0> functions_;

  // The file-level block.
  Block root_block_;

  const ParseTree* parse_tree_;
};

}  // namespace Carbon

#endif  // TOOLCHAIN_SEMANTICS_SEMANTICS_IR_H_
