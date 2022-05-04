// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TOOLCHAIN_SEMANTICS_SEMANTICS_IR_H_
#define TOOLCHAIN_SEMANTICS_SEMANTICS_IR_H_

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/semantics/nodes/function.h"

namespace Carbon {

// Provides semantic analysis on a ParseTree.
class SemanticsIR {
 public:
  // Provides a link back to a semantic node in a name scope.
  class Node {
   public:
    Node() : Node(Kind::Invalid, -1) {}

   private:
    friend class SemanticsIR;

    // The kind of token. These correspond to the lists on SemanticsIR which
    // will be indexed into.
    enum class Kind {
      Invalid,
      Function,
    };

    Node(Kind kind, int32_t index) : kind_(kind), index_(index) {
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
    void Add(llvm::StringRef name, Node named_entity);

    auto nodes() const -> llvm::ArrayRef<Node> { return nodes_; }
    auto name_lookup() const -> const llvm::StringMap<Node>& {
      return name_lookup_;
    }

   private:
    friend class SemanticsIR;

    llvm::SmallVector<Node> nodes_;
    llvm::StringMap<Node> name_lookup_;
  };

  auto GetFunction(Node node) const -> llvm::Optional<Semantics::Function> {
    if (node.kind_ != Node::Kind::Function) {
      return llvm::None;
    }
    return functions_[node.index_];
  }

  auto root_block() const -> const Block& { return root_block_; }

 private:
  friend class SemanticsIRFactory;

  explicit SemanticsIR(const ParseTree& parse_tree)
      : parse_tree_(&parse_tree) {}

  // Creates a function and adds it to the enclosing block.
  void AddFunction(Block& block, Semantics::Function function);

  // Indexed by Token::Function.
  llvm::SmallVector<Semantics::Function, 0> functions_;

  // The file-level block.
  Block root_block_;

  const ParseTree* parse_tree_;
};

}  // namespace Carbon

#endif  // TOOLCHAIN_SEMANTICS_SEMANTICS_IR_H_
