// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TOOLCHAIN_SEMANTICS_SEMANTICS_IR_H_
#define TOOLCHAIN_SEMANTICS_SEMANTICS_IR_H_

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/semantics/nodes/function.h"
#include "toolchain/semantics/nodes/pattern_binding.h"

namespace Carbon {

namespace Testing {
class SemanticsIRSingleton;
}  // namespace Testing

// Provides semantic analysis on a ParseTree.
class SemanticsIR {
 public:
  // Provides a link back to a semantic node in a name scope.
  class Node {
   public:
    Node() : Node(Kind::Invalid, -1) {}

   private:
    friend class SemanticsIR;
    friend class Testing::SemanticsIRSingleton;

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
    friend class SemanticsIRFactory;

    llvm::SmallVector<Node> nodes_;
    llvm::StringMap<Node> name_lookup_;
  };

  void Print(llvm::raw_ostream& out, Node node) const;

  auto root_block() const -> const Block& { return root_block_; }

 private:
  friend class SemanticsIRFactory;
  friend class Testing::SemanticsIRSingleton;

  explicit SemanticsIR(const ParseTree& parse_tree)
      : parse_tree_(&parse_tree) {}

  // Creates a function and adds it to the enclosing block.
  void AddFunction(Block& block, Semantics::Function function);

  void Print(llvm::raw_ostream& out, ParseTree::Node node) const;
  void Print(llvm::raw_ostream& out, Semantics::DeclaredName name) const;
  void Print(llvm::raw_ostream& out, Semantics::Expression expr) const;
  void Print(llvm::raw_ostream& out, Semantics::Function function) const;
  void Print(llvm::raw_ostream& out, Semantics::Literal literal) const;
  void Print(llvm::raw_ostream& out, Semantics::PatternBinding binding) const;

  // Indexed by Token::Function.
  llvm::SmallVector<Semantics::Function, 0> functions_;

  // The file-level block.
  Block root_block_;

  const ParseTree* parse_tree_;
};

}  // namespace Carbon

#endif  // TOOLCHAIN_SEMANTICS_SEMANTICS_IR_H_
