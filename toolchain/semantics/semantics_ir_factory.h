// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TOOLCHAIN_SEMANTICS_SEMANTICS_IR_FACTORY_H_
#define TOOLCHAIN_SEMANTICS_SEMANTICS_IR_FACTORY_H_

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/iterator_range.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/semantics/declared_name.h"
#include "toolchain/semantics/semantics_ir.h"

namespace Carbon {

// The main semantic analysis entry.
class SemanticsIRFactory {
 public:
  // Builds the SemanticsIR without doing any substantial semantic analysis.
  static auto Build(const ParseTree& parse_tree) -> SemanticsIR;

 private:
  using ParseTreeIterator = std::reverse_iterator<ParseTree::PostorderIterator>;

  enum class Ordering {
    Required,
    Optional,
    Repeated,
  };

  struct NodeHandler {
    ParseNodeKind kind;
    std::function<void()> handler;
    Ordering ordering = Ordering::Required;
  };

  explicit SemanticsIRFactory(const ParseTree& parse_tree)
      : semantics_(parse_tree) {}

  void Build();

  // New
  void RequireNodeEmpty(ParseTree::Node node);

  // Parses the children of the current node, calling handlers for the
  // respective kind and producing errors if unexpected kinds are found.
  // This is used in parsing to help guarantee that subtrees are properly
  // parsed without skipping nodes.
  void TransformChildrenAsOrderedNodes(llvm::ArrayRef<NodeHandler> handlers);

  // Parses the children of the current node as a list, calling item_handler
  // per item and expecting separators.
  void TransformChildrenAsList(ParseNodeKind item_kind,
                               std::function<void()> item_handler,
                               ParseNodeKind separator, ParseNodeKind list_end);

  auto TransformDeclaredName(ParseTree::Node node) -> Semantics::DeclaredName;
  void TransformFunctionDeclaration(ParseTree::Node node,
                                    SemanticsIR::Block& block);
  void TransformParameterList(ParseTree::Node node);
  void TransformPattern(ParseTree::Node node);
  void TransformPatternBinding(ParseTree::Node node);

  // Convenience accessor.
  auto parse_tree() -> const ParseTree& { return *semantics_.parse_tree_; }

  SemanticsIR semantics_;
};

}  // namespace Carbon

#endif  // TOOLCHAIN_SEMANTICS_SEMANTICS_IR_FACTORY_H_
