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
      : semantics_(parse_tree),
        range_(llvm::reverse(parse_tree.postorder())),
        cursor_(range_.begin()) {}

  void Build();

  void MovePastChildlessNode();

  auto NodeUnexpectedError() -> std::string;
  auto NodeKindWrongError(ParseNodeKind expected) -> std::string;

  auto TryHandleCursor(llvm::ArrayRef<NodeHandler> handlers,
                       size_t& handlers_index) -> bool;

  // Parses the children of the current node, calling handlers for the
  // respective kind and producing errors if unexpected kinds are found.
  // This is used in parsing to help guarantee that subtrees are properly
  // parsed without skipping nodes.
  void TransformCursorAsOrderedNodes(llvm::ArrayRef<NodeHandler> handlers);

  // Parses the children of the current node as a list, calling item_handler
  // per item and expecting separators.
  void TransformCursorAsList(ParseNodeKind item_kind,
                             std::function<void()> item_handler,
                             ParseNodeKind separator, ParseNodeKind list_end);

  auto TransformDeclaredName() -> Semantics::DeclaredName;
  void TransformPatternBinding();
  void TransformFunctionDeclaration(SemanticsIR::Block& block);
  void TransformParameterList();

  // Convenience accessor.
  auto parse_tree() -> const ParseTree& { return *semantics_.parse_tree_; }

  auto GetSubtreeEnd() -> ParseTreeIterator {
    auto iter = cursor_;
    iter += parse_tree().node_subtree_size(*cursor_);
    return iter;
  }

  SemanticsIR semantics_;
  llvm::iterator_range<ParseTreeIterator> range_;
  ParseTreeIterator cursor_;
};

}  // namespace Carbon

#endif  // TOOLCHAIN_SEMANTICS_SEMANTICS_IR_FACTORY_H_
