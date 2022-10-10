// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_NODES_FUNCTION_H_
#define CARBON_TOOLCHAIN_SEMANTICS_NODES_FUNCTION_H_

#include "common/ostream.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/semantics/node_kind.h"
#include "toolchain/semantics/node_ref.h"

namespace Carbon::Semantics {

// Represents `fn name(params...) [-> return_expr] body`.
class Function {
 public:
  static constexpr NodeKind Kind = NodeKind::Function;

  Function(ParseTree::Node node, NodeId id, llvm::SmallVector<NodeRef> body)
      : node_(node), id_(id), body_(body) {}

  void Print(
      llvm::raw_ostream& out, int indent,
      std::function<void(int, llvm::ArrayRef<NodeRef>)> print_block) const {
    out << "Function(\n";
    int content_intent = indent + 4;
    out.indent(content_intent);
    out << id_ << ",\n";
    out.indent(content_intent);
    print_block(content_intent, body_);
    out << ")";
  }

  auto node() const -> ParseTree::Node { return node_; }
  auto id() const -> NodeId { return id_; }
  auto set_body(llvm::SmallVector<NodeRef, 0> body) { body_ = std::move(body); }
  auto body() const -> llvm::ArrayRef<NodeRef> { return body_; }

 private:
  // The FunctionDeclaration node.
  ParseTree::Node node_;

  // The function's ID.
  NodeId id_;

  llvm::SmallVector<NodeRef> body_;
};

}  // namespace Carbon::Semantics

#endif  // CARBON_TOOLCHAIN_SEMANTICS_NODES_FUNCTION_H_
