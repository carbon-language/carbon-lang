// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_NODES_INTEGER_LITERAL_H_
#define CARBON_TOOLCHAIN_SEMANTICS_NODES_INTEGER_LITERAL_H_

#include "common/ostream.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/semantics/node_kind.h"

namespace Carbon::Semantics {

// Represents all kinds of literals: `1`, `i32`, etc.
class IntegerLiteral {
 public:
  static constexpr NodeKind Kind = NodeKind::IntegerLiteral;

  explicit IntegerLiteral(ParseTree::Node node, NodeId id,
                          const llvm::APInt& value)
      : node_(node), id_(id), value_(&value) {}

  void Print(llvm::raw_ostream& out) const {
    out << "IntegerLiteral(" << id_ << ", " << *value_ << ")";
  }

  auto node() const -> ParseTree::Node { return node_; }
  auto id() const -> NodeId { return id_; }
  auto value() const -> const llvm::APInt& { return *value_; }

 private:
  ParseTree::Node node_;
  NodeId id_;
  const llvm::APInt* value_;
};

}  // namespace Carbon::Semantics

#endif  // CARBON_TOOLCHAIN_SEMANTICS_NODES_INTEGER_LITERAL_H_
