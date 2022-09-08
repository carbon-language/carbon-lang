// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_NODES_SET_NAME_H_
#define CARBON_TOOLCHAIN_SEMANTICS_NODES_SET_NAME_H_

#include "common/ostream.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/semantics/node_kind.h"

namespace Carbon::Semantics {

// Represents `fn name(params...) [-> return_expr] body`.
class SetName {
 public:
  static constexpr NodeKind Kind = NodeKind::SetName;

  SetName(ParseTree::Node node, llvm::StringRef name, NodeId target_id)
      : node_(node), name_(name), target_id_(target_id) {}

  void Print(llvm::raw_ostream& out) const {
    out << "SetName(`" << name_ << "`, " << target_id_ << ")";
  }

  auto node() const -> ParseTree::Node { return node_; }
  auto name() const -> llvm::StringRef { return name_; }
  auto target_id() const -> NodeId { return target_id_; }

 private:
  // The name node.
  ParseTree::Node node_;

  // The name to assign.
  llvm::StringRef name_;

  // The ID being named.
  NodeId target_id_;
};

}  // namespace Carbon::Semantics

#endif  // CARBON_TOOLCHAIN_SEMANTICS_NODES_SET_NAME_H_
