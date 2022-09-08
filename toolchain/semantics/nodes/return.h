// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_NODES_RETURN_H_
#define CARBON_TOOLCHAIN_SEMANTICS_NODES_RETURN_H_

#include "common/ostream.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/semantics/node_kind.h"
#include "toolchain/semantics/node_ref.h"

namespace Carbon::Semantics {

// Represents `return [expr];`
class Return {
 public:
  static constexpr NodeKind Kind = NodeKind::Return;

  Return(ParseTree::Node node, llvm::Optional<NodeId> target_id)
      : node_(node), target_id_(target_id) {}

  void Print(llvm::raw_ostream& out) const {
    out << "Return(";
    if (target_id_) {
      out << *target_id_;
    } else {
      out << "None";
    }
    out << ")";
  }

  auto node() const -> ParseTree::Node { return node_; }
  auto target_id() const -> const llvm::Optional<NodeId>& { return target_id_; }

 private:
  ParseTree::Node node_;
  llvm::Optional<NodeId> target_id_;
};

}  // namespace Carbon::Semantics

#endif  // CARBON_TOOLCHAIN_SEMANTICS_NODES_RETURN_H_
