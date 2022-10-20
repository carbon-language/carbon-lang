// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_node.h"

namespace Carbon {

static auto PrintArgs(llvm::raw_ostream& /*out*/,
                      const SemanticsNodeArgs::None /*no_args*/) {}

static auto PrintArgs(llvm::raw_ostream& out, SemanticsNodeId one_node) {
  out << one_node;
}

static auto PrintArgs(llvm::raw_ostream& out, SemanticsTwoNodeIds two_nodes) {
  out << two_nodes.nodes[0] << ", " << two_nodes.nodes[1];
}

static auto PrintArgs(llvm::raw_ostream& out,
                      SemanticsIdentifierId identifier) {
  out << identifier;
}

static auto PrintArgs(llvm::raw_ostream& out,
                      SemanticsIntegerLiteralId identifier) {
  out << identifier;
}

void SemanticsNode::Print(llvm::raw_ostream& out) const {
  out << kind_ << "(";
  switch (kind_) {
#define CARBON_SEMANTICS_NODE_KIND(Name, Args) \
  case SemanticsNodeKind::Name():              \
    PrintArgs(out, one_of_args_.Args);         \
    break;
#include "toolchain/semantics/semantics_node_kind.def"
  }
  out << ")";
}

}  // namespace Carbon
