// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_node.h"

namespace Carbon {

static auto PrintArgs(llvm::raw_ostream& /*out*/,
                      const SemanticsNode::NoArgs /*no_args*/) {}

static auto PrintArgs(llvm::raw_ostream& out, SemanticsNodeId arg0) {
  out << arg0;
}

static auto PrintArgs(llvm::raw_ostream& out,
                      SemanticsIdentifierId identifier) {
  out << identifier;
}

static auto PrintArgs(llvm::raw_ostream& out,
                      SemanticsIntegerLiteralId integer_literal) {
  out << integer_literal;
}

static auto PrintArgs(llvm::raw_ostream& out, SemanticsNodeBlockId node_block) {
  out << node_block;
}

template <typename T0, typename T1>
static auto PrintArgs(llvm::raw_ostream& out, std::pair<T0, T1> args) {
  PrintArgs(out, args.first);
  out << ", ";
  PrintArgs(out, args.second);
}

void SemanticsNode::Print(llvm::raw_ostream& out) const {
  out << kind_ << "(";
  switch (kind_) {
#define CARBON_SEMANTICS_NODE_KIND(Name) \
  case SemanticsNodeKind::Name():        \
    PrintArgs(out, Get##Name());         \
    break;
#include "toolchain/semantics/semantics_node_kind.def"
  }
  out << ")";
}

}  // namespace Carbon
