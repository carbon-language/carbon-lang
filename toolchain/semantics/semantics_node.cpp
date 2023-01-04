// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_node.h"

#include "toolchain/semantics/semantics_builtin_kind.h"

namespace Carbon {

static auto PrintArgs(llvm::raw_ostream& /*out*/,
                      const SemanticsNode::NoArgs /*no_args*/) {}

template <typename T>
static auto PrintArgs(llvm::raw_ostream& out, T arg) {
  out << arg;
}

template <typename T0, typename T1>
static auto PrintArgs(llvm::raw_ostream& out, std::pair<T0, T1> args) {
  out << args.first << ", " << args.second;
}

void SemanticsNode::Print(llvm::raw_ostream& out) const {
  out << kind_ << "(";
  switch (kind_) {
#define CARBON_SEMANTICS_NODE_KIND(Name) \
  case SemanticsNodeKind::Name:          \
    PrintArgs(out, GetAs##Name());       \
    break;
#include "toolchain/semantics/semantics_node_kind.def"
  }
  out << ")";
  if (type_.index != -1) {
    out << ": " << type_;
  }
}

}  // namespace Carbon
