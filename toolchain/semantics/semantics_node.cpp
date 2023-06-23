// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_node.h"

#include "toolchain/semantics/semantics_builtin_kind.h"

namespace Carbon {

static auto PrintArgs(llvm::raw_ostream& /*out*/,
                      const SemanticsNode::NoArgs /*no_args*/) -> void {}

template <typename T>
static auto PrintArgs(llvm::raw_ostream& out, T arg) -> void {
  out << ", arg0: " << arg;
}

template <typename T0, typename T1>
static auto PrintArgs(llvm::raw_ostream& out, std::pair<T0, T1> args) -> void {
  PrintArgs(out, args.first);
  out << ", arg1: " << args.second;
}

void SemanticsNode::Print(llvm::raw_ostream& out) const {
  out << "{kind: " << kind_;
  switch (kind_) {
#define CARBON_SEMANTICS_NODE_KIND(Name) \
  case SemanticsNodeKind::Name:          \
    PrintArgs(out, GetAs##Name());       \
    break;
#include "toolchain/semantics/semantics_node_kind.def"
  }
  if (type_id_.is_valid()) {
    out << ", type: " << type_id_;
  }
  out << "}";
}

}  // namespace Carbon
