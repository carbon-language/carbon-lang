// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_node.h"

namespace Carbon::SemIR {

static auto PrintArgs(llvm::raw_ostream& /*out*/,
                      const Node::NoArgs /*no_args*/) -> void {}

template <typename T>
static auto PrintArgs(llvm::raw_ostream& out, T arg) -> void {
  out << ", arg0: " << arg;
}

template <typename T0, typename T1>
static auto PrintArgs(llvm::raw_ostream& out, std::pair<T0, T1> args) -> void {
  PrintArgs(out, args.first);
  out << ", arg1: " << args.second;
}

auto operator<<(llvm::raw_ostream& out, const Node& node)
    -> llvm::raw_ostream& {
  out << "{kind: " << node.kind_;
  // clang warns on unhandled enum values; clang-tidy is incorrect here.
  // NOLINTNEXTLINE(bugprone-switch-missing-default-case)
  switch (node.kind_) {
#define CARBON_SEMANTICS_NODE_KIND(Name) \
  case NodeKind::Name:                   \
    PrintArgs(out, node.GetAs##Name());  \
    break;
#include "toolchain/semantics/semantics_node_kind.def"
  }
  if (node.type_id_.is_valid()) {
    out << ", type: " << node.type_id_;
  }
  out << "}";
  return out;
}

}  // namespace Carbon::SemIR
