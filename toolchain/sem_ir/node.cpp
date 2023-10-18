// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/node.h"

namespace Carbon::SemIR {

auto Node::Print(llvm::raw_ostream& out) const -> void {
  out << "{kind: " << kind_;

  auto print_args = [&](auto... args) {
    int n = 0;
    ((out << ", arg" << n++ << ": " << args), ...);
  };

  // clang warns on unhandled enum values; clang-tidy is incorrect here.
  // NOLINTNEXTLINE(bugprone-switch-missing-default-case)
  switch (kind_) {
#define CARBON_SEM_IR_NODE_KIND(Name)                       \
  case Name::Kind:                                          \
    std::apply(print_args, As<SemIR::Name>().args_tuple()); \
    break;
#include "toolchain/sem_ir/node_kind.def"
  }
  if (type_id_.is_valid()) {
    out << ", type: " << type_id_;
  }
  out << "}";
}

}  // namespace Carbon::SemIR
