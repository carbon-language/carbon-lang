// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef NDEBUG

#include "toolchain/parse/dump.h"

#include "common/raw_string_ostream.h"
#include "toolchain/lex/dump.h"

namespace Carbon::Parse {

LLVM_DUMP_METHOD auto Dump(const Tree& tree, Lex::TokenIndex token)
    -> std::string {
  return Lex::Dump(tree.tokens(), token);
}

LLVM_DUMP_METHOD auto Dump(const Tree& tree, NodeId node_id) -> std::string {
  RawStringOstream out;
  if (!node_id.has_value()) {
    out << "NodeId(<none>)";
    return out.TakeStr();
  }

  auto kind = tree.node_kind(node_id);
  auto token = tree.node_token(node_id);

  out << "NodeId(kind: " << kind
      << ", token: " << Lex::Dump(tree.tokens(), token) << ")";
  return out.TakeStr();
}

}  // namespace Carbon::Parse

#endif  // NDEBUG
