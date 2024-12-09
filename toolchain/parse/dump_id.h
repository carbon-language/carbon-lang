// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_PARSE_DUMP_ID_H_
#define CARBON_TOOLCHAIN_PARSE_DUMP_ID_H_

#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/node_ids.h"

namespace Carbon::Parse {

class Tree;

auto DumpIdImpl(NodeId node_id, const Tree& tree) -> void;

// A set of DumpId() overloads that dump an object to stderr, useful for calling
// inside a debugger. These are all exposed as part of the `Parse::Tree` API.
//
// This class is inherited by `Parse::Tree`, which provides itself as the
// template parameter. The methods are provided here instead of on `Tree`
// directly to avoid cluttering the `Tree` class with overloads for every
// dumpable id type.
template <class Tree>
class DumpIdMethods {
  static_assert(std::same_as<Tree, ::Carbon::Parse::Tree>);

 public:
  LLVM_DUMP_METHOD auto DumpId(Lex::TokenIndex token) const -> void {
    static_cast<const Tree&>(*this).tokens().DumpId(token);
  }
  LLVM_DUMP_METHOD auto DumpId(NodeId node_id) const -> void {
    DumpIdImpl(node_id, static_cast<const Tree&>(*this));
    Newline();
  }

 private:
  auto Newline() const -> void;
};

}  // namespace Carbon::Parse

#endif  // CARBON_TOOLCHAIN_PARSE_DUMP_ID_H_
