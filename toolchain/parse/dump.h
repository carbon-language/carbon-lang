// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_PARSE_DUMP_H_
#define CARBON_TOOLCHAIN_PARSE_DUMP_H_

#include "toolchain/lex/dump.h"
#include "toolchain/parse/node_ids.h"

namespace Carbon::Parse {

class Tree;

namespace DumpOverloads {

auto Dump(NodeId node_id, const Tree& tree) -> void;

}  // namespace DumpOverloads

namespace Internal {

template <typename T>
concept HasDumpOverload = requires(const T& t, const Tree& tree) {
  { DumpOverloads::Dump(t, tree) };
};

}

// A set of Dump() overloads that dump an object to stderr, useful for calling
// inside a debugger. These are all exposed as part of the `Parse::Tree` API.
//
// This class is inherited by `Parse::Tree`, which provides itself as the
// template parameter.
template <class Tree>
class DumpMethods {
  static_assert(std::same_as<Tree, ::Carbon::Parse::Tree>);

 public:
#define CARBON_LEX_DUMP_TYPE(Type)                    \
  LLVM_DUMP_METHOD auto Dump(const Type& t) -> void { \
    Dispatch(t, static_cast<const Tree&>(*this));     \
  }
#include "toolchain/lex/dump.def"
#define CARBON_PARSE_DUMP_TYPE(Type)                  \
  LLVM_DUMP_METHOD auto Dump(const Type& t) -> void { \
    Dispatch(t, static_cast<const Tree&>(*this));     \
  }
#include "toolchain/parse/dump.def"

 private:
  template <class T>
    requires(Parse::Internal::HasDumpOverload<T>)
  auto Dispatch(const T& t, const Tree& tree) -> void {
    Parse::DumpOverloads::Dump(t, tree);
  }
  template <class T>
    requires(!Parse::Internal::HasDumpOverload<T>)
  auto Dispatch(const T& t, const Tree& tree) -> void {
    Lex::DumpOverloads::Dump(t, *tree.tokens_);
  }
};

}  // namespace Carbon::Parse

#endif  // CARBON_TOOLCHAIN_PARSE_DUMP_H_
