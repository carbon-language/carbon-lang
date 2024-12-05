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

// Dump implementation methods. Each one takes `T` or `const T&` for the type it
// will dump along with a `const Tree&`. It should dump to `llvm::errs()`.
auto Dump(NodeId node_id, const Tree& tree) -> void;

}  // namespace DumpOverloads

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
    Lex::DumpOverloads::Dump(t, *static_cast<const Tree&>(*this).tokens_);     \
  }
#include "toolchain/lex/dump.def"
#define CARBON_PARSE_DUMP_TYPE(Type)                  \
  LLVM_DUMP_METHOD auto Dump(const Type& t) -> void { \
    Parse::DumpOverloads::Dump(t, static_cast<const Tree&>(*this));     \
  }
#include "toolchain/parse/dump.def"
};

}  // namespace Carbon::Parse

#endif  // CARBON_TOOLCHAIN_PARSE_DUMP_H_
