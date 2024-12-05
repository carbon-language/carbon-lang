// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_DUMP_H_
#define CARBON_TOOLCHAIN_CHECK_DUMP_H_

#include "toolchain/lex/dump.h"
#include "toolchain/lex/token_kind.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/dump.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

class Context;

namespace DumpOverloads {

auto Dump(SemIR::LocId loc_id, const Context& context) -> void;

// Overrides Lex::DumpOverloads::Dump to provide additional context in the
// check stage.
auto Dump(Lex::TokenKind token_kind, const Context& context) -> void;

}  // namespace DumpOverloads

namespace Internal {

template <typename T>
concept HasDumpOverload = requires(const T& t, const Context& context) {
  { DumpOverloads::Dump(t, context) };
};

}

// A set of Dump() overloads that dump an object to stderr, useful for calling
// inside a debugger. These are all exposed as part of the `Check::Context`
// API.
//
// This class is inherited by `Check::Context`, which provides itself as the
// template parameter.
template <class Context>
class DumpMethods {
  static_assert(std::same_as<Context, ::Carbon::Check::Context>);

 public:
#define CARBON_LEX_DUMP_TYPE(Type)                    \
  LLVM_DUMP_METHOD auto Dump(const Type& t) -> void { \
    Dispatch(t, static_cast<const Context&>(*this));  \
  }
#include "toolchain/lex/dump.def"
#define CARBON_PARSE_DUMP_TYPE(Type)                  \
  LLVM_DUMP_METHOD auto Dump(const Type& t) -> void { \
    Dispatch(t, static_cast<const Context&>(*this));  \
  }
#include "toolchain/parse/dump.def"
#define CARBON_CHECK_DUMP_TYPE(Type)                  \
  LLVM_DUMP_METHOD auto Dump(const Type& t) -> void { \
    Dispatch(t, static_cast<const Context&>(*this));  \
  }
#include "toolchain/check/dump.def"

 private:
  template <class T>
  auto Dispatch(const T& t, const Context& context) -> void {
    if constexpr (Check::Internal::HasDumpOverload<T>) {
      Check::DumpOverloads::Dump(t, context);
    } else if constexpr (Parse::Internal::HasDumpOverload<T>) {
      Parse::DumpOverloads::Dump(t, context.parse_tree());
    } else {
      Lex::DumpOverloads::Dump(t, context.tokens());
    }
  }
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_DUMP_H_
