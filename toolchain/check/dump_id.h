// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_DUMP_ID_H_
#define CARBON_TOOLCHAIN_CHECK_DUMP_ID_H_

#include "common/ostream.h"
#include "toolchain/lex/token_kind.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/tree.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

class Context;

auto DumpIdImpl(SemIR::LocId loc_id, const Context& context) -> void;

// A set of DumpId() overloads that dump an object to stderr, useful for calling
// inside a debugger. These are all exposed as part of the `Check::Context` API.
//
// This class is inherited by `Check::Context`, which provides itself as the
// template parameter. The methods are provided here instead of on `Context`
// directly to avoid cluttering the `Context` class with overloads for every
// dumpable id type.
template <class Context>
class DumpIdMethods {
  static_assert(std::same_as<Context, ::Carbon::Check::Context>);

 public:
  LLVM_DUMP_METHOD auto DumpId(Lex::TokenIndex token) const -> void {
    static_cast<const Context&>(*this).tokens().DumpId(token);
  }
  LLVM_DUMP_METHOD auto DumpId(Parse::NodeId node_id) const -> void {
    static_cast<const Context&>(*this).parse_tree().DumpId(node_id);
  }
  LLVM_DUMP_METHOD auto DumpId(SemIR::LocId loc_id) const -> void {
    DumpIdImpl(loc_id, static_cast<const Context&>(*this));
    llvm::errs() << "\n";
  }
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_DUMP_ID_H_
