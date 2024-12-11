// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_PARSE_DUMP_ID_H_
#define CARBON_TOOLCHAIN_PARSE_DUMP_ID_H_

#ifndef NDEBUG

#include "toolchain/parse/tree.h"

namespace Carbon::Parse {

auto DumpIdImpl(const Tree& tree, NodeId node_id) -> void;

auto DumpId(const Tree& tree, Lex::TokenIndex token) -> void;

auto DumpId(const Tree& tree, NodeId node_id) -> void;

}  // namespace Carbon::Parse

#endif  // NDEBUG

#endif  // CARBON_TOOLCHAIN_PARSE_DUMP_ID_H_
