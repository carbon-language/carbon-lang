// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_DUMPER_H_
#define CARBON_TOOLCHAIN_CHECK_DUMPER_H_

#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/tree.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

class Dumper {
 public:
  static auto Dump(const Lex::TokenizedBuffer& tokens,
                   const Parse::Tree& parse_tree, SemIR::LocId loc_id) -> void;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_DUMPER_H_
