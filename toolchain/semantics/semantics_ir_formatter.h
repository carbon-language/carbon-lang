// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_FORMATTER_H_
#define CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_FORMATTER_H_

#include "llvm/Support/raw_ostream.h"
#include "toolchain/semantics/semantics_ir.h"

namespace Carbon::SemIR {

auto FormatFile(const TokenizedBuffer& tokenized_buffer,
                const ParseTree& parse_tree, const File& semantics_ir,
                llvm::raw_ostream& out) -> void;

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_FORMATTER_H_
