// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_FORMAT_FORMAT_H_
#define CARBON_TOOLCHAIN_FORMAT_FORMAT_H_

#include "common/ostream.h"
#include "toolchain/lex/tokenized_buffer.h"

namespace Carbon::Format {

// Formats file content (based on tokens) to the out stream. Returns false if
// there was an error during formatting, and the formatted stream shouldn't be
// used (in that case, the caller might want to use the original content).
auto Format(const Lex::TokenizedBuffer& tokens, llvm::raw_ostream& out) -> bool;

}  // namespace Carbon::Format

#endif  // CARBON_TOOLCHAIN_FORMAT_FORMAT_H_
