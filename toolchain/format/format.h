// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_FORMAT_FORMAT_H_
#define CARBON_TOOLCHAIN_FORMAT_FORMAT_H_

#include "common/ostream.h"
#include "toolchain/parse/tree.h"

namespace Carbon::Format {

// Formats file content to the out stream, driven by the parse tree (and the
// tokens it references). Returns false if the input had lex or parse errors;
// best-effort formatted output is still produced, and the caller decides
// whether to use it (the driver does, and reflects errors in its exit code).
auto Format(const Parse::Tree& tree, llvm::raw_ostream& out) -> bool;

}  // namespace Carbon::Format

#endif  // CARBON_TOOLCHAIN_FORMAT_FORMAT_H_
