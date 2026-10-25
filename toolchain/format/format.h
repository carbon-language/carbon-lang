// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_FORMAT_FORMAT_H_
#define CARBON_TOOLCHAIN_FORMAT_FORMAT_H_

#include "common/ostream.h"
#include "toolchain/format/style.h"
#include "toolchain/parse/tree.h"

namespace Carbon::Format {

// Formats the file represented by `tree` to `out` according to `style`.
// Produces best-effort output even when lex or parse errors exist, and returns
// true if the input had no errors.
auto Format(const Parse::Tree& tree, llvm::raw_ostream& out,
            const Style& style = Style()) -> bool;

}  // namespace Carbon::Format

#endif  // CARBON_TOOLCHAIN_FORMAT_FORMAT_H_
