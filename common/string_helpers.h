// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef COMMON_STRING_HELPERS_H_
#define COMMON_STRING_HELPERS_H_

#include <optional>
#include <string>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace Carbon {

// Note llvm StringExtras has significant functionality which is intended to be
// complementary to this.

// Unescapes Carbon escape sequences in the source string. Returns std::nullopt
// on bad input. `is_block_string` enables escaping unique to block string
// literals, such as \<newline>.
auto UnescapeStringLiteral(llvm::StringRef source, bool is_block_string = false)
    -> std::optional<std::string>;

// Parses a block string literal in `source`.
auto ParseBlockStringLiteral(llvm::StringRef source)
    -> llvm::Expected<std::string>;

}  // namespace Carbon

#endif  // COMMON_STRING_HELPERS_H_
