// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef COMMON_STRING_HELPERS_H_
#define COMMON_STRING_HELPERS_H_

#include <optional>
#include <string>

#include "llvm/ADT/StringRef.h"

namespace Carbon {

// Note llvm StringExtras has significant functionality which is intended to be
// complementary to this.

// Unescapes Carbon escape sequences in the source string. Returns std::nullopt
// on bad input.
auto UnescapeStringLiteral(llvm::StringRef source)
    -> std::optional<std::string>;

}  // namespace Carbon

#endif  // COMMON_STRING_HELPERS_H_
