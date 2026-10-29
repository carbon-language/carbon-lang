// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_FORMAT_CPP_SNIPPET_H_
#define CARBON_TOOLCHAIN_FORMAT_CPP_SNIPPET_H_

#include <optional>
#include <string>

#include "llvm/ADT/StringRef.h"
#include "toolchain/format/style.h"

namespace Carbon::Format {

// If `literal_text` is a multi-line string literal (`'''...'''` or
// `#'''...'''#`) holding C++ code whose formatting changes it, returns the
// rewritten literal with its body reformatted by clang-format and its body
// lines and closing delimiter indented to `indent` columns.
//
// A literal holds C++ when its file type indicator names C++ (such as `'''cpp`
// or `#'''cpp`) or when `force_cpp` is true (used for the body of an
// `inline Cpp` or `import Cpp inline` declaration).
//
// Returns `std::nullopt` when the literal is not an eligible C++ multi-line
// literal, when `style.column_limit - indent < 20`, when formatting would risk
// altering Carbon string escapes or closing delimiters, when clang-format does
// not complete cleanly at a fixed point, or when the formatted literal equals
// `literal_text`.
auto CppSnippet(llvm::StringRef literal_text, int indent, const Style& style,
                bool force_cpp = false) -> std::optional<std::string>;

}  // namespace Carbon::Format

#endif  // CARBON_TOOLCHAIN_FORMAT_CPP_SNIPPET_H_
