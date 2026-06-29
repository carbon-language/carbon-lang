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

// If `literal_text` is the source spelling of a multi-line string literal that
// holds C++, returns the literal re-encoded with its body reformatted by
// clang-format, with the body lines and closing `'''` indented to `indent`
// columns.
//
// A literal holds C++ when its file type indicator names C++ (for example
// `'''cpp`), or when `force_cpp` is set -- used for the body of an `inline Cpp`
// (or `import Cpp inline`) declaration, which is C++ regardless of its
// indicator.
//
// clang-format runs under `style`, not its own defaults: in particular the body
// is reformatted to a column limit of `style.column_limit - indent`, so that
// once re-indented by `indent` columns it still respects Carbon's line length.
//
// Returns nullopt when the literal is not C++ to reformat -- a single-line or
// `#`-raw literal, a non-C++ indicator without `force_cpp`, a malformed or
// mis-indented body, or an empty body -- or when clang-format cannot format it.
// The caller then emits the literal unchanged. The returned text replaces the
// literal token verbatim, so its first line is still `'''<indicator>` and it
// carries no surrounding whitespace.
auto CppSnippet(llvm::StringRef literal_text, int indent, const Style& style,
                bool force_cpp = false) -> std::optional<std::string>;

}  // namespace Carbon::Format

#endif  // CARBON_TOOLCHAIN_FORMAT_CPP_SNIPPET_H_
