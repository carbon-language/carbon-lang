// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_FORMAT_STYLE_H_
#define CARBON_TOOLCHAIN_FORMAT_STYLE_H_

#include <cstdint>

namespace Carbon::Format {

// Formatting parameters for `carbon format`. A default-constructed `Style`
// represents Carbon's canonical formatting style.
struct Style {
  // Soft column limit for formatted output, exceeded only when no legal set of
  // line breaks avoids overrunning it.
  int column_limit = 80;

  // Columns added per block indentation level.
  int indent_width = 2;

  // Columns added past a line's starting indentation for continuation lines
  // when no nearer alignment anchor applies.
  int continuation_indent_width = 4;

  // Maximum number of consecutive blank lines preserved between items.
  int max_empty_lines_to_keep = 1;

  // Penalty charged per column past `column_limit`. Exceeding every split
  // penalty ensures any legal break is preferred over overrunning the limit.
  int64_t penalty_excess_character = 1'000'000;

  // Penalty for breaking immediately after an opening `(`, `[`, or inline `{`
  // before its first element.
  int penalty_break_before_first_call_parameter = 19;

  // Penalty for breaking after `=` onto its right-hand side.
  int penalty_break_assignment = 2;

  // Whether to align consecutive same-indent trailing `//` comments into a
  // shared column. Standalone trailing comments keep a single leading space.
  bool align_trailing_comments = true;

  // Whether to reformat C++ multi-line string literals (those with a C++ file
  // type indicator such as `'''cpp`, or the body of an `inline Cpp` or
  // `import Cpp inline` declaration) with clang-format.
  bool format_cpp_snippets = true;
};

}  // namespace Carbon::Format

#endif  // CARBON_TOOLCHAIN_FORMAT_STYLE_H_
