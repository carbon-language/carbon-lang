// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_FORMAT_STYLE_H_
#define CARBON_TOOLCHAIN_FORMAT_STYLE_H_

namespace Carbon::Format {

// Formatting parameters for `carbon format`. A default-constructed `Style`
// represents Carbon's canonical formatting style.
struct Style {
  // Columns added per block indentation level.
  int indent_width = 2;

  // Columns added past a line's starting indentation for continuation lines
  // when no nearer alignment anchor applies.
  int continuation_indent_width = 4;

  // Maximum number of consecutive blank lines preserved between items.
  int max_empty_lines_to_keep = 1;
};

}  // namespace Carbon::Format

#endif  // CARBON_TOOLCHAIN_FORMAT_STYLE_H_
