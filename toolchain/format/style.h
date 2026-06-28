// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_FORMAT_STYLE_H_
#define CARBON_TOOLCHAIN_FORMAT_STYLE_H_

#include <cstdint>

namespace Carbon::Format {

// The tunable knobs of the formatter, gathered into one object so the canonical
// style is a single source of truth and a future configuration surface has
// somewhere to write. A default-constructed `Style` is Carbon's canonical
// style; its values follow clang-format's LLVM style except where noted in
// toolchain/docs/format.md. Knobs are added here as features come to need them
// rather than mirroring clang-format's whole `FormatStyle` up front.
struct Style {
  // The column lines are laid out to fit within where possible. This is a soft
  // limit, exceeded only when no legal set of breaks avoids it. LLVM.
  int column_limit = 80;

  // The number of columns each brace-nesting level indents by. LLVM.
  int indent_width = 2;

  // The number of columns a continuation line indents past its statement's own
  // indentation, when no nearer alignment anchor (such as an open bracket)
  // applies. LLVM.
  int continuation_indent_width = 4;

  // The greatest number of consecutive blank lines kept between content. LLVM.
  int max_empty_lines_to_keep = 1;

  // The penalty for each column a line runs past `column_limit`. It dwarfs
  // every split penalty, which is what makes the limit soft. LLVM
  // (`PenaltyExcessCharacter`).
  int64_t penalty_excess_character = 1'000'000;

  // The penalty for breaking before the first element of a bracketed list
  // (rather than packing it onto the opening line). LLVM
  // (`PenaltyBreakBeforeFirstCallParameter`).
  int penalty_break_before_first_call_parameter = 19;

  // The penalty for breaking after `=` onto the right-hand side. LLVM
  // (`PenaltyBreakAssignment`).
  int penalty_break_assignment = 2;
};

}  // namespace Carbon::Format

#endif  // CARBON_TOOLCHAIN_FORMAT_STYLE_H_
