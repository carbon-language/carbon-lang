// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_FORMAT_LINE_WRAPPER_H_
#define CARBON_TOOLCHAIN_FORMAT_LINE_WRAPPER_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/format/style.h"
#include "toolchain/format/token_info.h"
#include "toolchain/lex/token_index.h"
#include "toolchain/lex/tokenized_buffer.h"

namespace Carbon::Format {

// Per-token constraints from comments embedded inside an unwrapped line.
// When provided to `SolveLineBreaks`, both vectors have size `line.size()`.
struct LineBreakConstraints {
  // Indices before which a line break is mandatory because a trailing or
  // full-line comment precedes `line[i]`.
  llvm::BitVector must_break_before;
  // Extra columns (`1 + comment_width` when present, else `0`) occupied by a
  // trailing comment attached after `line[i]` on the same physical line.
  llvm::SmallVector<int> trailing_comment_widths;
};

// Chooses line breaks for a single unwrapped line `line` starting at column
// `indent` under `style`, minimizing total split and excess-column penalty via
// uniform-cost search (falling back to greedy first-fit packing if the state
// cap is hit).
//
// Returns one entry per token in `line`: the column to indent to if a line
// break precedes that token, or -1 if it stays on the previous token's line.
// `result[0]` is always -1.
auto SolveLineBreaks(const Lex::TokenizedBuffer& tokens,
                     const TokenInfoStore& token_infos,
                     llvm::ArrayRef<Lex::TokenIndex> line, int indent,
                     const Style& style,
                     const LineBreakConstraints* constraints = nullptr)
    -> llvm::SmallVector<int>;

}  // namespace Carbon::Format

#endif  // CARBON_TOOLCHAIN_FORMAT_LINE_WRAPPER_H_
