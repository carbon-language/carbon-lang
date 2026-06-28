// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_FORMAT_LINE_WRAPPER_H_
#define CARBON_TOOLCHAIN_FORMAT_LINE_WRAPPER_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/format/token_info.h"
#include "toolchain/lex/token_index.h"
#include "toolchain/lex/tokenized_buffer.h"

namespace Carbon::Format {

// The column lines are laid out to fit within where possible. This is a soft
// limit: the solver overruns it only when no legal set of breaks avoids it (see
// `SolveLineBreaks`). LLVM-style default; a future configurable style object
// will replace this constant. See toolchain/docs/format.md.
inline constexpr int ColumnLimit = 80;

// The number of columns a continuation line is indented past its statement's
// own indentation, when no nearer alignment anchor (such as an open bracket)
// applies. LLVM-style default.
inline constexpr int ContinuationIndentWidth = 4;

// Solves the layout of one unwrapped line, choosing where to insert line breaks
// so as to minimize a penalty under the soft column limit. This is a
// uniform-cost (Dijkstra) shortest-path search over layout states, mirroring
// clang-format's `ContinuationIndenter`; see toolchain/docs/format.md.
//
// `line` is the line's tokens (with kinds in `tokens` and the formatter's
// annotations in `token_infos`) and `indent` is the column its first token
// starts at. Returns one entry per token: the column to indent to if a line
// break should precede that token, or -1 if no break precedes it. The first
// token's entry is always -1 (the caller positions it at `indent`).
auto SolveLineBreaks(const Lex::TokenizedBuffer& tokens,
                     const TokenInfoStore& token_infos,
                     llvm::ArrayRef<Lex::TokenIndex> line, int indent)
    -> llvm::SmallVector<int>;

}  // namespace Carbon::Format

#endif  // CARBON_TOOLCHAIN_FORMAT_LINE_WRAPPER_H_
