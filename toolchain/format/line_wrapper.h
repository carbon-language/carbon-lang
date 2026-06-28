// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_FORMAT_LINE_WRAPPER_H_
#define CARBON_TOOLCHAIN_FORMAT_LINE_WRAPPER_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/format/style.h"
#include "toolchain/format/token_info.h"
#include "toolchain/lex/token_index.h"
#include "toolchain/lex/tokenized_buffer.h"

namespace Carbon::Format {

// Solves the layout of one unwrapped line, choosing where to insert line breaks
// so as to minimize a penalty under the soft column limit (`style`). This is a
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
                     llvm::ArrayRef<Lex::TokenIndex> line, int indent,
                     const Style& style) -> llvm::SmallVector<int>;

}  // namespace Carbon::Format

#endif  // CARBON_TOOLCHAIN_FORMAT_LINE_WRAPPER_H_
