// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_FORMAT_COMMENT_H_
#define CARBON_TOOLCHAIN_FORMAT_COMMENT_H_

#include <string>

#include "llvm/ADT/StringRef.h"

namespace Carbon::Format {

// Formats one `//` comment block for output at `indent` columns within
// `column_limit`. `comment_text` is a comment as returned by
// `Lex::TokenizedBuffer::GetCommentText`: a run of consecutive comment lines
// coalesced into one block, so it may span several physical lines (the first
// without leading indentation, the rest with their original indentation), and
// it may end in a newline.
//
// Each physical line is re-indented to `indent` (the lexer keeps each comment
// line's original indentation, which need not match the surrounding code) and,
// if it would still exceed `column_limit`, wrapped at whitespace onto further
// `//` lines at the same indent. A single word too long to fit is left on its
// own over-long line rather than broken. Consecutive lines are never merged,
// and a line that already fits keeps its content verbatim.
//
// Returns the formatted comment text: fully-indented lines joined by newlines,
// with trailing whitespace stripped and no trailing newline.
auto CommentText(llvm::StringRef comment_text, int indent, int column_limit)
    -> std::string;

}  // namespace Carbon::Format

#endif  // CARBON_TOOLCHAIN_FORMAT_COMMENT_H_
