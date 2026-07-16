// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/format/whitespace_manager.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "toolchain/lex/token_kind.h"

namespace Carbon::Format {

auto WhitespaceManager::AddToken(int newlines, int spaces, int indent_level,
                                 int nesting_level, Lex::TokenIndex token,
                                 llvm::StringRef rewritten) -> void {
  changes_.push_back({.is_token = true,
                      .newlines = newlines,
                      .spaces = spaces,
                      .token = token,
                      .indent_level = indent_level,
                      .nesting_level = nesting_level,
                      .rewritten = rewritten.str()});
}

auto WhitespaceManager::AddVerbatimGapToken(std::string gap, int indent_level,
                                            int nesting_level,
                                            Lex::TokenIndex token) -> void {
  // The gap text is emitted verbatim; `newlines` still records the line breaks
  // it holds so the alignment pass partitions physical lines correctly.
  int newlines = static_cast<int>(llvm::StringRef(gap).count('\n'));
  changes_.push_back({.is_token = true,
                      .newlines = newlines,
                      .spaces = 0,
                      .token = token,
                      .indent_level = indent_level,
                      .nesting_level = nesting_level,
                      .is_verbatim_gap = true,
                      .verbatim_gap = std::move(gap)});
}

auto WhitespaceManager::AddRaw(int newlines, std::string text) -> void {
  changes_.push_back({.is_token = false,
                      .newlines = newlines,
                      .spaces = 0,
                      .raw = std::move(text)});
}

auto WhitespaceManager::AddTrailingComment(std::string text) -> void {
  // No line break: the comment appends to the current line, separated by a
  // single space (alignment may add more `padding` later).
  changes_.push_back({.is_token = false,
                      .is_trailing_comment = true,
                      .newlines = 0,
                      .spaces = 1,
                      .raw = std::move(text)});
}

auto WhitespaceManager::ComputeStartColumns() -> void {
  int column = 0;
  for (Change& change : changes_) {
    if (!change.is_token) {
      if (change.is_trailing_comment) {
        // A trailing comment continues the current line: its column is the
        // previous content's end plus the separating spaces and any padding.
        column += change.spaces + change.padding;
        change.start_column = column;
        column += static_cast<int>(change.raw.size());
      } else {
        // A full-line comment block ends at a line boundary, so the next token
        // (which necessarily starts a new line) recomputes from scratch.
        column = 0;
      }
      continue;
    }
    if (change.is_verbatim_gap) {
      // A verbatim gap positions its token by its own text: after its last
      // newline if it has one, else appended to the current column.
      llvm::StringRef gap = change.verbatim_gap;
      size_t last_break = gap.rfind('\n');
      if (last_break == llvm::StringRef::npos) {
        column += static_cast<int>(gap.size());
      } else {
        column = static_cast<int>(gap.size() - last_break - 1);
      }
    } else {
      column =
          (change.newlines > 0 ? 0 : column) + change.spaces + change.padding;
    }
    change.start_column = column;
    llvm::StringRef text = tokens_->GetTokenText(change.token);
    // A multi-line token (such as a multi-line string literal) ends on its
    // last physical line, whose width (not the token's full byte length)
    // determines the column after it.
    size_t last_line = text.rfind('\n');
    if (last_line == llvm::StringRef::npos) {
      column += static_cast<int>(text.size());
    } else {
      column = static_cast<int>(text.size() - last_line - 1);
    }
  }
}

auto WhitespaceManager::AlignChanges(
    llvm::function_ref<auto(int, int)->int> find_match) -> void {
  ComputeStartColumns();
  int n = changes_.size();

  // The matched change of each line in the run being built, and the indent
  // those lines share.
  llvm::SmallVector<int> run;
  int run_indent = -1;
  auto finalize = [&] {
    if (run.size() >= 2) {
      int target = 0;
      for (int idx : run) {
        target = std::max(target, changes_[idx].start_column);
      }
      for (int idx : run) {
        changes_[idx].padding += target - changes_[idx].start_column;
      }
    }
    run.clear();
    run_indent = -1;
  };

  int i = 0;
  while (i < n) {
    // The current line spans `[i, j)`: its first change plus any following
    // changes that carry no line break (same-line tokens and a trailing
    // comment).
    int j = i + 1;
    while (j < n && changes_[j].newlines == 0) {
      ++j;
    }
    // The line's indent and bracket nesting come from its first token change;
    // a comment-only line has neither.
    int indent = -1;
    int nesting = -1;
    for (int k = i; k < j; ++k) {
      if (changes_[k].is_token) {
        indent = changes_[k].indent_level;
        nesting = changes_[k].nesting_level;
        break;
      }
    }
    int match = find_match(i, j);

    // A blank line always breaks the run.
    if (changes_[i].newlines > 1) {
      finalize();
    }
    if (match < 0) {
      // A wrapped continuation line (still inside brackets) neither joins nor
      // breaks the run, mirroring clang-format's deeper-nesting skip; any
      // other unmatched line, including a comment line, breaks it.
      if (nesting <= 0) {
        finalize();
      }
    } else {
      if (!run.empty() && indent != run_indent) {
        finalize();
      }
      if (run.empty()) {
        run_indent = indent;
      }
      run.push_back(match);
    }
    i = j;
  }
  finalize();
}

auto WhitespaceManager::AlignTrailingComments() -> void {
  if (!style_.align_trailing_comments) {
    return;
  }
  // The line's trailing comment is its last raw change, if any.
  AlignChanges([&](int begin, int end) {
    for (int k = end - 1; k >= begin; --k) {
      if (changes_[k].is_trailing_comment) {
        return k;
      }
    }
    return -1;
  });
}

auto WhitespaceManager::Generate(llvm::SmallVectorImpl<TokenSpan>& token_map)
    -> std::string {
  AlignTrailingComments();

  std::string output;
  for (const Change& change : changes_) {
    if (change.is_verbatim_gap) {
      // The token's original leading source text, emitted in place of any
      // computed line breaks and indentation.
      output.append(change.verbatim_gap);
    } else {
      output.append(change.newlines, '\n');
    }
    if (!change.is_token) {
      // A trailing comment appends to the current line after its separating
      // spaces (plus any alignment padding); a full-line block is verbatim.
      if (change.is_trailing_comment) {
        output.append(change.spaces + change.padding, ' ');
      }
      output.append(change.raw);
      continue;
    }
    if (!change.is_verbatim_gap) {
      output.append(change.spaces + change.padding, ' ');
    }
    if (!change.rewritten.empty()) {
      // A rewritten token (for example a reformatted C++ snippet) is emitted in
      // place of its source text and is not a `TokenSpan` anchor, so its edit
      // folds into the surrounding gap in `ComputeReplacements`.
      output.append(change.rewritten);
      continue;
    }
    llvm::StringRef text = tokens_->GetTokenText(change.token);
    // A lexer-inserted recovery token's text does not exist in the source (its
    // byte offset is synthesized, and can even overlap a neighboring token),
    // so it cannot anchor a minimal edit. Skipping its span leaves the emitted
    // text inside the surrounding gap, where the gap's edit inserts it.
    if (!tokens_->IsRecoveryToken(change.token)) {
      token_map.push_back({.source_begin = tokens_->GetByteOffset(change.token),
                           .output_begin = static_cast<int32_t>(output.size()),
                           .length = static_cast<int32_t>(text.size())});
    }
    output.append(text.data(), text.size());
  }
  // Content never carries its own trailing newline (line breaks are attributed
  // to the following content), so a non-empty file ends with one final newline.
  if (!changes_.empty()) {
    output.push_back('\n');
  }
  return output;
}

}  // namespace Carbon::Format
