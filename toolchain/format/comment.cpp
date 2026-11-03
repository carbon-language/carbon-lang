// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/format/comment.h"

#include <string>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

namespace Carbon::Format {

// Wraps one over-long comment line onto multiple lines at `indentation`,
// appending them to `lines`. `content` is the line's text after the `//`
// marker; its leading whitespace joins the `//` to form the prefix repeated on
// every produced line, and the rest is split at whitespace runs so each line
// fits `column_limit`. The retained text is kept verbatim, so interior spacing
// survives. A single word too long to fit is left on its own over-long line
// rather than broken.
static auto WrapCommentBody(llvm::StringRef content,
                            llvm::StringRef indentation, int column_limit,
                            llvm::SmallVectorImpl<std::string>& lines) -> void {
  size_t body_start = content.find_first_not_of(" \t");
  if (body_start == llvm::StringRef::npos) {
    body_start = content.size();
  }
  std::string prefix =
      (llvm::Twine(indentation) + "//" + content.take_front(body_start)).str();
  llvm::StringRef body = content.drop_front(body_start);
  // The columns available to the body text on each line, at least one so a
  // pathological limit still makes progress word by word.
  int available = std::max(1, column_limit - static_cast<int>(prefix.size()));
  while (static_cast<int>(body.size()) > available) {
    // Split at the last whitespace run whose preceding text fits. If even the
    // first word does not fit, keep it whole on an over-long line and split
    // after it.
    // `find_last_of`'s bound is exclusive, so search one past `available` to
    // allow a split exactly at the last fitting column.
    size_t split = body.find_last_of(" \t", available + 1);
    while (split != llvm::StringRef::npos && split > 0 &&
           (body[split - 1] == ' ' || body[split - 1] == '\t')) {
      --split;
    }
    if (split == llvm::StringRef::npos || split == 0) {
      split = body.find_first_of(" \t", available + 1);
      if (split == llvm::StringRef::npos) {
        break;
      }
      while (split > 0 && (body[split - 1] == ' ' || body[split - 1] == '\t')) {
        --split;
      }
    }
    lines.push_back(prefix + body.take_front(split).str());
    body = body.drop_front(split).ltrim(" \t");
  }
  lines.push_back(prefix + body.str());
}

auto CommentText(llvm::StringRef comment_text, int indent, int column_limit)
    -> std::string {
  llvm::SmallVector<std::string> lines;
  std::string indentation(indent, ' ');

  llvm::SmallVector<llvm::StringRef> raw_lines;
  comment_text.split(raw_lines, '\n');
  for (llvm::StringRef raw : raw_lines) {
    // Drop the original indentation (and any trailing whitespace); each comment
    // line starts at `//`. The trailing empty piece after the block's final
    // newline trims to empty and is skipped.
    llvm::StringRef line = raw.trim();
    if (line.empty()) {
      continue;
    }
    if (indent + static_cast<int>(line.size()) <= column_limit) {
      // Fits as-is: re-indent and keep the content verbatim.
      lines.push_back(indentation + line.str());
      continue;
    }
    llvm::StringRef content = line.drop_front(2);
    if (!content.empty() && content.front() != ' ' && content.front() != '\t') {
      // Not a wrappable `// ` comment body: a `//===`-style divider or a
      // lexically invalid comment kept best-effort. Re-indent it verbatim,
      // over-long, rather than mangle it by word-wrapping.
      lines.push_back(indentation + line.str());
      continue;
    }
    WrapCommentBody(content, indentation, column_limit, lines);
  }
  return llvm::join(lines, "\n");
}

}  // namespace Carbon::Format
