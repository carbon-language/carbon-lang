// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/format/cpp_snippet.h"

#include <optional>
#include <string>

#include "clang/Format/Format.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "toolchain/format/style.h"

namespace Carbon::Format {

static constexpr llvm::StringRef MultiLineDelimiter = "'''";
static constexpr int MinCppSnippetColumns = 20;

// Returns whether `indicator` (the file type indicator following the opening
// delimiter) names C++, matching common C++ source and header extensions
// case-insensitively.
static auto IsCppIndicator(llvm::StringRef indicator) -> bool {
  for (llvm::StringRef cpp :
       {"cpp", "cc", "cxx", "c++", "h", "hpp", "hxx", "hh"}) {
    if (indicator.equals_insensitive(cpp)) {
      return true;
    }
  }
  return false;
}

// Returns `introducer` (a multi-line literal's introducer line after the
// opening delimiter, with any trailing `\r` already removed) with any trailing
// `//` comment stripped. Mirrors `Lex::StringLiteral::Introducer::Lex`: a `//`
// followed by horizontal whitespace or end-of-line starts a comment.
static auto StripIntroducerComment(llvm::StringRef introducer)
    -> llvm::StringRef {
  for (size_t slashes = introducer.find("//"); slashes != llvm::StringRef::npos;
       slashes = introducer.find("//", slashes + 1)) {
    llvm::StringRef after_slashes = introducer.drop_front(slashes + 2);
    if (after_slashes.empty() || after_slashes.starts_with(' ') ||
        after_slashes.starts_with('\t')) {
      return introducer.take_front(slashes);
    }
  }
  return introducer;
}

// Runs clang-format once over `code` under `clang_style`, returning
// `std::nullopt` if formatting is incomplete or replacements fail to apply.
static auto RunClangFormat(llvm::StringRef code,
                           const clang::format::FormatStyle& clang_style)
    -> std::optional<std::string> {
  llvm::SmallVector<clang::tooling::Range> ranges = {
      clang::tooling::Range(0, code.size())};
  clang::format::FormattingAttemptStatus status;
  clang::tooling::Replacements replacements = clang::format::reformat(
      clang_style, code, ranges, "snippet.cpp", &status);
  if (!status.FormatComplete) {
    return std::nullopt;
  }
  llvm::Expected<std::string> formatted =
      clang::tooling::applyAllReplacements(code, replacements);
  if (!formatted) {
    llvm::consumeError(formatted.takeError());
    return std::nullopt;
  }
  return std::move(*formatted);
}

// Reformats de-indented C++ `snippet` using clang-format configured to match
// Carbon's C++ conventions and remaining line budget (`style.column_limit -
// indent`). Token-reordering and comment-injecting passes are disabled so only
// whitespace changes, and output that introduces a closing delimiter or Carbon
// escape sequence or fails to reach a fixed point is rejected.
static auto ReformatCpp(llvm::StringRef snippet, const Style& style, int indent,
                        llvm::StringRef close_delimiter,
                        llvm::StringRef escape_prefix)
    -> std::optional<std::string> {
  if (style.column_limit - indent < MinCppSnippetColumns) {
    return std::nullopt;
  }

  clang::format::FormatStyle clang_style = clang::format::getLLVMStyle();
  clang_style.ColumnLimit = style.column_limit - indent;
  clang_style.IndentWidth = style.indent_width;
  clang_style.ContinuationIndentWidth = style.continuation_indent_width;
  clang_style.UseTab = clang::format::FormatStyle::UT_Never;
  clang_style.PointerAlignment = clang::format::FormatStyle::PAS_Left;
  clang_style.DerivePointerAlignment = false;
  clang_style.SortIncludes.Enabled = false;
  clang_style.SortUsingDeclarations = clang::format::FormatStyle::SUD_Never;
  clang_style.FixNamespaceComments = false;

  std::optional<std::string> formatted = RunClangFormat(snippet, clang_style);
  if (!formatted) {
    return std::nullopt;
  }
  if (*formatted == snippet) {
    return formatted;
  }
  if (llvm::StringRef(*formatted).contains(close_delimiter) ||
      llvm::StringRef(*formatted).contains(escape_prefix)) {
    return std::nullopt;
  }
  std::optional<std::string> reformatted =
      RunClangFormat(*formatted, clang_style);
  if (!reformatted || *reformatted != *formatted) {
    return std::nullopt;
  }
  return formatted;
}

auto CppSnippet(llvm::StringRef literal_text, int indent, const Style& style,
                bool force_cpp) -> std::optional<std::string> {
  if (indent < 0 || style.column_limit - indent < MinCppSnippetColumns) {
    return std::nullopt;
  }

  size_t hash_count = 0;
  while (hash_count < literal_text.size() && literal_text[hash_count] == '#') {
    ++hash_count;
  }
  llvm::SmallString<16> open_delimiter;
  open_delimiter.append(hash_count, '#');
  open_delimiter.append(MultiLineDelimiter);
  llvm::SmallString<16> close_delimiter(MultiLineDelimiter);
  close_delimiter.append(hash_count, '#');
  llvm::SmallString<16> escape_prefix("\\");
  escape_prefix.append(hash_count, '#');

  if (literal_text.size() <
          open_delimiter.size() + close_delimiter.size() + 1 ||
      !literal_text.starts_with(open_delimiter) ||
      !literal_text.ends_with(close_delimiter)) {
    return std::nullopt;
  }

  llvm::StringRef after_open = literal_text.drop_front(open_delimiter.size());
  size_t body_start = after_open.find('\n');
  if (body_start == llvm::StringRef::npos) {
    return std::nullopt;
  }
  llvm::StringRef introducer = after_open.take_front(body_start);
  introducer.consume_back("\r");
  for (char c : introducer) {
    if (static_cast<unsigned char>(c) < ' ' && c != '\t') {
      return std::nullopt;
    }
  }
  llvm::StringRef indicator = StripIntroducerComment(introducer).trim(" \t");
  if (indicator.find_first_of("'#\"") != llvm::StringRef::npos) {
    return std::nullopt;
  }
  if (!force_cpp && !IsCppIndicator(indicator)) {
    return std::nullopt;
  }

  llvm::StringRef rest = after_open.drop_front(body_start + 1);
  size_t closing_start = rest.rfind('\n');
  llvm::StringRef closing_line = closing_start == llvm::StringRef::npos
                                     ? rest
                                     : rest.drop_front(closing_start + 1);
  llvm::StringRef body = closing_start == llvm::StringRef::npos
                             ? ""
                             : rest.take_front(closing_start);
  llvm::StringRef base_indent = closing_line.drop_back(close_delimiter.size());
  if (base_indent.find_first_not_of(" \t") != llvm::StringRef::npos) {
    return std::nullopt;
  }

  std::string snippet;
  llvm::SmallVector<llvm::StringRef> lines;
  body.split(lines, '\n');
  for (llvm::StringRef line : lines) {
    line.consume_back("\r");
    if (line.starts_with(base_indent)) {
      llvm::StringRef stripped = line.drop_front(base_indent.size());
      snippet.append(stripped.data(), stripped.size());
    } else if (!line.trim(" \t").empty()) {
      return std::nullopt;
    }
    snippet.push_back('\n');
  }
  if (llvm::StringRef(snippet).trim().empty()) {
    return std::nullopt;
  }
  // Reject bodies containing the literal's closing delimiter or an active
  // Carbon escape prefix (`\` in `'''...'''`, `\#` in `#'''...'''#`, etc.) so
  // clang-format cannot alter Carbon escape semantics or split escapes across
  // lines. Raw `#'''` literals allow plain C++ `\` escapes and macro splices.
  if (llvm::StringRef(snippet).contains(close_delimiter) ||
      llvm::StringRef(snippet).contains(escape_prefix)) {
    return std::nullopt;
  }
  for (char c : snippet) {
    // Reject control characters (including literal tabs, which are invalid in
    // Carbon string literal content unless escaped).
    if (static_cast<unsigned char>(c) < ' ' && c != '\n') {
      return std::nullopt;
    }
  }

  std::optional<std::string> formatted =
      ReformatCpp(snippet, style, indent, close_delimiter, escape_prefix);
  if (!formatted) {
    return std::nullopt;
  }

  std::string result;
  result.append(open_delimiter.begin(), open_delimiter.end());
  result.append(introducer.data(), introducer.size());
  result.push_back('\n');
  llvm::SmallVector<llvm::StringRef> formatted_lines;
  llvm::StringRef(*formatted).rtrim('\n').split(formatted_lines, '\n');
  for (llvm::StringRef line : formatted_lines) {
    if (!line.empty()) {
      result.append(indent, ' ');
      result.append(line.data(), line.size());
    }
    result.push_back('\n');
  }
  result.append(indent, ' ');
  result.append(close_delimiter.begin(), close_delimiter.end());
  if (result == literal_text) {
    return std::nullopt;
  }
  return result;
}

}  // namespace Carbon::Format
