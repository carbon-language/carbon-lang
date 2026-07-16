// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/format/cpp_snippet.h"

#include <algorithm>
#include <optional>
#include <string>

#include "clang/Format/Format.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "toolchain/format/style.h"

namespace Carbon::Format {

// The `'''` that opens and closes a multi-line string literal.
static constexpr llvm::StringRef MultiLineDelimiter = "'''";

// Whether `indicator` (the file type indicator following the opening `'''`)
// names C++, matching common C++ source and header extensions
// case-insensitively.
//
// TODO: This fixed set could become a `Style` knob if other embedded
// languages (or custom indicators) ever need formatting.
static auto IsCppIndicator(llvm::StringRef indicator) -> bool {
  for (llvm::StringRef cpp :
       {"cpp", "cc", "cxx", "c++", "h", "hpp", "hxx", "hh"}) {
    if (indicator.equals_insensitive(cpp)) {
      return true;
    }
  }
  return false;
}

// Returns `introducer` (a multi-line literal's introducer line, after the
// opening `'''`) with any trailing comment removed: a `//` followed by
// whitespace or the end of the line begins one, and ends the file type
// indicator the way trailing whitespace does. Mirrors the lexer's
// `StringLiteral::Introducer::Lex`.
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

// Runs clang-format once over `code` under `clang_style`, returning the
// formatted text or nullopt if it cannot be applied.
static auto RunClangFormat(llvm::StringRef code,
                           const clang::format::FormatStyle& clang_style)
    -> std::optional<std::string> {
  llvm::SmallVector<clang::tooling::Range> ranges = {
      clang::tooling::Range(0, code.size())};
  clang::tooling::Replacements replacements =
      clang::format::reformat(clang_style, code, ranges, "snippet.cpp");
  llvm::Expected<std::string> formatted =
      clang::tooling::applyAllReplacements(code, replacements);
  if (!formatted) {
    llvm::consumeError(formatted.takeError());
    return std::nullopt;
  }
  return std::move(*formatted);
}

// Reformats `snippet` (raw, de-indented C++) with clang-format's LLVM style,
// overridden by Carbon's `style`, returning the formatted text or nullopt if it
// cannot be formatted. The snippet is reformatted to a column limit of
// `style.column_limit - indent`, because the result is re-indented by `indent`
// columns when placed back in the literal and must still fit Carbon's limit.
static auto ReformatCpp(llvm::StringRef snippet, const Style& style, int indent)
    -> std::optional<std::string> {
  clang::format::FormatStyle clang_style = clang::format::getLLVMStyle();
  // Carbon's style governs, overriding clang-format's inherited LLVM defaults.
  // A column limit of at least 1 is required, as clang-format treats 0 as "no
  // limit" rather than a hard wrap.
  clang_style.ColumnLimit = std::max(1, style.column_limit - indent);
  clang_style.IndentWidth = style.indent_width;
  clang_style.ContinuationIndentWidth = style.continuation_indent_width;
  clang_style.UseTab = clang::format::FormatStyle::UT_Never;
  // Pointer and reference declarators bind to the type (`T* p`), matching
  // Carbon's own C++ code style rather than the LLVM default. The alignment is
  // pinned, never derived from the snippet's existing spelling, so the output
  // does not depend on the input's habits.
  clang_style.PointerAlignment = clang::format::FormatStyle::PAS_Left;
  clang_style.DerivePointerAlignment = false;

  std::optional<std::string> formatted = RunClangFormat(snippet, clang_style);
  if (!formatted) {
    return std::nullopt;
  }
  if (*formatted == snippet) {
    // Already formatted -- a fixed point by definition, since reformatting is
    // deterministic. This is the common case, and skips the verification run.
    return formatted;
  }
  // A `'''` can form in the output even though the input was screened -- for
  // example by joining a line ending in `''` with one starting in `'` -- and
  // would re-form a closing delimiter once re-indented. Leave such a snippet
  // alone, as with the input-side guard.
  if (llvm::StringRef(*formatted).contains(MultiLineDelimiter)) {
    return std::nullopt;
  }
  // Only rewrite a snippet whose formatting is a fixed point. clang-format is
  // not idempotent on every input (notably malformed code, which the embedded
  // body may well be), and rewriting to a non-fixed-point would make the
  // surrounding formatter non-idempotent. When it isn't stable, leave the
  // literal unchanged.
  std::optional<std::string> reformatted =
      RunClangFormat(*formatted, clang_style);
  if (!reformatted || *reformatted != *formatted) {
    return std::nullopt;
  }
  return formatted;
}

auto CppSnippet(llvm::StringRef literal_text, int indent, const Style& style,
                bool force_cpp) -> std::optional<std::string> {
  // Only a plain `'''`-delimited multi-line literal can be a C++ snippet; a
  // single-line literal, a `#`-raw literal, or the malformed `"""` form is left
  // alone.
  if (!literal_text.starts_with(MultiLineDelimiter) ||
      !literal_text.ends_with(MultiLineDelimiter)) {
    return std::nullopt;
  }

  // The introducer line runs from after the opening `'''` to the first
  // newline; the body begins after that newline. The whole line is preserved
  // verbatim on reassembly.
  llvm::StringRef after_open =
      literal_text.drop_front(MultiLineDelimiter.size());
  size_t body_start = after_open.find('\n');
  if (body_start == llvm::StringRef::npos) {
    return std::nullopt;
  }
  llvm::StringRef introducer = after_open.take_front(body_start);
  // The body is C++ when forced (an `inline Cpp` literal) or when the file type
  // indicator -- the introducer line minus any trailing comment -- names C++ (a
  // `'''cpp` literal).
  llvm::StringRef indicator = StripIntroducerComment(introducer);
  if (!force_cpp && !IsCppIndicator(indicator.trim())) {
    return std::nullopt;
  }

  // The closing `'''` sits on its own final line, whose leading whitespace is
  // the indentation stripped from every body line (Carbon's multi-line string
  // de-indent rule). Split that line off from the body.
  llvm::StringRef rest = after_open.drop_front(body_start + 1);
  size_t closing_start = rest.rfind('\n');
  llvm::StringRef closing_line = closing_start == llvm::StringRef::npos
                                     ? rest
                                     : rest.drop_front(closing_start + 1);
  llvm::StringRef body = closing_start == llvm::StringRef::npos
                             ? ""
                             : rest.take_front(closing_start);
  llvm::StringRef base_indent =
      closing_line.drop_back(MultiLineDelimiter.size());
  if (base_indent.find_first_not_of(" \t") != llvm::StringRef::npos) {
    // Non-whitespace before the closing `'''`; not a well-formed block to
    // touch.
    return std::nullopt;
  }

  // De-indent each body line by stripping `base_indent`. A line that doesn't
  // share that indentation (and isn't blank) means the source is mis-indented,
  // so leave the literal alone rather than risk a wrong rewrite.
  std::string snippet;
  llvm::SmallVector<llvm::StringRef> lines;
  body.split(lines, '\n');
  for (llvm::StringRef line : lines) {
    if (line.starts_with(base_indent)) {
      llvm::StringRef stripped = line.drop_front(base_indent.size());
      snippet.append(stripped.data(), stripped.size());
    } else if (!line.trim().empty()) {
      return std::nullopt;
    }
    snippet.push_back('\n');
  }
  if (llvm::StringRef(snippet).trim().empty()) {
    // Nothing to format.
    return std::nullopt;
  }
  if (llvm::StringRef(snippet).contains(MultiLineDelimiter)) {
    // A `'''` in the body (for example inside a C++ string or comment) could
    // re-form a closing delimiter once the body is re-indented, producing a
    // malformed literal. Leave such a snippet alone rather than risk that.
    return std::nullopt;
  }
  for (char c : snippet) {
    // clang-format can assert or misbehave on a control character such as a
    // NUL, so leave a body containing one unformatted; real C++ has none. A tab
    // is excluded too: clang-format preserves tabs inside string and character
    // literals, and a literal tab is invalid in Carbon multi-line string
    // content (it must be escaped), so re-encoding one would produce a
    // malformed literal.
    if (static_cast<unsigned char>(c) < ' ' && c != '\n') {
      return std::nullopt;
    }
  }

  std::optional<std::string> formatted = ReformatCpp(snippet, style, indent);
  if (!formatted) {
    return std::nullopt;
  }

  // Reassemble the literal: the original `'''<introducer>` line, then each
  // formatted line re-indented to `indent`, then the closing `'''` at `indent`.
  std::string result;
  result.append(MultiLineDelimiter.data(), MultiLineDelimiter.size());
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
  result.append(MultiLineDelimiter.data(), MultiLineDelimiter.size());
  return result;
}

}  // namespace Carbon::Format
