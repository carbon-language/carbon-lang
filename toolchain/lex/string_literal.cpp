// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lex/string_literal.h"

#include "common/check.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/ErrorHandling.h"
#include "toolchain/lex/character_set.h"
#include "toolchain/lex/helpers.h"

namespace Carbon::Lex {

using DiagnosticEmitter = Diagnostics::Emitter<const char*>;

static constexpr char MultiLineIndicator[] = R"(''')";
static constexpr char DoubleQuotedMultiLineIndicator[] = R"(""")";

struct StringLiteral::Introducer {
  // The kind of string being introduced.
  MultiLineKind kind;
  // The terminator for the string, without any '#' suffixes.
  llvm::StringRef terminator;
  // The length of the introducer, including the file type indicator and
  // newline for a multi-line string literal.
  int prefix_size;

  // Lex the introducer for a string literal, after any '#'s.
  static auto Lex(llvm::StringRef source_text) -> std::optional<Introducer>;
};

// Lex the introducer for a string literal, after any '#'s.
//
// We lex multi-line literals when spelled with either ''' or """ for error
// recovery purposes, and reject """ literals after lexing.
auto StringLiteral::Introducer::Lex(llvm::StringRef source_text)
    -> std::optional<Introducer> {
  MultiLineKind kind = NotMultiLine;
  llvm::StringRef indicator;
  if (source_text.starts_with(MultiLineIndicator)) {
    kind = MultiLine;
    indicator = llvm::StringRef(MultiLineIndicator);
  } else if (source_text.starts_with(DoubleQuotedMultiLineIndicator)) {
    kind = MultiLineWithDoubleQuotes;
    indicator = llvm::StringRef(DoubleQuotedMultiLineIndicator);
  }

  if (kind != NotMultiLine) {
    // The rest of the line must be a valid file type indicator: a sequence of
    // characters containing neither '#' nor '"' followed by a newline.
    auto prefix_end = source_text.find_first_of("#\n\"", indicator.size());
    if (prefix_end != llvm::StringRef::npos &&
        source_text[prefix_end] == '\n') {
      // Include the newline in the prefix size.
      return Introducer{.kind = kind,
                        .terminator = indicator,
                        .prefix_size = static_cast<int>(prefix_end + 1)};
    }
  }

  if (!source_text.empty() && source_text[0] == '"') {
    return Introducer{
        .kind = NotMultiLine, .terminator = "\"", .prefix_size = 1};
  }

  return std::nullopt;
}

namespace {
// A set of 'char' values.
struct alignas(8) CharSet {
  bool Elements[UCHAR_MAX + 1];

  constexpr CharSet(std::initializer_list<char> chars) : Elements() {
    for (char c : chars) {
      Elements[static_cast<unsigned char>(c)] = true;
    }
  }

  constexpr auto operator[](char c) const -> bool {
    return Elements[static_cast<unsigned char>(c)];
  }
};
}  // namespace

auto StringLiteral::Lex(llvm::StringRef source_text)
    -> std::optional<StringLiteral> {
  int64_t cursor = 0;
  const int64_t source_text_size = source_text.size();

  // Determine the number of hashes prefixing.
  while (cursor < source_text_size && source_text[cursor] == '#') {
    ++cursor;
  }
  const int hash_level = cursor;

  const std::optional<Introducer> introducer =
      Introducer::Lex(source_text.substr(hash_level));
  if (!introducer) {
    return std::nullopt;
  }

  cursor += introducer->prefix_size;
  const int prefix_len = cursor;

  llvm::SmallString<16> terminator(introducer->terminator);
  llvm::SmallString<16> escape("\\");

  // The terminator and escape sequence marker require a number of '#'s
  // matching the leading sequence of '#'s.
  terminator.resize(terminator.size() + hash_level, '#');
  escape.resize(escape.size() + hash_level, '#');

  bool content_needs_validation = false;

  // TODO: Detect indent / dedent for multi-line string literals in order to
  // stop parsing on dedent before a terminator is found.
  for (; cursor < source_text_size; ++cursor) {
    // Use a lookup table to allow us to quickly skip uninteresting characters.
    static constexpr CharSet InterestingChars = {'\\', '\n', '"', '\'', '\t'};
    if (!InterestingChars[source_text[cursor]]) {
      continue;
    }

    // This switch and loop structure relies on multi-character terminators and
    // escape sequences starting with a predictable character and not containing
    // embedded and unescaped terminators or newlines.
    switch (source_text[cursor]) {
      case '\t':
        // Tabs have extra validation.
        content_needs_validation = true;
        break;
      case '\\':
        if (escape.size() == 1 ||
            source_text.substr(cursor + 1).starts_with(escape.substr(1))) {
          content_needs_validation = true;
          cursor += escape.size();
          // If there's either not a character following the escape, or it's a
          // single-line string and the escaped character is a newline, we
          // should stop here.
          if (cursor >= source_text_size || (introducer->kind == NotMultiLine &&
                                             source_text[cursor] == '\n')) {
            llvm::StringRef text = source_text.take_front(cursor);
            return StringLiteral(text, text.drop_front(prefix_len),
                                 content_needs_validation, hash_level,
                                 introducer->kind,
                                 /*is_terminated=*/false);
          }
        }
        break;
      case '\n':
        if (introducer->kind == NotMultiLine) {
          llvm::StringRef text = source_text.take_front(cursor);
          return StringLiteral(text, text.drop_front(prefix_len),
                               content_needs_validation, hash_level,
                               introducer->kind,
                               /*is_terminated=*/false);
        }
        break;
      case '"':
      case '\'':
        if (source_text.substr(cursor).starts_with(terminator)) {
          llvm::StringRef text =
              source_text.substr(0, cursor + terminator.size());
          llvm::StringRef content =
              source_text.substr(prefix_len, cursor - prefix_len);
          return StringLiteral(text, content, content_needs_validation,
                               hash_level, introducer->kind,
                               /*is_terminated=*/true);
        }
        break;
      default:
        // No action for non-terminators.
        break;
    }
  }
  // No terminator was found.
  return StringLiteral(source_text, source_text.drop_front(prefix_len),
                       content_needs_validation, hash_level, introducer->kind,
                       /*is_terminated=*/false);
}

// Given a string that contains at least one newline, find the indent (the
// leading sequence of horizontal whitespace) of its final line.
static auto ComputeIndentOfFinalLine(llvm::StringRef text) -> llvm::StringRef {
  int indent_end = text.size();
  for (int i = indent_end - 1; i >= 0; --i) {
    if (text[i] == '\n') {
      int indent_start = i + 1;
      return text.substr(indent_start, indent_end - indent_start);
    }
    if (!IsSpace(text[i])) {
      indent_end = i;
    }
  }
  llvm_unreachable("Given text is required to contain a newline.");
}

// Check the literal is indented properly, if it's a multi-line litera.
// Find the leading whitespace that should be removed from each line of a
// multi-line string literal.
static auto CheckIndent(DiagnosticEmitter& emitter, llvm::StringRef text,
                        llvm::StringRef content) -> llvm::StringRef {
  // Find the leading horizontal whitespace on the final line of this literal.
  // Note that for an empty literal, this might not be inside the content.
  llvm::StringRef indent = ComputeIndentOfFinalLine(text);

  // The last line is not permitted to contain any content after its
  // indentation.
  if (indent.end() != content.end()) {
    CARBON_DIAGNOSTIC(
        ContentBeforeStringTerminator, Error,
        "only whitespace is permitted before the closing `'''` of a "
        "multi-line string");
    emitter.Emit(indent.end(), ContentBeforeStringTerminator);
  }

  return indent;
}

// Expand a `\u{HHHHHH}` escape sequence into a sequence of UTF-8 code units.
static auto ExpandUnicodeEscapeSequence(DiagnosticEmitter& emitter,
                                        llvm::StringRef digits,
                                        char*& buffer_cursor) -> bool {
  unsigned code_point;
  if (!CanLexInt(emitter, digits)) {
    return false;
  }
  if (digits.getAsInteger(16, code_point) || code_point > 0x10FFFF) {
    CARBON_DIAGNOSTIC(UnicodeEscapeTooLarge, Error,
                      "code point specified by `\\u{{...}}` escape is greater "
                      "than 0x10FFFF");
    emitter.Emit(digits.begin(), UnicodeEscapeTooLarge);
    return false;
  }

  if (code_point >= 0xD800 && code_point < 0xE000) {
    CARBON_DIAGNOSTIC(UnicodeEscapeSurrogate, Error,
                      "code point specified by `\\u{{...}}` escape is a "
                      "surrogate character");
    emitter.Emit(digits.begin(), UnicodeEscapeSurrogate);
    return false;
  }

  // Convert the code point to a sequence of UTF-8 code units.
  // Every code point fits in 6 UTF-8 code units.
  const llvm::UTF32 utf32_code_units[1] = {code_point};
  const llvm::UTF32* src_pos = utf32_code_units;
  auto*& buffer_cursor_as_utf8 = reinterpret_cast<llvm::UTF8*&>(buffer_cursor);
  llvm::ConversionResult conv_result = llvm::ConvertUTF32toUTF8(
      &src_pos, src_pos + 1, &buffer_cursor_as_utf8, buffer_cursor_as_utf8 + 6,
      llvm::strictConversion);
  if (conv_result != llvm::conversionOK) {
    llvm_unreachable("conversion of valid code point to UTF-8 cannot fail");
  }
  return true;
}

// Appends a character to the buffer and advances the cursor.
static auto AppendChar(char*& buffer_cursor, char append_char) -> void {
  buffer_cursor[0] = append_char;
  ++buffer_cursor;
}

// Appends the front of contents to the buffer and advances the cursor.
static auto AppendFrontOfContents(char*& buffer_cursor,
                                  llvm::StringRef contents, size_t len_or_npos)
    -> void {
  auto len =
      len_or_npos == llvm::StringRef::npos ? contents.size() : len_or_npos;
  memcpy(buffer_cursor, contents.data(), len);
  buffer_cursor += len;
}

// Expand an escape sequence, appending the expanded value to the given
// `result` string. `content` is the string content, starting from the first
// character after the escape sequence introducer (for example, the `n` in
// `\n`), and will be updated to remove the leading escape sequence.
static auto ExpandAndConsumeEscapeSequence(DiagnosticEmitter& emitter,
                                           llvm::StringRef& content,
                                           char*& buffer_cursor) -> void {
  CARBON_CHECK(!content.empty(), "should have escaped closing delimiter");
  char first = content.front();
  content = content.drop_front(1);

  switch (first) {
    case 't':
      AppendChar(buffer_cursor, '\t');
      return;
    case 'n':
      AppendChar(buffer_cursor, '\n');
      return;
    case 'r':
      AppendChar(buffer_cursor, '\r');
      return;
    case '"':
      AppendChar(buffer_cursor, '"');
      return;
    case '\'':
      AppendChar(buffer_cursor, '\'');
      return;
    case '\\':
      AppendChar(buffer_cursor, '\\');
      return;
    case '0':
      AppendChar(buffer_cursor, '\0');
      if (!content.empty() && IsDecimalDigit(content.front())) {
        CARBON_DIAGNOSTIC(
            DecimalEscapeSequence, Error,
            "decimal digit follows `\\0` escape sequence. Use `\\x00` instead "
            "of `\\0` if the next character is a digit");
        emitter.Emit(content.begin(), DecimalEscapeSequence);
        return;
      }
      return;
    case 'x':
      if (content.size() >= 2 && IsUpperHexDigit(content[0]) &&
          IsUpperHexDigit(content[1])) {
        AppendChar(buffer_cursor, static_cast<char>(llvm::hexFromNibbles(
                                      content[0], content[1])));
        content = content.drop_front(2);
        return;
      }
      CARBON_DIAGNOSTIC(HexadecimalEscapeMissingDigits, Error,
                        "escape sequence `\\x` must be followed by two "
                        "uppercase hexadecimal digits, for example `\\x0F`");
      emitter.Emit(content.begin(), HexadecimalEscapeMissingDigits);
      break;
    case 'u': {
      llvm::StringRef remaining = content;
      if (remaining.consume_front("{")) {
        llvm::StringRef digits = remaining.take_while(IsUpperHexDigit);
        remaining = remaining.drop_front(digits.size());
        if (!digits.empty() && remaining.consume_front("}")) {
          if (!ExpandUnicodeEscapeSequence(emitter, digits, buffer_cursor)) {
            break;
          }
          content = remaining;
          return;
        }
      }
      CARBON_DIAGNOSTIC(
          UnicodeEscapeMissingBracedDigits, Error,
          "escape sequence `\\u` must be followed by a braced sequence of "
          "uppercase hexadecimal digits, for example `\\u{{70AD}}`");
      emitter.Emit(content.begin(), UnicodeEscapeMissingBracedDigits);
      break;
    }
    default:
      CARBON_DIAGNOSTIC(UnknownEscapeSequence, Error,
                        "unrecognized escape sequence `{0}`", char);
      emitter.Emit(content.begin() - 1, UnknownEscapeSequence, first);
      break;
  }

  // If we get here, we didn't recognize this escape sequence and have already
  // issued a diagnostic. For error recovery purposes, expand this escape
  // sequence to itself, dropping the introducer (for example, `\q` -> `q`).
  AppendChar(buffer_cursor, first);
}

// Expand any escape sequences in the given string literal.
static auto ExpandEscapeSequencesAndRemoveIndent(
    DiagnosticEmitter& emitter, llvm::StringRef contents, int hash_level,
    llvm::StringRef indent, char* buffer) -> llvm::StringRef {
  char* buffer_cursor = buffer;

  llvm::SmallString<16> escape("\\");
  escape.resize(1 + hash_level, '#');

  // Process each line of the string literal.
  while (true) {
    // Every non-empty line (that contains anything other than horizontal
    // whitespace) is required to start with the string's indent. For error
    // recovery, remove all leading whitespace if the indent doesn't match.
    if (!contents.consume_front(indent)) {
      const char* line_start = contents.begin();
      contents = contents.drop_while(IsHorizontalWhitespace);
      if (!contents.starts_with("\n")) {
        CARBON_DIAGNOSTIC(
            MismatchedIndentInString, Error,
            "indentation does not match that of the closing `'''` in "
            "multi-line string literal");
        emitter.Emit(line_start, MismatchedIndentInString);
      }
    }

    // Tracks the position at the last time we expanded an escape to ensure we
    // don't misinterpret it as unescaped when backtracking.
    char* buffer_last_escape = buffer_cursor;

    // Process the contents of the line.
    while (true) {
      // Append the next segment of plain text.
      auto end_of_regular_text = contents.find_if([](char c) {
        return c == '\n' || c == '\\' ||
               (IsHorizontalWhitespace(c) && c != ' ');
      });
      AppendFrontOfContents(buffer_cursor, contents, end_of_regular_text);
      if (end_of_regular_text == llvm::StringRef::npos) {
        return llvm::StringRef(buffer, buffer_cursor - buffer);
      }
      contents = contents.drop_front(end_of_regular_text);

      if (contents.consume_front("\n")) {
        // Trailing whitespace in the source before a newline doesn't contribute
        // to the string literal value. However, escaped whitespace (like `\t`)
        // and any whitespace just before that does contribute.
        while (buffer_cursor > buffer_last_escape) {
          char back = *(buffer_cursor - 1);
          if (back == '\n' || !IsSpace(back)) {
            break;
          }
          --buffer_cursor;
        }
        AppendChar(buffer_cursor, '\n');
        // Move onto to the next line.
        break;
      }

      if (IsHorizontalWhitespace(contents.front())) {
        // Horizontal whitespace other than ` ` is valid only at the end of a
        // line.
        CARBON_CHECK(contents.front() != ' ',
                     "should not have stopped at a plain space");
        auto after_space = contents.find_if_not(IsHorizontalWhitespace);
        if (after_space == llvm::StringRef::npos ||
            contents[after_space] != '\n') {
          // TODO: Include the source range of the whitespace up to
          // `contents.begin() + after_space` in the diagnostic.
          CARBON_DIAGNOSTIC(
              InvalidHorizontalWhitespaceInString, Error,
              "whitespace other than plain space must be expressed with an "
              "escape sequence in a string literal");
          emitter.Emit(contents.begin(), InvalidHorizontalWhitespaceInString);
          // Include the whitespace in the string contents for error recovery.
          AppendFrontOfContents(buffer_cursor, contents, after_space);
        }
        contents = contents.substr(after_space);
        continue;
      }

      if (!contents.consume_front(escape)) {
        // This is not an escape sequence, just a raw `\`.
        AppendChar(buffer_cursor, contents.front());
        contents = contents.drop_front(1);
        continue;
      }

      if (contents.consume_front("\n")) {
        // An escaped newline ends the line without producing any content and
        // without trimming trailing whitespace.
        break;
      }

      // Handle this escape sequence.
      ExpandAndConsumeEscapeSequence(emitter, contents, buffer_cursor);
      buffer_last_escape = buffer_cursor;
    }
  }
}

auto StringLiteral::ComputeValue(llvm::BumpPtrAllocator& allocator,
                                 DiagnosticEmitter& emitter) const
    -> llvm::StringRef {
  if (!is_terminated_) {
    return "";
  }
  if (multi_line_ == MultiLineWithDoubleQuotes) {
    CARBON_DIAGNOSTIC(
        MultiLineStringWithDoubleQuotes, Error,
        "use `'''` delimiters for a multi-line string literal, not `\"\"\"`");
    emitter.Emit(text_.begin(), MultiLineStringWithDoubleQuotes);
  }
  llvm::StringRef indent =
      multi_line_ ? CheckIndent(emitter, text_, content_) : llvm::StringRef();
  if (!content_needs_validation_ && (!multi_line_ || indent.empty())) {
    return content_;
  }

  // "Expanding" escape sequences should only ever shorten content. As a
  // consequence, the output string should allows fit within this allocation.
  // Although this may waste some space, it avoids a reallocation.
  auto result = ExpandEscapeSequencesAndRemoveIndent(
      emitter, content_, hash_level_, indent,
      allocator.Allocate<char>(content_.size()));
  CARBON_CHECK(result.size() <= content_.size(),
               "Content grew from {0} to {1}: `{2}`", content_.size(),
               result.size(), content_);
  return result;
}

}  // namespace Carbon::Lex
