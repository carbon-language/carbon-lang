// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lexer/string_literal.h"

#include "common/check.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "toolchain/lexer/character_set.h"

namespace Carbon {

using LexerDiagnosticEmitter = DiagnosticEmitter<const char*>;

struct ContentBeforeStringTerminator
    : SimpleDiagnostic<ContentBeforeStringTerminator> {
  static constexpr llvm::StringLiteral ShortName = "syntax-invalid-string";
  static constexpr llvm::StringLiteral Message =
      "Only whitespace is permitted before the closing `\"\"\"` of a "
      "multi-line string.";
};

struct UnicodeEscapeTooLarge : SimpleDiagnostic<UnicodeEscapeTooLarge> {
  static constexpr llvm::StringLiteral ShortName = "syntax-invalid-string";
  static constexpr llvm::StringLiteral Message =
      "Code point specified by `\\u{...}` escape is greater than 0x10FFFF.";
};

struct UnicodeEscapeSurrogate : SimpleDiagnostic<UnicodeEscapeSurrogate> {
  static constexpr llvm::StringLiteral ShortName = "syntax-invalid-string";
  static constexpr llvm::StringLiteral Message =
      "Code point specified by `\\u{...}` escape is a surrogate character.";
};

struct UnicodeEscapeMissingBracedDigits
    : SimpleDiagnostic<UnicodeEscapeMissingBracedDigits> {
  static constexpr llvm::StringLiteral ShortName = "syntax-invalid-string";
  static constexpr llvm::StringLiteral Message =
      "Escape sequence `\\u` must be followed by a braced sequence of "
      "uppercase hexadecimal digits, for example `\\u{70AD}`.";
};

struct HexadecimalEscapeMissingDigits
    : SimpleDiagnostic<HexadecimalEscapeMissingDigits> {
  static constexpr llvm::StringLiteral ShortName = "syntax-invalid-string";
  static constexpr llvm::StringLiteral Message =
      "Escape sequence `\\x` must be followed by two "
      "uppercase hexadecimal digits, for example `\\x0F`.";
};

struct DecimalEscapeSequence : SimpleDiagnostic<DecimalEscapeSequence> {
  static constexpr llvm::StringLiteral ShortName = "syntax-invalid-string";
  static constexpr llvm::StringLiteral Message =
      "Decimal digit follows `\\0` escape sequence. Use `\\x00` instead of "
      "`\\0` if the next character is a digit.";
};

struct UnknownEscapeSequence {
  static constexpr llvm::StringLiteral ShortName = "syntax-invalid-string";
  static constexpr const char* Message = "Unrecognized escape sequence `{0}`.";

  char first;

  auto Format() -> std::string { return llvm::formatv(Message, first).str(); }
};

struct MismatchedIndentInString : SimpleDiagnostic<MismatchedIndentInString> {
  static constexpr llvm::StringLiteral ShortName = "syntax-invalid-string";
  static constexpr llvm::StringLiteral Message =
      "Indentation does not match that of the closing \"\"\" in multi-line "
      "string literal.";
};

struct InvalidHorizontalWhitespaceInString
    : SimpleDiagnostic<InvalidHorizontalWhitespaceInString> {
  static constexpr llvm::StringLiteral ShortName = "syntax-invalid-string";
  static constexpr llvm::StringLiteral Message =
      "Whitespace other than plain space must be expressed with an escape "
      "sequence in a string literal.";
};

/*
static auto TakeMultiLineStringLiteralPrefix(llvm::StringRef source_text)
    -> llvm::StringRef {
  llvm::StringRef remaining = source_text;
  if (!remaining.consume_front(R"(""")")) {
    return llvm::StringRef();
  }

  // The rest of the line must be a valid file type indicator: a sequence of
  // characters containing neither '#' nor '"' followed by a newline.
  remaining = remaining.drop_until(
      [](char c) { return c == '"' || c == '#' || c == '\n'; });
  if (!remaining.consume_front("\n")) {
    return llvm::StringRef();
  }

  return source_text.take_front(remaining.begin() - source_text.begin());
}
*/

// Returns the end of the prefix for a multi-line string literal, including the
// file type indicator and following newline. If it's not multi-line, returns
// nullptr.
static auto GetMultiLineStringLiteralPrefixEnd(const char* start,
                                               const char* source_text_end)
    -> const char* {
  constexpr char Quotes[] = R"(""")";
  if (source_text_end - start < static_cast<int>(strlen(Quotes)) ||
      memcmp(start, Quotes, strlen(Quotes) * sizeof(char)) != 0) {
    return nullptr;
  }
  for (const char* cursor = start + strlen(Quotes); cursor < source_text_end;
       ++cursor) {
    switch (*cursor) {
      case '#':
      case '"':
        return nullptr;
      case '\n':
        return cursor + 1;
    }
  }
  return nullptr;
}

// If source_text begins with a string literal token, extract and return
// information on that token.
auto LexedStringLiteral::Lex(llvm::StringRef source_text)
    -> llvm::Optional<LexedStringLiteral> {
  const char* text_begin = source_text.begin();
  const char* source_text_end = source_text.end();

  size_t hash_level = source_text.find_first_not_of('#');
  if (hash_level == llvm::StringRef::npos) {
    return llvm::None;
  }
  const char* content_begin = text_begin + hash_level;

  llvm::SmallString<16> terminator("\"");
  llvm::SmallString<16> escape("\\");

  const char* multi_line_prefix_end =
      GetMultiLineStringLiteralPrefixEnd(content_begin, source_text_end);
  bool multi_line = multi_line_prefix_end != nullptr;
  if (multi_line) {
    content_begin = multi_line_prefix_end;
    terminator = R"(""")";
  } else if (content_begin < source_text_end && *content_begin == '"') {
    ++content_begin;
  } else {
    return llvm::None;
  }

  // The terminator and escape sequence marker require a number of '#'s
  // matching the leading sequence of '#'s.
  terminator.resize(terminator.size() + hash_level, '#');
  escape.resize(escape.size() + hash_level, '#');

  for (const char* cursor = content_begin; cursor != source_text_end;
       ++cursor) {
    switch (*cursor) {
      case '\\':
        // This relies on multi-character escape sequences not containing an
        // embedded and unescaped terminator or newline.
        if (hash_level == 0 || llvm::StringRef(cursor, source_text_end - cursor)
                                   .startswith(escape)) {
          cursor += escape.size();
          // If there's either not a character following the escape, or it's a
          // single-line string and the escaped character is a newline, we
          // should stop here.
          if (cursor == source_text_end || (!multi_line && *cursor == '\n')) {
            return llvm::None;
          }
        }
        break;
      case '\n':
        if (!multi_line) {
          return llvm::None;
        }
        break;
      case '\"': {
        if (llvm::StringRef(cursor, source_text_end - cursor)
                .startswith(terminator)) {
          llvm::StringRef text(text_begin,
                               cursor - text_begin + terminator.size());
          llvm::StringRef content(content_begin, cursor - content_begin);
          return LexedStringLiteral(text, content, hash_level, multi_line);
        }
        break;
      }
    }
  }
  // Let LexError figure out how to recover from an unterminated string
  // literal.
  return llvm::None;

  /*
  fastbuild
  BM_ValidString_Simple                  305411 ns       305385 ns         2266
  BM_ValidString_Multiline               307486 ns       307446 ns         2282
  BM_ValidString_Raw                     306693 ns       306667 ns         2293
  BM_IncompleteWithEscapes_Simple        275252 ns       275225 ns         2503
  BM_IncompleteWithEscapes_Multiline     263120 ns       263085 ns         2608
  BM_IncompleteWithEscapes_Raw           693668 ns       693588 ns          997
  opt
  BM_ValidString_Simple                   69245 ns        69238 ns        10060
  BM_ValidString_Multiline                67667 ns        67661 ns        10185
  BM_ValidString_Raw                      67697 ns        67691 ns        10244
  BM_IncompleteWithEscapes_Simple         68324 ns        68315 ns        10399
  BM_IncompleteWithEscapes_Multiline      50578 ns        50574 ns        13943
  BM_IncompleteWithEscapes_Raw           134478 ns       134462 ns         5288
  */
  /*
    const char* begin = source_text.begin();

    int hash_level = 0;
    while (source_text.consume_front("#")) {
      ++hash_level;
    }

    llvm::SmallString<16> terminator("\"");
    llvm::SmallString<16> escape("\\");

    llvm::StringRef multi_line_prefix =
        TakeMultiLineStringLiteralPrefix(source_text);
    bool multi_line = !multi_line_prefix.empty();
    if (multi_line) {
      source_text = source_text.drop_front(multi_line_prefix.size());
      terminator = R"(""")";
    } else if (!source_text.consume_front("\"")) {
      return llvm::None;
    }

    // The terminator and escape sequence marker require a number of '#'s
    // matching the leading sequence of '#'s.
    const int terminator_size = terminator.size() + hash_level;
    terminator.resize(terminator_size, '#');
    const int escape_size = escape.size() + hash_level;
    escape.resize(escape_size, '#');

    const char* cursor = source_text.begin();
    const char* source_text_end = source_text.end();
    while (cursor != source_text_end) {
      switch (*cursor) {
        case '\\':
          // This relies on multi-character escape sequences not containing an
          // embedded and unescaped terminator or newline.
          if (escape_size == 1 ||
              llvm::StringRef(cursor, source_text_end - cursor)
                  .startswith(escape)) {
            cursor += escape_size;
            // If there's either not a character following the escape, or it's a
            // single-line string and the escaped character is a newline, we
            // should stop here.
            if (cursor == source_text_end || (!multi_line && *cursor == '\n')) {
              return llvm::None;
            }
          }
          break;
        case '\n':
          if (!multi_line) {
            return llvm::None;
          }
          break;
        case '\"': {
          if (llvm::StringRef(cursor, source_text_end - cursor)
                  .startswith(terminator)) {
            return LexedStringLiteral(
                llvm::StringRef(begin, cursor - begin + terminator_size),
                llvm::StringRef(source_text.begin(),
                                cursor - source_text.begin()),
                hash_level, multi_line);
          }
          break;
        }
      }
      ++cursor;
    }
    return llvm::None;
  */

  /*
BM_ValidString_Simple                  545808 ns       545716 ns         1288
BM_ValidString_Multiline               544358 ns       544288 ns         1294
BM_ValidString_Raw                     560736 ns       560682 ns         1278
BM_IncompleteWithEscapes_Simple       2410269 ns      2409978 ns          290
BM_IncompleteWithEscapes_Multiline    2288780 ns      2288446 ns          306
BM_IncompleteWithEscapes_Raw          2530670 ns      2530243 ns          288
opt
BM_ValidString_Simple                  102361 ns       102350 ns         6788
BM_ValidString_Multiline               103207 ns       103198 ns         6878
BM_ValidString_Raw                     102159 ns       102139 ns         6746
BM_IncompleteWithEscapes_Simple        270925 ns       270880 ns         2535
BM_IncompleteWithEscapes_Multiline     269758 ns       269723 ns         2603
BM_IncompleteWithEscapes_Raw           242358 ns       242332 ns         2865
  */
  /*
    int introducer_size = source_text.begin() - begin;
    llvm::StringRef look_for = "\\\n\"";
    size_t cursor = 0;
    const size_t source_text_size = source_text.size();
    for (;;) {
      cursor = source_text.find_first_of(look_for, cursor);
      if (cursor == llvm::StringRef::npos) {
        return llvm::None;
      }
      switch (source_text[cursor]) {
        case '\\':
          if (escape_size == 1 || source_text.substr(cursor).startswith(escape))
    { cursor += escape_size; if (!multi_line && source_text_size > cursor &&
                source_text[cursor] == '\n') {
              return llvm::None;
            }
          }
          break;
        case '\n':
          if (!multi_line) {
            return llvm::None;
          }
          break;
        case '\"': {
          if (source_text.substr(cursor).startswith(terminator)) {
            return LexedStringLiteral(
                llvm::StringRef(begin,
                                introducer_size + cursor + terminator_size),
                source_text.substr(0, cursor), hash_level, multi_line);
          }
          break;
        }
        default:
          FATAL() << "find_first_of must correspond to an above character";
      }
      ++cursor;
    }
  */

  /*
fastbuild
BM_ValidString_Simple                 6687822 ns      6687109 ns          105
BM_ValidString_Multiline              4734796 ns      4734125 ns          149
BM_ValidString_Raw                    6650717 ns      6649633 ns          104
BM_IncompleteWithEscapes_Simple       5274067 ns      5273569 ns          132
BM_IncompleteWithEscapes_Multiline    3831291 ns      3830516 ns          183
BM_IncompleteWithEscapes_Raw          4227835 ns      4227115 ns          166
opt
BM_ValidString_Simple                  642729 ns       642643 ns         1067
BM_ValidString_Multiline               703019 ns       702949 ns          990
BM_ValidString_Raw                     801925 ns       801792 ns          874
BM_IncompleteWithEscapes_Simple        500082 ns       499979 ns         1368
BM_IncompleteWithEscapes_Multiline     563000 ns       562947 ns         1247
BM_IncompleteWithEscapes_Raw           486858 ns       486816 ns         1451
  */
  /*
    const char* content_begin = source_text.begin();
    const char* content_end = content_begin;
    while (!source_text.consume_front(terminator)) {
      // Let LexError figure out how to recover from an unterminated string
      // literal.
      if (source_text.empty()) {
        return llvm::None;
      }

      // Consume an escape sequence marker if present.
      (void)source_text.consume_front(escape);

      // Then consume one more character, either of the content or of an
      // escape sequence. This can be a newline in a multi-line string literal.
      // This relies on multi-character escape sequences not containing an
      // embedded and unescaped terminator or newline.
      if (!multi_line && source_text.startswith("\n")) {
        return llvm::None;
      }
      source_text = source_text.substr(1);
      content_end = source_text.begin();
    }

    return LexedStringLiteral(
        llvm::StringRef(begin, source_text.begin() - begin),
        llvm::StringRef(content_begin, content_end - content_begin), hash_level,
        multi_line);
        */
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
static auto CheckIndent(LexerDiagnosticEmitter& emitter, llvm::StringRef text,
                        llvm::StringRef content) -> llvm::StringRef {
  // Find the leading horizontal whitespace on the final line of this literal.
  // Note that for an empty literal, this might not be inside the content.
  llvm::StringRef indent = ComputeIndentOfFinalLine(text);

  // The last line is not permitted to contain any content after its
  // indentation.
  if (indent.end() != content.end()) {
    emitter.EmitError<ContentBeforeStringTerminator>(indent.end());
  }

  return indent;
}

// Expand a `\u{HHHHHH}` escape sequence into a sequence of UTF-8 code units.
static auto ExpandUnicodeEscapeSequence(LexerDiagnosticEmitter& emitter,
                                        llvm::StringRef digits,
                                        std::string& result) -> bool {
  unsigned code_point;
  if (digits.getAsInteger(16, code_point) || code_point > 0x10FFFF) {
    emitter.EmitError<UnicodeEscapeTooLarge>(digits.begin());
    return false;
  }

  if (code_point >= 0xD800 && code_point < 0xE000) {
    emitter.EmitError<UnicodeEscapeSurrogate>(digits.begin());
    return false;
  }

  // Convert the code point to a sequence of UTF-8 code units.
  // Every code point fits in 6 UTF-8 code units.
  const llvm::UTF32 utf32_code_units[1] = {code_point};
  llvm::UTF8 utf8_code_units[6];
  const llvm::UTF32* src_pos = utf32_code_units;
  llvm::UTF8* dest_pos = utf8_code_units;
  llvm::ConversionResult conv_result = llvm::ConvertUTF32toUTF8(
      &src_pos, src_pos + 1, &dest_pos, dest_pos + 6, llvm::strictConversion);
  if (conv_result != llvm::conversionOK) {
    llvm_unreachable("conversion of valid code point to UTF-8 cannot fail");
  }
  result.insert(result.end(), reinterpret_cast<char*>(utf8_code_units),
                reinterpret_cast<char*>(dest_pos));
  return true;
}

// Expand an escape sequence, appending the expanded value to the given
// `result` string. `content` is the string content, starting from the first
// character after the escape sequence introducer (for example, the `n` in
// `\n`), and will be updated to remove the leading escape sequence.
static auto ExpandAndConsumeEscapeSequence(LexerDiagnosticEmitter& emitter,
                                           llvm::StringRef& content,
                                           std::string& result) -> void {
  CHECK(!content.empty()) << "should have escaped closing delimiter";
  char first = content.front();
  content = content.drop_front(1);

  switch (first) {
    case 't':
      result += '\t';
      return;
    case 'n':
      result += '\n';
      return;
    case 'r':
      result += '\r';
      return;
    case '"':
      result += '"';
      return;
    case '\'':
      result += '\'';
      return;
    case '\\':
      result += '\\';
      return;
    case '0':
      result += '\0';
      if (!content.empty() && IsDecimalDigit(content.front())) {
        emitter.EmitError<DecimalEscapeSequence>(content.begin());
        return;
      }
      return;
    case 'x':
      if (content.size() >= 2 && IsUpperHexDigit(content[0]) &&
          IsUpperHexDigit(content[1])) {
        result +=
            static_cast<char>(llvm::hexFromNibbles(content[0], content[1]));
        content = content.drop_front(2);
        return;
      }
      emitter.EmitError<HexadecimalEscapeMissingDigits>(content.begin());
      break;
    case 'u': {
      llvm::StringRef remaining = content;
      if (remaining.consume_front("{")) {
        llvm::StringRef digits = remaining.take_while(IsUpperHexDigit);
        remaining = remaining.drop_front(digits.size());
        if (!digits.empty() && remaining.consume_front("}")) {
          if (!ExpandUnicodeEscapeSequence(emitter, digits, result)) {
            break;
          }
          content = remaining;
          return;
        }
      }
      emitter.EmitError<UnicodeEscapeMissingBracedDigits>(content.begin());
      break;
    }
    default:
      emitter.EmitError<UnknownEscapeSequence>(content.begin() - 1,
                                               {.first = first});
      break;
  }

  // If we get here, we didn't recognize this escape sequence and have already
  // issued a diagnostic. For error recovery purposes, expand this escape
  // sequence to itself, dropping the introducer (for example, `\q` -> `q`).
  result += first;
}

// Expand any escape sequences in the given string literal.
static auto ExpandEscapeSequencesAndRemoveIndent(
    LexerDiagnosticEmitter& emitter, llvm::StringRef contents, int hash_level,
    llvm::StringRef indent) -> std::string {
  std::string result;
  result.reserve(contents.size());

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
      if (!contents.startswith("\n")) {
        emitter.EmitError<MismatchedIndentInString>(line_start);
      }
    }

    // Process the contents of the line.
    while (true) {
      auto end_of_regular_text = contents.find_if([](char c) {
        return c == '\n' || c == '\\' ||
               (IsHorizontalWhitespace(c) && c != ' ');
      });
      result += contents.substr(0, end_of_regular_text);
      contents = contents.substr(end_of_regular_text);

      if (contents.empty()) {
        return result;
      }

      if (contents.consume_front("\n")) {
        // Trailing whitespace before a newline doesn't contribute to the string
        // literal value.
        while (!result.empty() && result.back() != '\n' &&
               IsSpace(result.back())) {
          result.pop_back();
        }
        result += '\n';
        // Move onto to the next line.
        break;
      }

      if (IsHorizontalWhitespace(contents.front())) {
        // Horizontal whitespace other than ` ` is valid only at the end of a
        // line.
        CHECK(contents.front() != ' ')
            << "should not have stopped at a plain space";
        auto after_space = contents.find_if_not(IsHorizontalWhitespace);
        if (after_space == llvm::StringRef::npos ||
            contents[after_space] != '\n') {
          // TODO: Include the source range of the whitespace up to
          // `contents.begin() + after_space` in the diagnostic.
          emitter.EmitError<InvalidHorizontalWhitespaceInString>(
              contents.begin());
          // Include the whitespace in the string contents for error recovery.
          result += contents.substr(0, after_space);
        }
        contents = contents.substr(after_space);
        continue;
      }

      if (!contents.consume_front(escape)) {
        // This is not an escape sequence, just a raw `\`.
        result += contents.front();
        contents = contents.drop_front(1);
        continue;
      }

      if (contents.consume_front("\n")) {
        // An escaped newline ends the line without producing any content and
        // without trimming trailing whitespace.
        break;
      }

      // Handle this escape sequence.
      ExpandAndConsumeEscapeSequence(emitter, contents, result);
    }
  }
}

auto LexedStringLiteral::ComputeValue(LexerDiagnosticEmitter& emitter) const
    -> std::string {
  llvm::StringRef indent =
      multi_line_ ? CheckIndent(emitter, text_, content_) : llvm::StringRef();
  return ExpandEscapeSequencesAndRemoveIndent(emitter, content_, hash_level_,
                                              indent);
}

}  // namespace Carbon
