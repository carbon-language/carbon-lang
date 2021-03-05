// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "lexer/string_literal.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"

namespace Carbon {

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

  struct Substitutions { char first; };
  static auto Format(const Substitutions& subst) -> std::string {
    return llvm::formatv(Message, subst.first).str();
  }
};

struct MismatchedIndentInString : SimpleDiagnostic<MismatchedIndentInString> {
  static constexpr llvm::StringLiteral ShortName = "syntax-invalid-string";
  static constexpr llvm::StringLiteral Message =
      "Indentation does not match that of the closing \"\"\" in multi-line "
      "string literal.";
};

// TODO(zygoloid): Update this to match whatever we decide qualifies as
// acceptable whitespace.
static bool isSpace(char c) {
  return c == ' ' || c == '\n' || c == '\t';
}

static constexpr llvm::StringLiteral HorizontalWhitespace = " \t";

static bool isUpperHexDigit(char c) {
  return ('0' <= c && c <= '9') || ('A' <= c && c <= 'F');
}

// Find and return the opening characters of a multi-line string literal,
// after any '#'s, including the file type indicator and following newline.
static auto TakeMultiLineStringLiteralPrefix(llvm::StringRef source_text)
    -> llvm::StringRef {
  const char *begin = source_text.begin();
  if (!source_text.consume_front("\"\"\"")) {
    return llvm::StringRef();
  }

  // The rest of the line must be a valid file type indicator: a sequence of
  // characters containing neither '#' nor '"' followed by a newline.
  auto file_type_length = source_text.find_first_of("\"#\n");
  if (file_type_length == source_text.npos ||
      source_text[file_type_length] != '\n') {
    return llvm::StringRef();
  }

  return llvm::StringRef(begin, 3 + file_type_length + 1);
}

// If source_text begins with a string literal token, extract and return
// information on that token.
auto StringLiteralToken::Lex(llvm::StringRef source_text)
    -> llvm::Optional<StringLiteralToken> {
  const char *begin = source_text.begin();

  int hash_level = 0;
  while (source_text.consume_front("#"))
    ++hash_level;

  llvm::SmallString<16> terminator("\"");
  llvm::SmallString<16> escape("\\");

  llvm::StringRef multi_line_prefix =
      TakeMultiLineStringLiteralPrefix(source_text);
  bool multi_line = !multi_line_prefix.empty();
  if (multi_line) {
    source_text = source_text.drop_front(multi_line_prefix.size());
    terminator = "\"\"\"";
  } else if (!source_text.consume_front("\"")) {
    return llvm::None;
  }

  // The terminator and escape sequence marker require a number of '#'s
  // matching the leading sequence of '#'s.
  terminator.resize(terminator.size() + hash_level, '#');
  escape.resize(escape.size() + hash_level, '#');

  const char *content_begin = source_text.begin();
  const char *content_end = content_begin;
  while (!source_text.consume_front(terminator)) {
    // Let LexError figure out how to recover from an unterminated string
    // literal.
    if (source_text.empty())
      return llvm::None;
    if (!multi_line && source_text.startswith("\n"))
      return llvm::None;

    // Consume an escape sequence marker if present.
    (void)source_text.consume_front(escape);
    // Then consume one more character, either of the content or of an
    // escape sequence. This relies on multi-character escape sequences
    // not containing an embedded and unescaped terminator or newline.
    source_text = source_text.substr(1);
    content_end = source_text.begin();
  }

  return StringLiteralToken(
      llvm::StringRef(begin, source_text.begin() - begin),
      llvm::StringRef(content_begin, content_end - content_begin), hash_level,
      multi_line);
}

// Given a string that contains at least one newline, find the indent (the
// leading sequence of horizontal whitespace) of its final line.
static auto ComputeIndentOfFinalLine(llvm::StringRef text) -> llvm::StringRef {
  int indent_end = text.size();
  for (int i = indent_end - 1; i >= 0; --i) {
    if (text[i] == '\n') {
      return text.substr(i + 1, indent_end - i - 1);
    }
    if (!isSpace(text[i])) {
      indent_end = i;
    }
  }
  llvm_unreachable("Given text is required to contain a newline.");
}

namespace {
// The leading whitespace in a multi-line string literal.
struct Indent {
  llvm::StringRef indent;
  bool has_errors;
};
}

// Check the literal is indented properly, if it's a multi-line litera.
// Find the leading whitespace that should be removed from each line of a
// multi-line string literal.
static auto CheckIndent(DiagnosticEmitter& emitter, llvm::StringRef text,
                        llvm::StringRef content) -> Indent {
  // Find the leading horizontal whitespace on the final line of this literal.
  // Note that for an empty literal, this might not be inside the content.
  llvm::StringRef indent = ComputeIndentOfFinalLine(text);
  bool has_errors = false;

  // The last line is not permitted to contain any content after its
  // indentation.
  if (indent.end() != content.end()) {
    emitter.EmitError<ContentBeforeStringTerminator>();
    has_errors = true;
  }

  return {.indent = indent, .has_errors = has_errors};
}

// Expand a `\u{HHHHHH}` escape sequence into a sequence of UTF-8 code units.
static auto ExpandUnicodeEscapeSequence(DiagnosticEmitter& emitter,
                                        llvm::StringRef digits,
                                        std::string& result) -> bool {
  unsigned code_point;
  if (digits.getAsInteger(16, code_point) || code_point > 0x10FFFF) {
    emitter.EmitError<UnicodeEscapeTooLarge>();
    return false;
  }

  if (code_point >= 0xD800 && code_point < 0xE000) {
    emitter.EmitError<UnicodeEscapeSurrogate>();
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

static auto ExpandAndConsumeEscapeSequence(DiagnosticEmitter& emitter,
                                           llvm::StringRef& escape,
                                           std::string& result) -> bool {
  assert(!escape.empty() && "should have escaped closing delimiter");
  char first = escape.front();
  escape = escape.drop_front(1);

  switch (first) {
    case 't':
      result += '\t';
      return true;
    case 'n':
      result += '\n';
      return true;
    case 'r':
      result += '\r';
      return true;
    case '"':
      result += '"';
      return true;
    case '\'':
      result += '\'';
      return true;
    case '\\':
      result += '\\';
      return true;
    case '0':
      result += '\0';
      if (!escape.empty() && llvm::isDigit(escape.front())) {
        emitter.EmitError<DecimalEscapeSequence>();
        return false;
      }
      return true;
    case 'x':
      if (escape.size() >= 2 && isUpperHexDigit(escape[0]) && isUpperHexDigit(escape[1])) {
        result += static_cast<char>(llvm::hexFromNibbles(escape[0], escape[1]));
        escape = escape.drop_front(2);
        return true;
      }
      emitter.EmitError<HexadecimalEscapeMissingDigits>();
      break;
    case 'u': {
      if (!escape.empty() && escape.front() == '{') {
        const char *pos = escape.begin() + 1;
        while (pos != escape.end() && isUpperHexDigit(*pos)) {
          ++pos;
        }
        if (pos != escape.end() && pos != escape.begin() + 1 && *pos == '}') {
          llvm::StringRef digits(escape.begin() + 1,
                                 pos - (escape.begin() + 1));
          if (!ExpandUnicodeEscapeSequence(emitter, digits, result)) {
            break;
          }
          escape = escape.drop_front(digits.size() + 2);
          return true;
        }
      }
      emitter.EmitError<UnicodeEscapeMissingBracedDigits>();
      break;
    }
    default:
      emitter.EmitError<UnknownEscapeSequence>({.first = first});
      break;
  }

  // If we get here, we didn't recognize this escape sequence and have already
  // issued a diagnostic. For error recovery purposes, expand this escape
  // sequence to itself, dropping the introducer (for example, `\q` -> `q`).
  result += first;
  return false;
}

// Expand any escape sequences in the given string literal.
static auto ExpandEscapeSequencesAndRemoveIndent(DiagnosticEmitter& emitter,
                                                 llvm::StringRef contents,
                                                 int hash_level,
                                                 llvm::StringRef indent)
    -> StringLiteralToken::ExpandedValue {
  std::string result;
  result.reserve(contents.size());
  bool has_errors = false;

  llvm::SmallString<16> escape("\\");
  escape.resize(1 + hash_level, '#');

  // Process each line of the string literal.
  while (true) {
    // Every non-empty line (that contains anything other than horizontal
    // whitespace) is required to start with the string's indent. For error
    // recovery, remove all leading whitespace if the indent doesn't match.
    if (!contents.consume_front(indent)) {
      contents = contents.ltrim(HorizontalWhitespace);
      if (!contents.startswith("\n")) {
        emitter.EmitError<MismatchedIndentInString>();
        has_errors = true;
      }
    }

    // Process the contents of the line.
    while (true) {
      auto end_of_regular_text = contents.find_first_of("\n\\");
      result += contents.substr(0, end_of_regular_text);
      contents = contents.substr(end_of_regular_text);

      if (contents.empty()) {
        return {.result = result, .has_errors = has_errors};
      }

      if (contents.consume_front("\n")) {
        // Trailing whitespace before a newline doesn't contribute to the string
        // literal value.
        while (!result.empty() && result.back() != '\n' && isSpace(result.back()))
          result.pop_back();
        result += '\n';
        // Move onto to the next line.
        break;
      }

      if (!contents.consume_front(escape)) {
        // This is not an escape sequence, just a raw `\`.
        result += contents.front();
        contents = contents.drop_front(1);
        continue;
      }

      if (contents.consume_front("\n")) {
        // An escaped ends the line without producing any content and without
        // trimming trailing whitespace.
        break;
      }

      // Handle this escape sequence.
      if (!ExpandAndConsumeEscapeSequence(emitter, contents, result)) {
        has_errors = true;
      }
    }
  }
}

auto StringLiteralToken::ComputeValue(DiagnosticEmitter& emitter) const
    -> ExpandedValue {
  auto indent = multi_line ? CheckIndent(emitter, text, content) : Indent();
  auto result = ExpandEscapeSequencesAndRemoveIndent(emitter, content,
                                                     hash_level, indent.indent);
  result.has_errors |= indent.has_errors;
  return result;
}

}  // namespace Carbon
