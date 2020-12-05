// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "lexer/tokenized_buffer.h"

#include <algorithm>
#include <cmath>
#include <string>

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

namespace Carbon {

static auto TakeLeadingIntegerLiteral(llvm::StringRef source_text)
    -> llvm::StringRef {
  return source_text.take_while([](char c) { return llvm::isDigit(c); });
}

struct UnmatchedClosing {
  static constexpr llvm::StringLiteral ShortName = "syntax-balanced-delimiters";
  static constexpr llvm::StringLiteral Message =
      "Closing symbol without a corresponding opening symbol.";

  struct Substitutions {};
  static auto Format(const Substitutions&) -> std::string {
    return Message.str();
  }
};

struct MismatchedClosing {
  static constexpr llvm::StringLiteral ShortName = "syntax-balanced-delimiters";
  static constexpr llvm::StringLiteral Message =
      "Closing symbol does not match most recent opening symbol.";

  struct Substitutions {};
  static auto Format(const Substitutions&) -> std::string {
    return Message.str();
  }
};

struct UnrecognizedCharacters {
  static constexpr llvm::StringLiteral ShortName =
      "syntax-unrecognized-characters";
  static constexpr llvm::StringLiteral Message =
      "Encountered unrecognized characters while parsing.";

  struct Substitutions {};
  static auto Format(const Substitutions&) -> std::string {
    return Message.str();
  }
};

// Implementation of the lexer logic itself.
//
// The design is that lexing can loop over the source buffer, consuming it into
// tokens by calling into this API. This class handles the state and breaks down
// the different lexing steps that may be used. It directly updates the provided
// tokenized buffer with the lexed tokens.
class TokenizedBuffer::Lexer {
  TokenizedBuffer& buffer;
  DiagnosticEmitter& emitter;

  Line current_line;
  LineInfo* current_line_info;

  int current_column = 0;
  bool set_indent = false;

  llvm::SmallVector<Token, 8> open_groups;

 public:
  Lexer(TokenizedBuffer& buffer, DiagnosticEmitter& emitter)
      : buffer(buffer),
        emitter(emitter),
        current_line(buffer.AddLine({0, 0, 0})),
        current_line_info(&buffer.GetLineInfo(current_line)) {}

  auto SkipWhitespace(llvm::StringRef& source_text) -> bool {
    while (!source_text.empty()) {
      // We only support line-oriented commenting and lex comments as-if they
      // were whitespace. Any comment must be the only non-whitespace on the
      // line.
      if (source_text.startswith("//") && !set_indent) {
        // Check if the comment has a special starting sequence of three slashes
        // followed by a space. This represents a documentation comment that is
        // preserved as a token in the buffer. When parsing, these comments will
        // only be accepted in specific parts of the grammar and will be
        // associated with the parsed constructs as structure documentation. All
        // other comments are simply treated as whitespace.
        if (source_text.startswith("///")) {
          current_line_info->indent = current_column;
          set_indent = true;
          buffer.AddToken({.kind = TokenKind::DocComment(),
                           .token_line = current_line,
                           .column = current_column});
        }
        while (!source_text.empty() && source_text.front() != '\n') {
          ++current_column;
          source_text = source_text.drop_front();
        }
        if (source_text.empty())
          break;
      }

      switch (source_text.front()) {
        default:
          // If we find a non-whitespace character without exhausting the
          // buffer, return true to continue lexing.
          return true;

        case '\n':
          // New lines are special in order to track line structure.
          current_line_info->length = current_column;
          // If this is the last character in the source, directly return here
          // to avoid creating an empty line.
          source_text = source_text.drop_front();
          if (source_text.empty())
            return false;

          // Otherwise, add a line and set up to continue lexing.
          current_line = buffer.AddLine(
              {current_line_info->start + current_column + 1, 0, 0});
          current_line_info = &buffer.GetLineInfo(current_line);
          current_column = 0;
          set_indent = false;
          continue;

        case ' ':
        case '\t':
          // Skip other forms of whitespace while tracking column.
          // FIXME: This obviously needs looooots more work to handle unicode
          // whitespace as well as special handling to allow better tokenization
          // of operators. This is just a stub to check that our column
          // management works.
          ++current_column;
          source_text = source_text.drop_front();
          continue;
      }
    }

    assert(source_text.empty() && "Cannot reach here w/o finishing the text!");
    // Update the line length as this is also the end of a line.
    current_line_info->length = current_column;
    return false;
  }

  auto LexIntegerLiteral(llvm::StringRef& source_text) -> bool {
    llvm::StringRef int_text = TakeLeadingIntegerLiteral(source_text);
    if (int_text.empty())
      return false;
    llvm::APInt int_value;
    if (int_text.getAsInteger(/*Radix=*/0, int_value))
      return false;

    int int_column = current_column;
    current_column += int_text.size();
    source_text = source_text.drop_front(int_text.size());

    if (!set_indent) {
      current_line_info->indent = int_column;
      set_indent = true;
    }
    auto token = buffer.AddToken({.kind = TokenKind::IntegerLiteral(),
                                  .token_line = current_line,
                                  .column = int_column});
    buffer.GetTokenInfo(token).literal_index = buffer.int_literals.size();
    buffer.int_literals.push_back(std::move(int_value));
    return true;
  }

  auto LexSymbolToken(llvm::StringRef& source_text) -> bool {
    TokenKind kind = llvm::StringSwitch<TokenKind>(source_text)
#define CARBON_SYMBOL_TOKEN(Name, Spelling) \
  .StartsWith(Spelling, TokenKind::Name())
#include "lexer/token_registry.def"
                         .Default(TokenKind::Error());
    if (kind == TokenKind::Error())
      return false;

    if (!set_indent) {
      current_line_info->indent = current_column;
      set_indent = true;
    }

    CloseInvalidOpenGroups(kind);

    Token token = buffer.AddToken(
        {.kind = kind, .token_line = current_line, .column = current_column});
    current_column += kind.GetFixedSpelling().size();
    source_text = source_text.drop_front(kind.GetFixedSpelling().size());

    // Opening symbols just need to be pushed onto our queue of opening groups.
    if (kind.IsOpeningSymbol()) {
      open_groups.push_back(token);
      return true;
    }

    // Only closing symbols need further special handling.
    if (!kind.IsClosingSymbol())
      return true;

    TokenInfo& closing_token_info = buffer.GetTokenInfo(token);

    // Check that there is a matching opening symbol before we consume this as
    // a closing symbol.
    if (open_groups.empty()) {
      closing_token_info.kind = TokenKind::Error();
      closing_token_info.error_length = kind.GetFixedSpelling().size();
      buffer.has_errors = true;

      emitter.EmitError<UnmatchedClosing>(
          [](UnmatchedClosing::Substitutions&) {});
      // Note that this still returns true as we do consume a symbol.
      return true;
    }

    // Finally can handle a normal closing symbol.
    Token opening_token = open_groups.pop_back_val();
    TokenInfo& opening_token_info = buffer.GetTokenInfo(opening_token);
    opening_token_info.closing_token = token;
    closing_token_info.opening_token = opening_token;
    return true;
  }

  // Closes all open groups that cannot remain open across the symbol `K`.
  // Users may pass `Error` to close all open groups.
  auto CloseInvalidOpenGroups(TokenKind kind) -> void {
    if (!kind.IsClosingSymbol() && kind != TokenKind::Error())
      return;

    while (!open_groups.empty()) {
      Token opening_token = open_groups.back();
      TokenKind opening_kind = buffer.GetTokenInfo(opening_token).kind;
      if (kind == opening_kind.GetClosingSymbol())
        return;

      open_groups.pop_back();
      buffer.has_errors = true;
      emitter.EmitError<MismatchedClosing>(
          [](MismatchedClosing::Substitutions&) {});

      // TODO: do a smarter backwards scan for where to put the closing
      // token.
      Token closing_token =
          buffer.AddToken({.kind = opening_kind.GetClosingSymbol(),
                           .is_recovery = true,
                           .token_line = current_line,
                           .column = current_column});
      TokenInfo& opening_token_info = buffer.GetTokenInfo(opening_token);
      TokenInfo& closing_token_info = buffer.GetTokenInfo(closing_token);
      opening_token_info.closing_token = closing_token;
      closing_token_info.opening_token = opening_token;
    }
  }

  auto GetOrCreateIdentifier(llvm::StringRef text) -> Identifier {
    auto insert_result = buffer.identifier_map.insert(
        {text, Identifier(buffer.identifier_infos.size())});
    if (insert_result.second)
      buffer.identifier_infos.push_back({text});
    return insert_result.first->second;
  }

  auto LexKeywordOrIdentifier(llvm::StringRef& source_text) -> bool {
    if (!llvm::isAlpha(source_text.front()) && source_text.front() != '_')
      return false;

    if (!set_indent) {
      current_line_info->indent = current_column;
      set_indent = true;
    }

    // Take the valid characters off the front of the source buffer.
    llvm::StringRef identifier_text = source_text.take_while(
        [](char c) { return llvm::isAlnum(c) || c == '_'; });
    assert(!identifier_text.empty() && "Must have at least one character!");
    int identifier_column = current_column;
    current_column += identifier_text.size();
    source_text = source_text.drop_front(identifier_text.size());

    // Check if the text matches a keyword token, and if so use that.
    TokenKind kind = llvm::StringSwitch<TokenKind>(identifier_text)
#define CARBON_KEYWORD_TOKEN(Name, Spelling) .Case(Spelling, TokenKind::Name())
#include "lexer/token_registry.def"
                         .Default(TokenKind::Error());
    if (kind != TokenKind::Error()) {
      buffer.AddToken({.kind = kind,
                       .token_line = current_line,
                       .column = identifier_column});
      return true;
    }

    // Otherwise we have a generic identifier.
    buffer.AddToken({.kind = TokenKind::Identifier(),
                     .token_line = current_line,
                     .column = identifier_column,
                     .id = GetOrCreateIdentifier(identifier_text)});
    return true;
  }

  auto LexError(llvm::StringRef& source_text) -> void {
    llvm::StringRef error_text = source_text.take_while([](char c) {
      if (llvm::isAlnum(c))
        return false;
      switch (c) {
        case '_':
          return false;
        case '\t':
          return false;
        case '\n':
          return false;
      }
      return llvm::StringSwitch<bool>(llvm::StringRef(&c, 1))
#define CARBON_SYMBOL_TOKEN(Name, Spelling) .StartsWith(Spelling, false)
#include "lexer/token_registry.def"
          .Default(true);
    });
    if (error_text.empty()) {
      // TODO: Reimplement this to use the lexer properly. In the meantime,
      // guarantee that we eat at least one byte.
      error_text = source_text.take_front(1);
    }

    // Longer errors get to be two tokens.
    error_text = error_text.substr(0, std::numeric_limits<int32_t>::max());
    auto token = buffer.AddToken(
        {.kind = TokenKind::Error(),
         .token_line = current_line,
         .column = current_column,
         .error_length = static_cast<int32_t>(error_text.size())});
    // TODO: #19 - Need to convert to the diagnostics library.
    llvm::errs() << "ERROR: Line " << buffer.GetLineNumber(token) << ", Column "
                 << buffer.GetColumnNumber(token)
                 << ": Unrecognized characters!\n";

    current_column += error_text.size();
    source_text = source_text.drop_front(error_text.size());
    buffer.has_errors = true;
  }
};

auto TokenizedBuffer::Lex(SourceBuffer& source, DiagnosticEmitter& emitter)
    -> TokenizedBuffer {
  TokenizedBuffer buffer(source);
  Lexer lexer(buffer, emitter);

  llvm::StringRef source_text = source.Text();
  while (lexer.SkipWhitespace(source_text)) {
    // Each time we find non-whitespace characters, try each kind of token we
    // support lexing, from simplest to most complex.
    if (lexer.LexSymbolToken(source_text))
      continue;
    if (lexer.LexKeywordOrIdentifier(source_text))
      continue;
    if (lexer.LexIntegerLiteral(source_text))
      continue;
    lexer.LexError(source_text);
  }

  lexer.CloseInvalidOpenGroups(TokenKind::Error());
  return buffer;
}

auto TokenizedBuffer::GetKind(Token token) const -> TokenKind {
  return GetTokenInfo(token).kind;
}

auto TokenizedBuffer::GetLine(Token token) const -> Line {
  return GetTokenInfo(token).token_line;
}

auto TokenizedBuffer::GetLineNumber(Token token) const -> int {
  return GetLineNumber(GetLine(token));
}

auto TokenizedBuffer::GetColumnNumber(Token token) const -> int {
  return GetTokenInfo(token).column + 1;
}

auto TokenizedBuffer::GetTokenText(Token token) const -> llvm::StringRef {
  auto& token_info = GetTokenInfo(token);
  llvm::StringRef fixed_spelling = token_info.kind.GetFixedSpelling();
  if (!fixed_spelling.empty())
    return fixed_spelling;

  if (token_info.kind == TokenKind::Error()) {
    auto& line_info = GetLineInfo(token_info.token_line);
    int64_t token_start = line_info.start + token_info.column;
    return source->Text().substr(token_start, token_info.error_length);
  }

  // Documentation comment tokens refer back to the source text.
  if (token_info.kind == TokenKind::DocComment()) {
    auto& line_info = GetLineInfo(token_info.token_line);
    int64_t token_start = line_info.start + token_info.column;
    int64_t token_stop = line_info.start + line_info.length;
    return source->Text().slice(token_start, token_stop);
  }

  // Refer back to the source text to preserve oddities like radix or leading
  // 0's the author had.
  if (token_info.kind == TokenKind::IntegerLiteral()) {
    auto& line_info = GetLineInfo(token_info.token_line);
    int64_t token_start = line_info.start + token_info.column;
    return TakeLeadingIntegerLiteral(source->Text().substr(token_start));
  }

  assert(token_info.kind == TokenKind::Identifier() &&
         "Only identifiers have stored text!");
  return GetIdentifierText(token_info.id);
}

auto TokenizedBuffer::GetIdentifier(Token token) const -> Identifier {
  auto& token_info = GetTokenInfo(token);
  assert(token_info.kind == TokenKind::Identifier() &&
         "The token must be an identifier!");
  return token_info.id;
}

auto TokenizedBuffer::GetIntegerLiteral(Token token) const -> llvm::APInt {
  auto& token_info = GetTokenInfo(token);
  assert(token_info.kind == TokenKind::IntegerLiteral() &&
         "The token must be an integer literal!");
  return int_literals[token_info.literal_index];
}

auto TokenizedBuffer::GetMatchedClosingToken(Token opening_token) const
    -> Token {
  auto& opening_token_info = GetTokenInfo(opening_token);
  assert(opening_token_info.kind.IsOpeningSymbol() &&
         "The token must be an opening group symbol!");
  return opening_token_info.closing_token;
}

auto TokenizedBuffer::GetMatchedOpeningToken(Token closing_token) const
    -> Token {
  auto& closing_token_info = GetTokenInfo(closing_token);
  assert(closing_token_info.kind.IsClosingSymbol() &&
         "The token must be an closing group symbol!");
  return closing_token_info.opening_token;
}

auto TokenizedBuffer::IsRecoveryToken(Token token) const -> bool {
  return GetTokenInfo(token).is_recovery;
}

auto TokenizedBuffer::GetLineNumber(Line line) const -> int {
  return line.index + 1;
}

auto TokenizedBuffer::GetIndentColumnNumber(Line line) const -> int {
  return GetLineInfo(line).indent + 1;
}

auto TokenizedBuffer::GetIdentifierText(Identifier identifier) const
    -> llvm::StringRef {
  return identifier_infos[identifier.index].text;
}

auto TokenizedBuffer::PrintWidths::Widen(const PrintWidths& widths) -> void {
  index = std::max(widths.index, index);
  kind = std::max(widths.kind, kind);
  column = std::max(widths.column, column);
  line = std::max(widths.line, line);
  indent = std::max(widths.indent, indent);
}

auto TokenizedBuffer::GetTokenPrintWidths(Token token) const -> PrintWidths {
  PrintWidths widths = {};
  // Compute the printed width of the various token information. When numbers
  // here are printed in decimal, the number of digits needed is is one more than
  // the log-base-10 of the value.
  widths.index = std::log10(token_infos.size()) + 1;
  widths.kind = GetKind(token).Name().size();
  widths.line = std::log10(GetLineNumber(token)) + 1;
  widths.column = std::log10(GetColumnNumber(token)) + 1;
  widths.indent = std::log10(GetIndentColumnNumber(GetLine(token))) + 1;
  return widths;
}

auto TokenizedBuffer::Print(llvm::raw_ostream& output_stream) const -> void {
  if (Tokens().begin() == Tokens().end())
    return;

  PrintWidths widths = {};
  widths.index = std::log10(token_infos.size()) + 1;
  for (Token token : Tokens())
    widths.Widen(GetTokenPrintWidths(token));

  for (Token token : Tokens()) {
    PrintToken(output_stream, token, widths);
    output_stream << "\n";
  }
}

auto TokenizedBuffer::PrintToken(llvm::raw_ostream& output_stream,
                                 Token token) const -> void {
  PrintToken(output_stream, token, {});
}

auto TokenizedBuffer::PrintToken(llvm::raw_ostream& output_stream, Token token,
                                 PrintWidths widths) const -> void {
  widths.Widen(GetTokenPrintWidths(token));
  int token_index = token.index;
  auto& token_info = GetTokenInfo(token);
  llvm::StringRef token_text = GetTokenText(token);

  // Output the main chunk using one format string. We have to do the
  // justification manually in order to use the dynamically computed widths
  // and get the quotes included.
  output_stream << llvm::formatv(
      "token: { index: {0}, kind: {1}, line: {2}, column: {3}, indent: {4}, "
      "spelling: '{5}'",
      llvm::format_decimal(token_index, widths.index),
      llvm::right_justify(
          (llvm::Twine("'") + token_info.kind.Name() + "'").str(),
          widths.kind + 2),
      llvm::format_decimal(GetLineNumber(token_info.token_line), widths.line),
      llvm::format_decimal(GetColumnNumber(token), widths.column),
      llvm::format_decimal(GetIndentColumnNumber(token_info.token_line),
                           widths.indent),
      token_text);

  if (token_info.kind == TokenKind::Identifier())
    output_stream << ", identifier: " << GetIdentifier(token).index;
  else if (token_info.kind.IsOpeningSymbol())
    output_stream << ", closing_token: " << GetMatchedClosingToken(token).index;
  else if (token_info.kind.IsClosingSymbol())
    output_stream << ", opening_token: " << GetMatchedOpeningToken(token).index;

  if (token_info.is_recovery)
    output_stream << ", recovery: true";

  output_stream << " }";
}

auto TokenizedBuffer::GetLineInfo(Line line) -> LineInfo& {
  return line_infos[line.index];
}

auto TokenizedBuffer::GetLineInfo(Line line) const -> const LineInfo& {
  return line_infos[line.index];
}

auto TokenizedBuffer::AddLine(LineInfo info) -> Line {
  line_infos.push_back(info);
  return Line(line_infos.size() - 1);
}

auto TokenizedBuffer::GetTokenInfo(Token token) -> TokenInfo& {
  return token_infos[token.index];
}

auto TokenizedBuffer::GetTokenInfo(Token token) const -> const TokenInfo& {
  return token_infos[token.index];
}

auto TokenizedBuffer::AddToken(TokenInfo info) -> Token {
  token_infos.push_back(info);
  return Token(token_infos.size() - 1);
}

}  // namespace Carbon
