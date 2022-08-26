// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lexer/tokenized_buffer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <string>

#include "common/check.h"
#include "common/string_helpers.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "toolchain/lexer/character_set.h"
#include "toolchain/lexer/lex_helpers.h"
#include "toolchain/lexer/numeric_literal.h"
#include "toolchain/lexer/string_literal.h"

namespace Carbon {

// TODO: Move Overload and VariantMatch somewhere more central.

// Form an overload set from a list of functions. For example:
//
// ```
// auto overloaded = Overload{[] (int) {}, [] (float) {}};
// ```
template <typename... Fs>
struct Overload : Fs... {
  using Fs::operator()...;
};
template <typename... Fs>
Overload(Fs...) -> Overload<Fs...>;

// Pattern-match against the type of the value stored in the variant `V`. Each
// element of `fs` should be a function that takes one or more of the variant
// values in `V`.
template <typename V, typename... Fs>
auto VariantMatch(V&& v, Fs&&... fs) -> decltype(auto) {
  return std::visit(Overload{std::forward<Fs&&>(fs)...}, std::forward<V&&>(v));
}

// Implementation of the lexer logic itself.
//
// The design is that lexing can loop over the source buffer, consuming it into
// tokens by calling into this API. This class handles the state and breaks down
// the different lexing steps that may be used. It directly updates the provided
// tokenized buffer with the lexed tokens.
class TokenizedBuffer::Lexer {
 public:
  // Symbolic result of a lexing action. This indicates whether we successfully
  // lexed a token, or whether other lexing actions should be attempted.
  //
  // While it wraps a simple boolean state, its API both helps make the failures
  // more self documenting, and by consuming the actual token constructively
  // when one is produced, it helps ensure the correct result is returned.
  class LexResult {
   public:
    // Consumes (and discard) a valid token to construct a result
    // indicating a token has been produced. Relies on implicit conversions.
    // NOLINTNEXTLINE(google-explicit-constructor)
    LexResult(Token /*discarded_token*/) : LexResult(true) {}

    // Returns a result indicating no token was produced.
    static auto NoMatch() -> LexResult { return LexResult(false); }

    // Tests whether a token was produced by the lexing routine, and
    // the lexer can continue forming tokens.
    explicit operator bool() const { return formed_token_; }

   private:
    explicit LexResult(bool formed_token) : formed_token_(formed_token) {}

    bool formed_token_;
  };

  Lexer(TokenizedBuffer& buffer, DiagnosticConsumer& consumer)
      : buffer_(buffer),
        translator_(buffer, &current_column_),
        emitter_(translator_, consumer),
        token_translator_(buffer, &current_column_),
        token_emitter_(token_translator_, consumer),
        current_line_(buffer.AddLine({0, 0, 0})),
        current_line_info_(&buffer.GetLineInfo(current_line_)) {}

  // Perform the necessary bookkeeping to step past a newline at the current
  // line and column.
  auto HandleNewline() -> void {
    current_line_info_->length = current_column_;

    current_line_ = buffer_.AddLine(
        {current_line_info_->start + current_column_ + 1, 0, 0});
    current_line_info_ = &buffer_.GetLineInfo(current_line_);
    current_column_ = 0;
    set_indent_ = false;
  }

  auto NoteWhitespace() -> void {
    if (!buffer_.token_infos_.empty()) {
      buffer_.token_infos_.back().has_trailing_space = true;
    }
  }

  auto SkipWhitespace(llvm::StringRef& source_text) -> bool {
    const char* const whitespace_start = source_text.begin();

    while (!source_text.empty()) {
      // We only support line-oriented commenting and lex comments as-if they
      // were whitespace.
      if (source_text.startswith("//")) {
        // Any comment must be the only non-whitespace on the line.
        if (set_indent_) {
          CARBON_DIAGNOSTIC(TrailingComment, Error,
                            "Trailing comments are not permitted.");

          emitter_.Emit(source_text.begin(), TrailingComment);
        }
        // The introducer '//' must be followed by whitespace or EOF.
        if (source_text.size() > 2 && !IsSpace(source_text[2])) {
          CARBON_DIAGNOSTIC(NoWhitespaceAfterCommentIntroducer, Error,
                            "Whitespace is required after '//'.");
          emitter_.Emit(source_text.begin() + 2,
                        NoWhitespaceAfterCommentIntroducer);
        }
        while (!source_text.empty() && source_text.front() != '\n') {
          ++current_column_;
          source_text = source_text.drop_front();
        }
        if (source_text.empty()) {
          break;
        }
      }

      switch (source_text.front()) {
        default:
          // If we find a non-whitespace character without exhausting the
          // buffer, return true to continue lexing.
          CARBON_CHECK(!IsSpace(source_text.front()));
          if (whitespace_start != source_text.begin()) {
            NoteWhitespace();
          }
          return true;

        case '\n':
          // If this is the last character in the source, directly return here
          // to avoid creating an empty line.
          source_text = source_text.drop_front();
          if (source_text.empty()) {
            current_line_info_->length = current_column_;
            return false;
          }

          // Otherwise, add a line and set up to continue lexing.
          HandleNewline();
          continue;

        case ' ':
        case '\t':
          // Skip other forms of whitespace while tracking column.
          // TODO: This obviously needs looooots more work to handle unicode
          // whitespace as well as special handling to allow better tokenization
          // of operators. This is just a stub to check that our column
          // management works.
          ++current_column_;
          source_text = source_text.drop_front();
          continue;
      }
    }

    CARBON_CHECK(source_text.empty())
        << "Cannot reach here w/o finishing the text!";
    // Update the line length as this is also the end of a line.
    current_line_info_->length = current_column_;
    return false;
  }

  auto LexNumericLiteral(llvm::StringRef& source_text) -> LexResult {
    llvm::Optional<LexedNumericLiteral> literal =
        LexedNumericLiteral::Lex(source_text);
    if (!literal) {
      return LexResult::NoMatch();
    }

    int int_column = current_column_;
    int token_size = literal->text().size();
    current_column_ += token_size;
    source_text = source_text.drop_front(token_size);

    if (!set_indent_) {
      current_line_info_->indent = int_column;
      set_indent_ = true;
    }

    return VariantMatch(
        literal->ComputeValue(emitter_),
        [&](LexedNumericLiteral::IntegerValue&& value) {
          auto token = buffer_.AddToken({.kind = TokenKind::IntegerLiteral(),
                                         .token_line = current_line_,
                                         .column = int_column});
          buffer_.GetTokenInfo(token).literal_index =
              buffer_.literal_int_storage_.size();
          buffer_.literal_int_storage_.push_back(std::move(value.value));
          return token;
        },
        [&](LexedNumericLiteral::RealValue&& value) {
          auto token = buffer_.AddToken({.kind = TokenKind::RealLiteral(),
                                         .token_line = current_line_,
                                         .column = int_column});
          buffer_.GetTokenInfo(token).literal_index =
              buffer_.literal_int_storage_.size();
          buffer_.literal_int_storage_.push_back(std::move(value.mantissa));
          buffer_.literal_int_storage_.push_back(std::move(value.exponent));
          CARBON_CHECK(buffer_.GetRealLiteral(token).IsDecimal() ==
                       (value.radix == LexedNumericLiteral::Radix::Decimal));
          return token;
        },
        [&](LexedNumericLiteral::UnrecoverableError) {
          auto token = buffer_.AddToken({
              .kind = TokenKind::Error(),
              .token_line = current_line_,
              .column = int_column,
              .error_length = token_size,
          });
          return token;
        });
  }

  auto LexStringLiteral(llvm::StringRef& source_text) -> LexResult {
    llvm::Optional<LexedStringLiteral> literal =
        LexedStringLiteral::Lex(source_text);
    if (!literal) {
      return LexResult::NoMatch();
    }

    Line string_line = current_line_;
    int string_column = current_column_;
    int literal_size = literal->text().size();
    source_text = source_text.drop_front(literal_size);

    if (!set_indent_) {
      current_line_info_->indent = string_column;
      set_indent_ = true;
    }

    // Update line and column information.
    if (!literal->is_multi_line()) {
      current_column_ += literal_size;
    } else {
      for (char c : literal->text()) {
        if (c == '\n') {
          HandleNewline();
          // The indentation of all lines in a multi-line string literal is
          // that of the first line.
          current_line_info_->indent = string_column;
          set_indent_ = true;
        } else {
          ++current_column_;
        }
      }
    }

    if (literal->is_terminated()) {
      auto token =
          buffer_.AddToken({.kind = TokenKind::StringLiteral(),
                            .token_line = string_line,
                            .column = string_column,
                            .literal_index = static_cast<int32_t>(
                                buffer_.literal_string_storage_.size())});
      buffer_.literal_string_storage_.push_back(
          literal->ComputeValue(emitter_));
      return token;
    } else {
      CARBON_DIAGNOSTIC(UnterminatedString, Error,
                        "String is missing a terminator.");
      emitter_.Emit(literal->text().begin(), UnterminatedString);
      return buffer_.AddToken({.kind = TokenKind::Error(),
                               .token_line = string_line,
                               .column = string_column,
                               .error_length = literal_size});
    }
  }

  auto LexSymbolToken(llvm::StringRef& source_text) -> LexResult {
    TokenKind kind = llvm::StringSwitch<TokenKind>(source_text)
#define CARBON_SYMBOL_TOKEN(Name, Spelling) \
  .StartsWith(Spelling, TokenKind::Name())
#include "toolchain/lexer/token_registry.def"
                         .Default(TokenKind::Error());
    if (kind == TokenKind::Error()) {
      return LexResult::NoMatch();
    }

    if (!set_indent_) {
      current_line_info_->indent = current_column_;
      set_indent_ = true;
    }

    CloseInvalidOpenGroups(kind);

    const char* location = source_text.begin();
    Token token = buffer_.AddToken(
        {.kind = kind, .token_line = current_line_, .column = current_column_});
    current_column_ += kind.GetFixedSpelling().size();
    source_text = source_text.drop_front(kind.GetFixedSpelling().size());

    // Opening symbols just need to be pushed onto our queue of opening groups.
    if (kind.IsOpeningSymbol()) {
      open_groups_.push_back(token);
      return token;
    }

    // Only closing symbols need further special handling.
    if (!kind.IsClosingSymbol()) {
      return token;
    }

    TokenInfo& closing_token_info = buffer_.GetTokenInfo(token);

    // Check that there is a matching opening symbol before we consume this as
    // a closing symbol.
    if (open_groups_.empty()) {
      closing_token_info.kind = TokenKind::Error();
      closing_token_info.error_length = kind.GetFixedSpelling().size();

      CARBON_DIAGNOSTIC(
          UnmatchedClosing, Error,
          "Closing symbol without a corresponding opening symbol.");
      emitter_.Emit(location, UnmatchedClosing);
      // Note that this still returns true as we do consume a symbol.
      return token;
    }

    // Finally can handle a normal closing symbol.
    Token opening_token = open_groups_.pop_back_val();
    TokenInfo& opening_token_info = buffer_.GetTokenInfo(opening_token);
    opening_token_info.closing_token = token;
    closing_token_info.opening_token = opening_token;
    return token;
  }

  // Given a word that has already been lexed, determine whether it is a type
  // literal and if so form the corresponding token.
  auto LexWordAsTypeLiteralToken(llvm::StringRef word, int column)
      -> LexResult {
    if (word.size() < 2) {
      // Too short to form one of these tokens.
      return LexResult::NoMatch();
    }
    if (!('1' <= word[1] && word[1] <= '9')) {
      // Doesn't start with a valid initial digit.
      return LexResult::NoMatch();
    }

    llvm::Optional<TokenKind> kind;
    switch (word.front()) {
      case 'i':
        kind = TokenKind::IntegerTypeLiteral();
        break;
      case 'u':
        kind = TokenKind::UnsignedIntegerTypeLiteral();
        break;
      case 'f':
        kind = TokenKind::FloatingPointTypeLiteral();
        break;
      default:
        return LexResult::NoMatch();
    };

    llvm::StringRef suffix = word.substr(1);
    if (!CanLexInteger(emitter_, suffix)) {
      return buffer_.AddToken(
          {.kind = TokenKind::Error(),
           .token_line = current_line_,
           .column = column,
           .error_length = static_cast<int32_t>(word.size())});
    }
    llvm::APInt suffix_value;
    if (suffix.getAsInteger(10, suffix_value)) {
      return LexResult::NoMatch();
    }

    auto token = buffer_.AddToken(
        {.kind = *kind, .token_line = current_line_, .column = column});
    buffer_.GetTokenInfo(token).literal_index =
        buffer_.literal_int_storage_.size();
    buffer_.literal_int_storage_.push_back(std::move(suffix_value));
    return token;
  }

  // Closes all open groups that cannot remain open across the symbol `K`.
  // Users may pass `Error` to close all open groups.
  auto CloseInvalidOpenGroups(TokenKind kind) -> void {
    if (!kind.IsClosingSymbol() && kind != TokenKind::Error()) {
      return;
    }

    while (!open_groups_.empty()) {
      Token opening_token = open_groups_.back();
      TokenKind opening_kind = buffer_.GetTokenInfo(opening_token).kind;
      if (kind == opening_kind.GetClosingSymbol()) {
        return;
      }

      open_groups_.pop_back();
      CARBON_DIAGNOSTIC(
          MismatchedClosing, Error,
          "Closing symbol does not match most recent opening symbol.");
      token_emitter_.Emit(opening_token, MismatchedClosing);

      CARBON_CHECK(!buffer_.tokens().empty())
          << "Must have a prior opening token!";
      Token prev_token = buffer_.tokens().end()[-1];

      // TODO: do a smarter backwards scan for where to put the closing
      // token.
      Token closing_token = buffer_.AddToken(
          {.kind = opening_kind.GetClosingSymbol(),
           .has_trailing_space = buffer_.HasTrailingWhitespace(prev_token),
           .is_recovery = true,
           .token_line = current_line_,
           .column = current_column_});
      TokenInfo& opening_token_info = buffer_.GetTokenInfo(opening_token);
      TokenInfo& closing_token_info = buffer_.GetTokenInfo(closing_token);
      opening_token_info.closing_token = closing_token;
      closing_token_info.opening_token = opening_token;
    }
  }

  auto GetOrCreateIdentifier(llvm::StringRef text) -> Identifier {
    auto insert_result = buffer_.identifier_map_.insert(
        {text, Identifier(buffer_.identifier_infos_.size())});
    if (insert_result.second) {
      buffer_.identifier_infos_.push_back({text});
    }
    return insert_result.first->second;
  }

  auto LexKeywordOrIdentifier(llvm::StringRef& source_text) -> LexResult {
    if (!IsAlpha(source_text.front()) && source_text.front() != '_') {
      return LexResult::NoMatch();
    }

    if (!set_indent_) {
      current_line_info_->indent = current_column_;
      set_indent_ = true;
    }

    // Take the valid characters off the front of the source buffer.
    llvm::StringRef identifier_text =
        source_text.take_while([](char c) { return IsAlnum(c) || c == '_'; });
    CARBON_CHECK(!identifier_text.empty())
        << "Must have at least one character!";
    int identifier_column = current_column_;
    current_column_ += identifier_text.size();
    source_text = source_text.drop_front(identifier_text.size());

    // Check if the text is a type literal, and if so form such a literal.
    if (LexResult result =
            LexWordAsTypeLiteralToken(identifier_text, identifier_column)) {
      return result;
    }

    // Check if the text matches a keyword token, and if so use that.
    TokenKind kind = llvm::StringSwitch<TokenKind>(identifier_text)
#define CARBON_KEYWORD_TOKEN(Name, Spelling) .Case(Spelling, TokenKind::Name())
#include "toolchain/lexer/token_registry.def"
                         .Default(TokenKind::Error());
    if (kind != TokenKind::Error()) {
      return buffer_.AddToken({.kind = kind,
                               .token_line = current_line_,
                               .column = identifier_column});
    }

    // Otherwise we have a generic identifier.
    return buffer_.AddToken({.kind = TokenKind::Identifier(),
                             .token_line = current_line_,
                             .column = identifier_column,
                             .id = GetOrCreateIdentifier(identifier_text)});
  }

  auto LexError(llvm::StringRef& source_text) -> LexResult {
    llvm::StringRef error_text = source_text.take_while([](char c) {
      if (IsAlnum(c)) {
        return false;
      }
      switch (c) {
        case '_':
        case '\t':
        case '\n':
          return false;
      }
      return llvm::StringSwitch<bool>(llvm::StringRef(&c, 1))
#define CARBON_SYMBOL_TOKEN(Name, Spelling) .StartsWith(Spelling, false)
#include "toolchain/lexer/token_registry.def"
          .Default(true);
    });
    if (error_text.empty()) {
      // TODO: Reimplement this to use the lexer properly. In the meantime,
      // guarantee that we eat at least one byte.
      error_text = source_text.take_front(1);
    }

    auto token = buffer_.AddToken(
        {.kind = TokenKind::Error(),
         .token_line = current_line_,
         .column = current_column_,
         .error_length = static_cast<int32_t>(error_text.size())});
    CARBON_DIAGNOSTIC(UnrecognizedCharacters, Error,
                      "Encountered unrecognized characters while parsing.");
    emitter_.Emit(error_text.begin(), UnrecognizedCharacters);

    current_column_ += error_text.size();
    source_text = source_text.drop_front(error_text.size());
    return token;
  }

  auto AddEndOfFileToken() -> void {
    buffer_.AddToken({.kind = TokenKind::EndOfFile(),
                      .token_line = current_line_,
                      .column = current_column_});
  }

 private:
  TokenizedBuffer& buffer_;

  SourceBufferLocationTranslator translator_;
  LexerDiagnosticEmitter emitter_;

  TokenLocationTranslator token_translator_;
  TokenDiagnosticEmitter token_emitter_;

  Line current_line_;
  LineInfo* current_line_info_;

  int current_column_ = 0;
  bool set_indent_ = false;

  llvm::SmallVector<Token, 8> open_groups_;
};

auto TokenizedBuffer::Lex(SourceBuffer& source, DiagnosticConsumer& consumer)
    -> TokenizedBuffer {
  TokenizedBuffer buffer(source);
  ErrorTrackingDiagnosticConsumer error_tracking_consumer(consumer);
  Lexer lexer(buffer, error_tracking_consumer);

  llvm::StringRef source_text = source.text();
  while (lexer.SkipWhitespace(source_text)) {
    // Each time we find non-whitespace characters, try each kind of token we
    // support lexing, from simplest to most complex.
    Lexer::LexResult result = lexer.LexSymbolToken(source_text);
    if (!result) {
      result = lexer.LexKeywordOrIdentifier(source_text);
    }
    if (!result) {
      result = lexer.LexNumericLiteral(source_text);
    }
    if (!result) {
      result = lexer.LexStringLiteral(source_text);
    }
    if (!result) {
      result = lexer.LexError(source_text);
    }
    CARBON_CHECK(result) << "No token was lexed.";
  }

  // The end-of-file token is always considered to be whitespace.
  lexer.NoteWhitespace();

  lexer.CloseInvalidOpenGroups(TokenKind::Error());
  lexer.AddEndOfFileToken();

  if (error_tracking_consumer.seen_error()) {
    buffer.has_errors_ = true;
  }

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
  if (!fixed_spelling.empty()) {
    return fixed_spelling;
  }

  if (token_info.kind == TokenKind::Error()) {
    auto& line_info = GetLineInfo(token_info.token_line);
    int64_t token_start = line_info.start + token_info.column;
    return source_->text().substr(token_start, token_info.error_length);
  }

  // Refer back to the source text to preserve oddities like radix or digit
  // separators the author included.
  if (token_info.kind == TokenKind::IntegerLiteral() ||
      token_info.kind == TokenKind::RealLiteral()) {
    auto& line_info = GetLineInfo(token_info.token_line);
    int64_t token_start = line_info.start + token_info.column;
    llvm::Optional<LexedNumericLiteral> relexed_token =
        LexedNumericLiteral::Lex(source_->text().substr(token_start));
    CARBON_CHECK(relexed_token) << "Could not reform numeric literal token.";
    return relexed_token->text();
  }

  // Refer back to the source text to find the original spelling, including
  // escape sequences etc.
  if (token_info.kind == TokenKind::StringLiteral()) {
    auto& line_info = GetLineInfo(token_info.token_line);
    int64_t token_start = line_info.start + token_info.column;
    llvm::Optional<LexedStringLiteral> relexed_token =
        LexedStringLiteral::Lex(source_->text().substr(token_start));
    CARBON_CHECK(relexed_token) << "Could not reform string literal token.";
    return relexed_token->text();
  }

  // Refer back to the source text to avoid needing to reconstruct the
  // spelling from the size.
  if (token_info.kind.IsSizedTypeLiteral()) {
    auto& line_info = GetLineInfo(token_info.token_line);
    int64_t token_start = line_info.start + token_info.column;
    llvm::StringRef suffix =
        source_->text().substr(token_start + 1).take_while(IsDecimalDigit);
    return llvm::StringRef(suffix.data() - 1, suffix.size() + 1);
  }

  if (token_info.kind == TokenKind::EndOfFile()) {
    return llvm::StringRef();
  }

  CARBON_CHECK(token_info.kind == TokenKind::Identifier())
      << "Only identifiers have stored text!";
  return GetIdentifierText(token_info.id);
}

auto TokenizedBuffer::GetIdentifier(Token token) const -> Identifier {
  auto& token_info = GetTokenInfo(token);
  CARBON_CHECK(token_info.kind == TokenKind::Identifier())
      << "The token must be an identifier!";
  return token_info.id;
}

auto TokenizedBuffer::GetIntegerLiteral(Token token) const
    -> const llvm::APInt& {
  auto& token_info = GetTokenInfo(token);
  CARBON_CHECK(token_info.kind == TokenKind::IntegerLiteral())
      << "The token must be an integer literal!";
  return literal_int_storage_[token_info.literal_index];
}

auto TokenizedBuffer::GetRealLiteral(Token token) const -> RealLiteralValue {
  auto& token_info = GetTokenInfo(token);
  CARBON_CHECK(token_info.kind == TokenKind::RealLiteral())
      << "The token must be a real literal!";

  // Note that every real literal is at least three characters long, so we can
  // safely look at the second character to determine whether we have a
  // decimal or hexadecimal literal.
  auto& line_info = GetLineInfo(token_info.token_line);
  int64_t token_start = line_info.start + token_info.column;
  char second_char = source_->text()[token_start + 1];
  bool is_decimal = second_char != 'x' && second_char != 'b';

  return RealLiteralValue(this, token_info.literal_index, is_decimal);
}

auto TokenizedBuffer::GetStringLiteral(Token token) const -> llvm::StringRef {
  auto& token_info = GetTokenInfo(token);
  CARBON_CHECK(token_info.kind == TokenKind::StringLiteral())
      << "The token must be a string literal!";
  return literal_string_storage_[token_info.literal_index];
}

auto TokenizedBuffer::GetTypeLiteralSize(Token token) const
    -> const llvm::APInt& {
  auto& token_info = GetTokenInfo(token);
  CARBON_CHECK(token_info.kind.IsSizedTypeLiteral())
      << "The token must be a sized type literal!";
  return literal_int_storage_[token_info.literal_index];
}

auto TokenizedBuffer::GetMatchedClosingToken(Token opening_token) const
    -> Token {
  auto& opening_token_info = GetTokenInfo(opening_token);
  CARBON_CHECK(opening_token_info.kind.IsOpeningSymbol())
      << "The token must be an opening group symbol!";
  return opening_token_info.closing_token;
}

auto TokenizedBuffer::GetMatchedOpeningToken(Token closing_token) const
    -> Token {
  auto& closing_token_info = GetTokenInfo(closing_token);
  CARBON_CHECK(closing_token_info.kind.IsClosingSymbol())
      << "The token must be an closing group symbol!";
  return closing_token_info.opening_token;
}

auto TokenizedBuffer::HasLeadingWhitespace(Token token) const -> bool {
  auto it = TokenIterator(token);
  return it == tokens().begin() || GetTokenInfo(*(it - 1)).has_trailing_space;
}

auto TokenizedBuffer::HasTrailingWhitespace(Token token) const -> bool {
  return GetTokenInfo(token).has_trailing_space;
}

auto TokenizedBuffer::IsRecoveryToken(Token token) const -> bool {
  return GetTokenInfo(token).is_recovery;
}

auto TokenizedBuffer::GetLineNumber(Line line) const -> int {
  return line.index_ + 1;
}

auto TokenizedBuffer::GetIndentColumnNumber(Line line) const -> int {
  return GetLineInfo(line).indent + 1;
}

auto TokenizedBuffer::GetIdentifierText(Identifier identifier) const
    -> llvm::StringRef {
  return identifier_infos_[identifier.index_].text;
}

auto TokenizedBuffer::PrintWidths::Widen(const PrintWidths& widths) -> void {
  index = std::max(widths.index, index);
  kind = std::max(widths.kind, kind);
  column = std::max(widths.column, column);
  line = std::max(widths.line, line);
  indent = std::max(widths.indent, indent);
}

// Compute the printed width of a number. When numbers are printed in decimal,
// the number of digits needed is is one more than the log-base-10 of the
// value. We handle a value of `zero` explicitly.
//
// This routine requires its argument to be *non-negative*.
static auto ComputeDecimalPrintedWidth(int number) -> int {
  CARBON_CHECK(number >= 0) << "Negative numbers are not supported.";
  if (number == 0) {
    return 1;
  }

  return static_cast<int>(std::log10(number)) + 1;
}

auto TokenizedBuffer::GetTokenPrintWidths(Token token) const -> PrintWidths {
  PrintWidths widths = {};
  widths.index = ComputeDecimalPrintedWidth(token_infos_.size());
  widths.kind = GetKind(token).Name().size();
  widths.line = ComputeDecimalPrintedWidth(GetLineNumber(token));
  widths.column = ComputeDecimalPrintedWidth(GetColumnNumber(token));
  widths.indent =
      ComputeDecimalPrintedWidth(GetIndentColumnNumber(GetLine(token)));
  return widths;
}

auto TokenizedBuffer::Print(llvm::raw_ostream& output_stream) const -> void {
  if (tokens().begin() == tokens().end()) {
    return;
  }

  PrintWidths widths = {};
  widths.index = ComputeDecimalPrintedWidth((token_infos_.size()));
  for (Token token : tokens()) {
    widths.Widen(GetTokenPrintWidths(token));
  }

  for (Token token : tokens()) {
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
  int token_index = token.index_;
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

  switch (token_info.kind) {
    case TokenKind::Identifier():
      output_stream << ", identifier: " << GetIdentifier(token).index_;
      break;
    case TokenKind::IntegerLiteral():
      output_stream << ", value: `";
      GetIntegerLiteral(token).print(output_stream, /*isSigned=*/false);
      output_stream << "`";
      break;
    case TokenKind::RealLiteral():
      output_stream << ", value: `" << GetRealLiteral(token) << "`";
      break;
    case TokenKind::StringLiteral():
      output_stream << ", value: `" << GetStringLiteral(token) << "`";
      break;
    default:
      if (token_info.kind.IsOpeningSymbol()) {
        output_stream << ", closing_token: "
                      << GetMatchedClosingToken(token).index_;
      } else if (token_info.kind.IsClosingSymbol()) {
        output_stream << ", opening_token: "
                      << GetMatchedOpeningToken(token).index_;
      }
      break;
  }

  if (token_info.has_trailing_space) {
    output_stream << ", has_trailing_space: true";
  }
  if (token_info.is_recovery) {
    output_stream << ", recovery: true";
  }

  output_stream << " }";
}

auto TokenizedBuffer::GetLineInfo(Line line) -> LineInfo& {
  return line_infos_[line.index_];
}

auto TokenizedBuffer::GetLineInfo(Line line) const -> const LineInfo& {
  return line_infos_[line.index_];
}

auto TokenizedBuffer::AddLine(LineInfo info) -> Line {
  line_infos_.push_back(info);
  return Line(static_cast<int>(line_infos_.size()) - 1);
}

auto TokenizedBuffer::GetTokenInfo(Token token) -> TokenInfo& {
  return token_infos_[token.index_];
}

auto TokenizedBuffer::GetTokenInfo(Token token) const -> const TokenInfo& {
  return token_infos_[token.index_];
}

auto TokenizedBuffer::AddToken(TokenInfo info) -> Token {
  token_infos_.push_back(info);
  return Token(static_cast<int>(token_infos_.size()) - 1);
}

auto TokenizedBuffer::TokenIterator::Print(llvm::raw_ostream& output) const
    -> void {
  output << token_.index_;
}

auto TokenizedBuffer::SourceBufferLocationTranslator::GetLocation(
    const char* loc) -> DiagnosticLocation {
  CARBON_CHECK(StringRefContainsPointer(buffer_->source_->text(), loc))
      << "location not within buffer";
  int64_t offset = loc - buffer_->source_->text().begin();

  // Find the first line starting after the given location. Note that we can't
  // inspect `line.length` here because it is not necessarily correct for the
  // final line during lexing (but will be correct later for the parse tree).
  auto line_it = std::partition_point(
      buffer_->line_infos_.begin(), buffer_->line_infos_.end(),
      [offset](const LineInfo& line) { return line.start <= offset; });
  bool incomplete_line_info = last_line_lexed_to_column_ != nullptr &&
                              line_it == buffer_->line_infos_.end();

  // Step back one line to find the line containing the given position.
  CARBON_CHECK(line_it != buffer_->line_infos_.begin())
      << "location precedes the start of the first line";
  --line_it;
  int line_number = line_it - buffer_->line_infos_.begin();
  int column_number = offset - line_it->start;

  // We might still be lexing the last line. If so, check to see if there are
  // any newline characters between the position we've finished lexing up to
  // and the given location.
  if (incomplete_line_info && column_number > *last_line_lexed_to_column_) {
    column_number = *last_line_lexed_to_column_;
    for (int64_t i = line_it->start + *last_line_lexed_to_column_; i != offset;
         ++i) {
      if (buffer_->source_->text()[i] == '\n') {
        ++line_number;
        column_number = 0;
      } else {
        ++column_number;
      }
    }
  }

  return {.file_name = buffer_->source_->filename().str(),
          .line_number = line_number + 1,
          .column_number = column_number + 1};
}

auto TokenizedBuffer::TokenLocationTranslator::GetLocation(Token token)
    -> DiagnosticLocation {
  // Map the token location into a position within the source buffer.
  auto& token_info = buffer_->GetTokenInfo(token);
  auto& line_info = buffer_->GetLineInfo(token_info.token_line);
  const char* token_start =
      buffer_->source_->text().begin() + line_info.start + token_info.column;

  // Find the corresponding file location.
  // TODO: Should we somehow indicate in the diagnostic location if this token
  // is a recovery token that doesn't correspond to the original source?
  return SourceBufferLocationTranslator(*buffer_, last_line_lexed_to_column_)
      .GetLocation(token_start);
}

}  // namespace Carbon
