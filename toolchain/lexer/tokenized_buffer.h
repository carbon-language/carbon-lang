// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LEXER_TOKENIZED_BUFFER_H_
#define CARBON_TOOLCHAIN_LEXER_TOKENIZED_BUFFER_H_

#include <cstdint>
#include <iterator>
#include <optional>

#include "common/ostream.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/raw_ostream.h"
#include "toolchain/common/index_base.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/lexer/token_kind.h"
#include "toolchain/source/source_buffer.h"

namespace Carbon {

class TokenizedBuffer;

// A buffer of tokenized Carbon source code.
//
// This is constructed by lexing the source code text into a series of tokens.
// The buffer provides lightweight handles to tokens and other lexed entities,
// as well as iterations to walk the sequence of tokens found in the buffer.
//
// Lexing errors result in a potentially incomplete sequence of tokens and
// `HasError` returning true.
class TokenizedBuffer {
 public:
  // A lightweight handle to a lexed token in a `TokenizedBuffer`.
  //
  // `Token` objects are designed to be passed by value, not reference or
  // pointer. They are also designed to be small and efficient to store in data
  // structures.
  //
  // `Token` objects from the same `TokenizedBuffer` can be compared with each
  // other, both for being the same token within the buffer, and to establish
  // relative position within the token stream that has been lexed out of the
  // buffer. `Token` objects from different `TokenizedBuffer`s cannot be
  // meaningfully compared.
  //
  // All other APIs to query a `Token` are on the `TokenizedBuffer`.
  struct Token : public ComparableIndexBase {
    using ComparableIndexBase::ComparableIndexBase;
  };

  // A lightweight handle to a lexed line in a `TokenizedBuffer`.
  //
  // `Line` objects are designed to be passed by value, not reference or
  // pointer. They are also designed to be small and efficient to store in data
  // structures.
  //
  // Each `Line` object refers to a specific line in the source code that was
  // lexed. They can be compared directly to establish that they refer to the
  // same line or the relative position of different lines within the source.
  //
  // All other APIs to query a `Line` are on the `TokenizedBuffer`.
  struct Line : public ComparableIndexBase {
    using ComparableIndexBase::ComparableIndexBase;
  };

  // A lightweight handle to a lexed identifier in a `TokenizedBuffer`.
  //
  // `Identifier` objects are designed to be passed by value, not reference or
  // pointer. They are also designed to be small and efficient to store in data
  // structures.
  //
  // Each identifier lexed is canonicalized to a single entry in the identifier
  // table. `Identifier` objects will compare equal if they refer to the same
  // identifier spelling. Where the identifier was written is not preserved.
  //
  // All other APIs to query a `Identifier` are on the `TokenizedBuffer`.
  struct Identifier : public IndexBase {
    using IndexBase::IndexBase;
  };

  // Random-access iterator over tokens within the buffer.
  class TokenIterator
      : public llvm::iterator_facade_base<
            TokenIterator, std::random_access_iterator_tag, const Token, int> {
   public:
    TokenIterator() = default;

    explicit TokenIterator(Token token) : token_(token) {}

    auto operator==(const TokenIterator& rhs) const -> bool {
      return token_ == rhs.token_;
    }
    auto operator<(const TokenIterator& rhs) const -> bool {
      return token_ < rhs.token_;
    }

    auto operator*() const -> const Token& { return token_; }

    using iterator_facade_base::operator-;
    auto operator-(const TokenIterator& rhs) const -> int {
      return token_.index - rhs.token_.index;
    }

    auto operator+=(int n) -> TokenIterator& {
      token_.index += n;
      return *this;
    }
    auto operator-=(int n) -> TokenIterator& {
      token_.index -= n;
      return *this;
    }

    // Prints the raw token index.
    auto Print(llvm::raw_ostream& output) const -> void;

   private:
    friend class TokenizedBuffer;

    Token token_;
  };

  // The value of a real literal.
  //
  // This is either a dyadic fraction (mantissa * 2^exponent) or a decadic
  // fraction (mantissa * 10^exponent).
  //
  // The `TokenizedBuffer` must outlive any `RealLiteralValue`s referring to
  // its tokens.
  class RealLiteralValue {
   public:
    // The mantissa, represented as an unsigned integer.
    [[nodiscard]] auto Mantissa() const -> const llvm::APInt& {
      return buffer_->literal_int_storage_[literal_index_];
    }
    // The exponent, represented as a signed integer.
    [[nodiscard]] auto Exponent() const -> const llvm::APInt& {
      return buffer_->literal_int_storage_[literal_index_ + 1];
    }
    // If false, the value is mantissa * 2^exponent.
    // If true, the value is mantissa * 10^exponent.
    [[nodiscard]] auto IsDecimal() const -> bool { return is_decimal_; }

    auto Print(llvm::raw_ostream& output_stream) const -> void {
      output_stream << Mantissa() << "*" << (is_decimal_ ? "10" : "2") << "^"
                    << Exponent();
    }

   private:
    friend class TokenizedBuffer;

    RealLiteralValue(const TokenizedBuffer* buffer, int32_t literal_index,
                     bool is_decimal)
        : buffer_(buffer),
          literal_index_(literal_index),
          is_decimal_(is_decimal) {}

    const TokenizedBuffer* buffer_;
    int32_t literal_index_;
    bool is_decimal_;
  };

  // A diagnostic location translator that maps token locations into source
  // buffer locations.
  class TokenLocationTranslator : public DiagnosticLocationTranslator<Token> {
   public:
    explicit TokenLocationTranslator(const TokenizedBuffer* buffer,
                                     int* last_line_lexed_to_column)
        : buffer_(buffer),
          last_line_lexed_to_column_(last_line_lexed_to_column) {}

    // Map the given token into a diagnostic location.
    auto GetLocation(Token token) -> DiagnosticLocation override;

   private:
    const TokenizedBuffer* buffer_;
    // Passed to SourceBufferLocationTranslator.
    int* last_line_lexed_to_column_;
  };

  // Lexes a buffer of source code into a tokenized buffer.
  //
  // The provided source buffer must outlive any returned `TokenizedBuffer`
  // which will refer into the source.
  static auto Lex(SourceBuffer& source, DiagnosticConsumer& consumer)
      -> TokenizedBuffer;

  [[nodiscard]] auto GetKind(Token token) const -> TokenKind;
  [[nodiscard]] auto GetLine(Token token) const -> Line;

  // Returns the 1-based line number.
  [[nodiscard]] auto GetLineNumber(Token token) const -> int;

  // Returns the 1-based column number.
  [[nodiscard]] auto GetColumnNumber(Token token) const -> int;

  // Returns the source text lexed into this token.
  [[nodiscard]] auto GetTokenText(Token token) const -> llvm::StringRef;

  // Returns the identifier associated with this token. The token kind must be
  // an `Identifier`.
  [[nodiscard]] auto GetIdentifier(Token token) const -> Identifier;

  // Returns the value of an `IntegerLiteral()` token.
  [[nodiscard]] auto GetIntegerLiteral(Token token) const -> const llvm::APInt&;

  // Returns the value of an `RealLiteral()` token.
  [[nodiscard]] auto GetRealLiteral(Token token) const -> RealLiteralValue;

  // Returns the value of a `StringLiteral()` token.
  [[nodiscard]] auto GetStringLiteral(Token token) const -> llvm::StringRef;

  // Returns the size specified in a `*TypeLiteral()` token.
  [[nodiscard]] auto GetTypeLiteralSize(Token token) const
      -> const llvm::APInt&;

  // Returns the closing token matched with the given opening token.
  //
  // The given token must be an opening token kind.
  [[nodiscard]] auto GetMatchedClosingToken(Token opening_token) const -> Token;

  // Returns the opening token matched with the given closing token.
  //
  // The given token must be a closing token kind.
  [[nodiscard]] auto GetMatchedOpeningToken(Token closing_token) const -> Token;

  // Returns whether the given token has leading whitespace.
  [[nodiscard]] auto HasLeadingWhitespace(Token token) const -> bool;
  // Returns whether the given token has trailing whitespace.
  [[nodiscard]] auto HasTrailingWhitespace(Token token) const -> bool;

  // Returns whether the token was created as part of an error recovery effort.
  //
  // For example, a closing paren inserted to match an unmatched paren.
  [[nodiscard]] auto IsRecoveryToken(Token token) const -> bool;

  // Returns the 1-based line number.
  [[nodiscard]] auto GetLineNumber(Line line) const -> int;

  // Returns the 1-based indentation column number.
  [[nodiscard]] auto GetIndentColumnNumber(Line line) const -> int;

  // Returns the text for an identifier.
  [[nodiscard]] auto GetIdentifierText(Identifier id) const -> llvm::StringRef;

  // Prints a description of the tokenized stream to the provided `raw_ostream`.
  //
  // It prints one line of information for each token in the buffer, including
  // the kind of token, where it occurs within the source file, indentation for
  // the associated line, the spelling of the token in source, and any
  // additional information tracked such as which unique identifier it is or any
  // matched grouping token.
  //
  // Each line is formatted as a YAML record:
  //
  // clang-format off
  // ```
  // token: { index: 0, kind: 'Semi', line: 1, column: 1, indent: 1, spelling: ';' }
  // ```
  // clang-format on
  //
  // This can be parsed as YAML using tools like `python-yq` combined with `jq`
  // on the command line. The format is also reasonably amenable to other
  // line-oriented shell tools from `grep` to `awk`.
  auto Print(llvm::raw_ostream& output_stream) const -> void;

  // Prints a description of a single token.  See `Print` for details on the
  // format.
  auto PrintToken(llvm::raw_ostream& output_stream, Token token) const -> void;

  // Returns true if the buffer has errors that are detectable at lexing time.
  [[nodiscard]] auto has_errors() const -> bool { return has_errors_; }

  [[nodiscard]] auto tokens() const -> llvm::iterator_range<TokenIterator> {
    return llvm::make_range(TokenIterator(Token(0)),
                            TokenIterator(Token(token_infos_.size())));
  }

  [[nodiscard]] auto size() const -> int { return token_infos_.size(); }

 private:
  // Implementation detail struct implementing the actual lexer logic.
  class Lexer;
  friend Lexer;

  // A diagnostic location translator that maps token locations into source
  // buffer locations.
  class SourceBufferLocationTranslator
      : public DiagnosticLocationTranslator<const char*> {
   public:
    explicit SourceBufferLocationTranslator(const TokenizedBuffer* buffer,
                                            int* last_line_lexed_to_column)
        : buffer_(buffer),
          last_line_lexed_to_column_(last_line_lexed_to_column) {}

    // Map the given position within the source buffer into a diagnostic
    // location.
    auto GetLocation(const char* loc) -> DiagnosticLocation override;

   private:
    const TokenizedBuffer* buffer_;
    // The last lexed column, for determining whether the last line should be
    // checked for unlexed newlines. May be null after lexing is complete.
    int* last_line_lexed_to_column_;
  };

  // Specifies minimum widths to use when printing a token's fields via
  // `printToken`.
  struct PrintWidths {
    // Widens `this` to the maximum of `this` and `new_width` for each
    // dimension.
    auto Widen(const PrintWidths& widths) -> void;

    int index;
    int kind;
    int line;
    int column;
    int indent;
  };

  struct TokenInfo {
    TokenKind kind;

    // Whether the token has trailing whitespace.
    bool has_trailing_space = false;

    // Whether the token was injected artificially during error recovery.
    bool is_recovery = false;

    // Line on which the Token starts.
    Line token_line;

    // Zero-based byte offset of the token within its line.
    int32_t column;

    // We may have up to 32 bits of payload, based on the kind of token.
    union {
      static_assert(
          sizeof(Token) <= sizeof(int32_t),
          "Unable to pack token and identifier index into the same space!");

      Identifier id;
      int32_t literal_index;
      Token closing_token;
      Token opening_token;
      int32_t error_length;
    };
  };

  struct LineInfo {
    // Zero-based byte offset of the start of the line within the source buffer
    // provided.
    int64_t start;

    // The byte length of the line. Does not include the newline character (or a
    // nul-terminator or EOF).
    int32_t length;

    // The byte offset from the start of the line of the first non-whitespace
    // character.
    int32_t indent;
  };

  struct IdentifierInfo {
    llvm::StringRef text;
  };

  // The constructor is merely responsible for trivial initialization of
  // members. A working object of this type is built with the `lex` function
  // above so that its return can indicate if an error was encountered while
  // lexing.
  explicit TokenizedBuffer(SourceBuffer& source) : source_(&source) {}

  auto GetLineInfo(Line line) -> LineInfo&;
  [[nodiscard]] auto GetLineInfo(Line line) const -> const LineInfo&;
  auto AddLine(LineInfo info) -> Line;
  auto GetTokenInfo(Token token) -> TokenInfo&;
  [[nodiscard]] auto GetTokenInfo(Token token) const -> const TokenInfo&;
  auto AddToken(TokenInfo info) -> Token;
  [[nodiscard]] auto GetTokenPrintWidths(Token token) const -> PrintWidths;
  auto PrintToken(llvm::raw_ostream& output_stream, Token token,
                  PrintWidths widths) const -> void;

  SourceBuffer* source_;

  llvm::SmallVector<TokenInfo, 16> token_infos_;

  llvm::SmallVector<LineInfo, 16> line_infos_;

  llvm::SmallVector<IdentifierInfo, 16> identifier_infos_;

  // Storage for integers that form part of the value of a numeric or type
  // literal.
  llvm::SmallVector<llvm::APInt, 16> literal_int_storage_;

  llvm::SmallVector<std::string, 16> literal_string_storage_;

  llvm::DenseMap<llvm::StringRef, Identifier> identifier_map_;

  bool has_errors_ = false;
};

// A diagnostic emitter that uses positions within a source buffer's text as
// its source of location information.
using LexerDiagnosticEmitter = DiagnosticEmitter<const char*>;

// A diagnostic emitter that uses tokens as its source of location information.
using TokenDiagnosticEmitter = DiagnosticEmitter<TokenizedBuffer::Token>;

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_LEXER_TOKENIZED_BUFFER_H_
