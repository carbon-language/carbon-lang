// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LEXER_TOKENIZED_BUFFER_H_
#define LEXER_TOKENIZED_BUFFER_H_

#include <stdint.h>

#include <iterator>

#include "diagnostics/diagnostic_emitter.h"
#include "lexer/token_kind.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "source/source_buffer.h"

namespace Carbon {

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
  // buffer.
  //
  // All other APIs to query a `Token` are on the `TokenizedBuffer`.
  class Token {
   public:
    Token() = default;

    bool operator==(const Token& rhs) const { return index == rhs.index; }
    bool operator!=(const Token& rhs) const { return index != rhs.index; }
    bool operator<(const Token& rhs) const { return index < rhs.index; }
    bool operator<=(const Token& rhs) const { return index <= rhs.index; }
    bool operator>(const Token& rhs) const { return index > rhs.index; }
    bool operator>=(const Token& rhs) const { return index >= rhs.index; }

   private:
    friend class TokenizedBuffer;

    explicit Token(int index) : index(index) {}

    int32_t index;
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
  class Line {
   public:
    Line() = default;

    bool operator==(const Line& rhs) const { return index == rhs.index; }
    bool operator!=(const Line& rhs) const { return index != rhs.index; }
    bool operator<(const Line& rhs) const { return index < rhs.index; }
    bool operator<=(const Line& rhs) const { return index <= rhs.index; }
    bool operator>(const Line& rhs) const { return index > rhs.index; }
    bool operator>=(const Line& rhs) const { return index >= rhs.index; }

   private:
    friend class TokenizedBuffer;

    explicit Line(int index) : index(index) {}

    int32_t index;
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
  class Identifier {
   public:
    Identifier() = default;

    // Most normal APIs are provided by the `TokenizedBuffer`, we just support
    // basic comparison operations.
    bool operator==(const Identifier& rhs) const { return index == rhs.index; }
    bool operator!=(const Identifier& rhs) const { return index != rhs.index; }

   private:
    friend class TokenizedBuffer;

    explicit Identifier(int index) : index(index) {}

    int32_t index;
  };

  // Random-access iterator over tokens within the buffer.
  class TokenIterator
      : public llvm::iterator_facade_base<
            TokenIterator, std::random_access_iterator_tag, Token, int> {
   public:
    TokenIterator() = default;

    explicit TokenIterator(Token token) : token(token) {}

    bool operator==(const TokenIterator& rhs) const {
      return token == rhs.token;
    }
    bool operator<(const TokenIterator& rhs) const { return token < rhs.token; }

    const Token& operator*() const { return token; }
    Token& operator*() { return token; }

    int operator-(const TokenIterator& rhs) const {
      return token.index - rhs.token.index;
    }

    TokenIterator& operator+=(int n) {
      token.index += n;
      return *this;
    }
    TokenIterator& operator-=(int n) {
      token.index -= n;
      return *this;
    }

   private:
    friend class TokenizedBuffer;

    Token token;
  };

  // Lexes a buffer of source code into a tokenized buffer.
  //
  // The provided source buffer must outlive any returned `TokenizedBuffer`
  // which will refer into the source.
  //
  // FIXME: Need to pass in some diagnostic machinery to report the details of
  // the error! Right now it prints to stderr.
  static TokenizedBuffer Lex(SourceBuffer& source, DiagnosticEmitter& emitter);

  // Returns true if the buffer has errors that are detectable at lexing time.
  auto HasErrors() const -> bool { return has_errors; }

  llvm::iterator_range<TokenIterator> Tokens() const {
    return llvm::make_range(TokenIterator(Token(0)),
                            TokenIterator(Token(token_infos.size())));
  }

  auto Size() const -> int { return token_infos.size(); }

  auto GetKind(Token token) const -> TokenKind;
  auto GetLine(Token token) const -> Line;

  // Returns the 1-based line number.
  auto GetLineNumber(Token token) const -> int;

  // Returns the 1-based column number.
  auto GetColumnNumber(Token token) const -> int;

  // Returns the source text lexed into this token.
  auto GetTokenText(Token token) const -> llvm::StringRef;

  // Returns the identifier associated with this token. The token kind must be
  // an `Identifier`.
  auto GetIdentifier(Token token) const -> Identifier;

  // Returns the value of an `IntegerLiteral()` token.
  auto GetIntegerLiteral(Token token) const -> llvm::APInt;

  // Returns the closing token matched with the given opening token.
  //
  // The given token must be an opening token kind.
  auto GetMatchedClosingToken(Token opening_token) const -> Token;

  // Returns the opening token matched with the given closing token.
  //
  // The given token must be a closing token kind.
  auto GetMatchedOpeningToken(Token closing_token) const -> Token;

  // Returns whether the token was created as part of an error recovery effort.
  //
  // For example, a closing paren inserted to match an unmatched paren.
  auto IsRecoveryToken(Token token) const -> bool;

  // Returns the 1-based line number.
  auto GetLineNumber(Line line) const -> int;

  // Returns the 1-based indentation column number.
  auto GetIndentColumnNumber(Line line) const -> int;

  // Returns the text for an identifier.
  auto GetIdentifierText(Identifier id) const -> llvm::StringRef;

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

  // Prints a description of a single token.  See `print` for details on the
  // format.
  auto PrintToken(llvm::raw_ostream& output_stream, Token token) const -> void;

 private:
  // Implementation detail struct implementing the actual lexer logic.
  class Lexer;
  friend Lexer;

  // Specifies minimum widths to use when printing a token's fields via
  // `printToken`.
  struct PrintWidths {
    int index;
    int kind;
    int column;
    int line;
    int indent;

    // Widens `this` to the maximum of `this` and `new_width` for each
    // dimension.
    void Widen(const PrintWidths& new_width);
  };

  struct TokenInfo {
    TokenKind kind;

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
    // null terminator or EOF).
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
  explicit TokenizedBuffer(SourceBuffer& source) : source(&source) {}

  auto GetLineInfo(Line line) -> LineInfo&;
  auto GetLineInfo(Line line) const -> const LineInfo&;
  auto AddLine(LineInfo info) -> Line;
  auto GetTokenInfo(Token token) -> TokenInfo&;
  auto GetTokenInfo(Token token) const -> const TokenInfo&;
  auto AddToken(TokenInfo info) -> Token;
  auto GetTokenPrintWidths(Token token) const -> PrintWidths;
  auto PrintToken(llvm::raw_ostream& output_stream, Token token,
                  PrintWidths widths) const -> void;

  SourceBuffer* source;

  llvm::SmallVector<TokenInfo, 16> token_infos;

  llvm::SmallVector<LineInfo, 16> line_infos;

  llvm::SmallVector<IdentifierInfo, 16> identifier_infos;

  llvm::SmallVector<llvm::APInt, 16> int_literals;

  llvm::DenseMap<llvm::StringRef, Identifier> identifier_map;

  bool has_errors = false;
};

}  // namespace Carbon

#endif  // LEXER_TOKENIZED_BUFFER_H_
