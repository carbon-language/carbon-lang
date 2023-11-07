// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LEX_LEXER_H_
#define CARBON_TOOLCHAIN_LEX_LEXER_H_

#include "common/check.h"
#include "llvm/ADT/StringRef.h"
#include "toolchain/base/value_store.h"
#include "toolchain/lex/tokenized_buffer.h"

namespace Carbon::Lex {

// Implementation of the lexer logic itself.
//
// The design is that lexing can loop over the source buffer, consuming it into
// tokens by calling into this API. This class handles the state and breaks down
// the different lexing steps that may be used. It directly updates the provided
// tokenized buffer with the lexed tokens.
class Lexer {
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

  Lexer(SharedValueStores& value_stores, SourceBuffer& source,
        DiagnosticConsumer& consumer)
      : buffer_(value_stores, source),
        consumer_(consumer),
        translator_(&buffer_),
        emitter_(translator_, consumer_),
        token_translator_(&buffer_),
        token_emitter_(token_translator_, consumer_) {}

  // Find all line endings and create the line data structures. Explicitly kept
  // out-of-line because this is a significant loop that is useful to have in
  // the profile and it doesn't simplify by inlining at all. But because it can,
  // the compiler will flatten this otherwise.
  auto CreateLines(llvm::StringRef source_text) -> void;

  auto current_line() -> Line { return Line(line_index_); }

  auto current_line_info() -> TokenizedBuffer::LineInfo* {
    return &buffer_.line_infos_[line_index_];
  }

  auto ComputeColumn(ssize_t position) -> int {
    CARBON_DCHECK(position >= current_line_info()->start);
    return position - current_line_info()->start;
  }

  auto NoteWhitespace() -> void {
    buffer_.token_infos_.back().has_trailing_space = true;
  }

  auto SkipHorizontalWhitespace(llvm::StringRef source_text, ssize_t& position)
      -> void;

  auto LexHorizontalWhitespace(llvm::StringRef source_text, ssize_t& position)
      -> void;

  auto LexVerticalWhitespace(llvm::StringRef source_text, ssize_t& position)
      -> void;

  auto LexCommentOrSlash(llvm::StringRef source_text, ssize_t& position)
      -> void;

  auto LexComment(llvm::StringRef source_text, ssize_t& position) -> void;

  auto LexNumericLiteral(llvm::StringRef source_text, ssize_t& position)
      -> LexResult;

  auto LexStringLiteral(llvm::StringRef source_text, ssize_t& position)
      -> LexResult;

  auto LexOneCharSymbolToken(llvm::StringRef source_text, TokenKind kind,
                             ssize_t& position) -> Token;

  auto LexOpeningSymbolToken(llvm::StringRef source_text, TokenKind kind,
                             ssize_t& position) -> LexResult;

  auto LexClosingSymbolToken(llvm::StringRef source_text, TokenKind kind,
                             ssize_t& position) -> LexResult;

  auto LexSymbolToken(llvm::StringRef source_text, ssize_t& position)
      -> LexResult;

  // Given a word that has already been lexed, determine whether it is a type
  // literal and if so form the corresponding token.
  auto LexWordAsTypeLiteralToken(llvm::StringRef word, int column) -> LexResult;

  // Closes all open groups that cannot remain open across a closing symbol.
  // Users may pass `Error` to close all open groups.
  auto CloseInvalidOpenGroups(TokenKind kind, ssize_t position) -> void;

  auto LexKeywordOrIdentifier(llvm::StringRef source_text, ssize_t& position)
      -> LexResult;

  auto LexKeywordOrIdentifierMaybeRaw(llvm::StringRef source_text,
                                      ssize_t& position) -> LexResult;

  auto LexError(llvm::StringRef source_text, ssize_t& position) -> LexResult;

  auto LexStartOfFile(llvm::StringRef source_text, ssize_t& position) -> void;

  auto LexEndOfFile(llvm::StringRef source_text, ssize_t position) -> void;

  // The main entry point for dispatching through the lexer's table. This method
  // should always fully consume the source text.
  auto Lex() && -> TokenizedBuffer;

 private:
  TokenizedBuffer buffer_;

  ssize_t line_index_;

  llvm::SmallVector<Token> open_groups_;

  ErrorTrackingDiagnosticConsumer consumer_;

  TokenizedBuffer::SourceBufferLocationTranslator translator_;
  LexerDiagnosticEmitter emitter_;

  TokenLocationTranslator token_translator_;
  TokenDiagnosticEmitter token_emitter_;
};

}  // namespace Carbon::Lex

#endif  // CARBON_TOOLCHAIN_LEX_LEXER_H_
