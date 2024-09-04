// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LEX_TOKENIZED_BUFFER_H_
#define CARBON_TOOLCHAIN_LEX_TOKENIZED_BUFFER_H_

#include <compare>
#include <cstdint>
#include <iterator>

#include "common/ostream.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/raw_ostream.h"
#include "toolchain/base/index_base.h"
#include "toolchain/base/mem_usage.h"
#include "toolchain/base/value_store.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/lex/token_index.h"
#include "toolchain/lex/token_kind.h"
#include "toolchain/source/source_buffer.h"

namespace Carbon::Lex {

class TokenizedBuffer;

// A lightweight handle to a lexed line in a `TokenizedBuffer`.
//
// `LineIndex` objects are designed to be passed by value, not reference or
// pointer. They are also designed to be small and efficient to store in data
// structures.
//
// Each `LineIndex` object refers to a specific line in the source code that was
// lexed. They can be compared directly to establish that they refer to the
// same line or the relative position of different lines within the source.
//
// All other APIs to query a `LineIndex` are on the `TokenizedBuffer`.
struct LineIndex : public IndexBase {
  static const LineIndex Invalid;
  using IndexBase::IndexBase;
};

constexpr LineIndex LineIndex::Invalid(LineIndex::InvalidIndex);

// Random-access iterator over tokens within the buffer.
class TokenIterator
    : public llvm::iterator_facade_base<TokenIterator,
                                        std::random_access_iterator_tag,
                                        const TokenIndex, int>,
      public Printable<TokenIterator> {
 public:
  TokenIterator() = delete;

  explicit TokenIterator(TokenIndex token) : token_(token) {}

  auto operator==(const TokenIterator& rhs) const -> bool {
    return token_ == rhs.token_;
  }
  auto operator<=>(const TokenIterator& rhs) const -> std::strong_ordering {
    return token_ <=> rhs.token_;
  }

  auto operator*() const -> const TokenIndex& { return token_; }

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

  TokenIndex token_;
};

// A diagnostic location converter that maps token locations into source
// buffer locations.
class TokenDiagnosticConverter : public DiagnosticConverter<TokenIndex> {
 public:
  explicit TokenDiagnosticConverter(const TokenizedBuffer* buffer)
      : buffer_(buffer) {}

  // Map the given token into a diagnostic location.
  auto ConvertLoc(TokenIndex token, ContextFnT context_fn) const
      -> DiagnosticLoc override;

 private:
  const TokenizedBuffer* buffer_;
};

// A buffer of tokenized Carbon source code.
//
// This is constructed by lexing the source code text into a series of tokens.
// The buffer provides lightweight handles to tokens and other lexed entities,
// as well as iterations to walk the sequence of tokens found in the buffer.
//
// Lexing errors result in a potentially incomplete sequence of tokens and
// `HasError` returning true.
class TokenizedBuffer : public Printable<TokenizedBuffer> {
 public:
  auto GetKind(TokenIndex token) const -> TokenKind;
  auto GetLine(TokenIndex token) const -> LineIndex;

  // Returns the 1-based line number.
  auto GetLineNumber(TokenIndex token) const -> int;

  // Returns the 1-based column number.
  auto GetColumnNumber(TokenIndex token) const -> int;

  // Returns the line and 1-based column number of the first character after
  // this token.
  auto GetEndLoc(TokenIndex token) const -> std::pair<LineIndex, int>;

  // Returns the source text lexed into this token.
  auto GetTokenText(TokenIndex token) const -> llvm::StringRef;

  // Returns the identifier associated with this token. The token kind must be
  // an `Identifier`.
  auto GetIdentifier(TokenIndex token) const -> IdentifierId;

  // Returns the value of an `IntLiteral()` token.
  auto GetIntLiteral(TokenIndex token) const -> IntId;

  // Returns the value of an `RealLiteral()` token.
  auto GetRealLiteral(TokenIndex token) const -> RealId;

  // Returns the value of a `StringLiteral()` token.
  auto GetStringLiteralValue(TokenIndex token) const -> StringLiteralValueId;

  // Returns the size specified in a `*TypeLiteral()` token.
  auto GetTypeLiteralSize(TokenIndex token) const -> IntId;

  // Returns the closing token matched with the given opening token.
  //
  // The given token must be an opening token kind.
  auto GetMatchedClosingToken(TokenIndex opening_token) const -> TokenIndex;

  // Returns the opening token matched with the given closing token.
  //
  // The given token must be a closing token kind.
  auto GetMatchedOpeningToken(TokenIndex closing_token) const -> TokenIndex;

  // Returns whether the given token has leading whitespace.
  auto HasLeadingWhitespace(TokenIndex token) const -> bool;
  // Returns whether the given token has trailing whitespace.
  auto HasTrailingWhitespace(TokenIndex token) const -> bool;

  // Returns whether the token was created as part of an error recovery effort.
  //
  // For example, a closing paren inserted to match an unmatched paren.
  auto IsRecoveryToken(TokenIndex token) const -> bool;

  // Returns the 1-based line number.
  auto GetLineNumber(LineIndex line) const -> int;

  // Returns the 1-based indentation column number.
  auto GetIndentColumnNumber(LineIndex line) const -> int;

  // Returns the next line handle.
  auto GetNextLine(LineIndex line) const -> LineIndex;

  // Returns the previous line handle.
  auto GetPrevLine(LineIndex line) const -> LineIndex;

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
  auto PrintToken(llvm::raw_ostream& output_stream, TokenIndex token) const
      -> void;

  // Collects memory usage of members.
  auto CollectMemUsage(MemUsage& mem_usage, llvm::StringRef label) const
      -> void;

  // Returns true if the buffer has errors that were detected at lexing time.
  auto has_errors() const -> bool { return has_errors_; }

  auto tokens() const -> llvm::iterator_range<TokenIterator> {
    return llvm::make_range(TokenIterator(TokenIndex(0)),
                            TokenIterator(TokenIndex(token_infos_.size())));
  }

  auto size() const -> int { return token_infos_.size(); }

  auto expected_parse_tree_size() const -> int {
    return expected_parse_tree_size_;
  }

  auto source() const -> const SourceBuffer& { return *source_; }

 private:
  friend class Lexer;
  friend class TokenDiagnosticConverter;

  // A diagnostic location converter that maps token locations into source
  // buffer locations.
  class SourceBufferDiagnosticConverter
      : public DiagnosticConverter<const char*> {
   public:
    explicit SourceBufferDiagnosticConverter(const TokenizedBuffer* buffer)
        : buffer_(buffer) {}

    // Map the given position within the source buffer into a diagnostic
    // location.
    auto ConvertLoc(const char* loc, ContextFnT context_fn) const
        -> DiagnosticLoc override;

   private:
    const TokenizedBuffer* buffer_;
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

  class TokenInfo {
   public:
    auto kind() const -> TokenKind { return TokenKind::FromInt(kind_); }
    auto has_leading_space() const -> bool { return has_leading_space_; }

    auto ident_id() const -> IdentifierId {
      CARBON_DCHECK(kind() == TokenKind::Identifier);
      return IdentifierId(token_payload_);
    }
    auto set_ident_id(IdentifierId ident_id) -> void {
      CARBON_DCHECK(kind() == TokenKind::Identifier);
      CARBON_DCHECK(ident_id.index < (2 << PayloadBits));
      token_payload_ = ident_id.index;
    }

    auto string_literal_id() const -> StringLiteralValueId {
      CARBON_DCHECK(kind() == TokenKind::StringLiteral);
      return StringLiteralValueId(token_payload_);
    }

    auto int_id() const -> IntId {
      CARBON_DCHECK(kind() == TokenKind::IntLiteral ||
                    kind() == TokenKind::IntTypeLiteral ||
                    kind() == TokenKind::UnsignedIntTypeLiteral ||
                    kind() == TokenKind::FloatTypeLiteral);
      return IntId(token_payload_);
    }

    auto real_id() const -> RealId {
      CARBON_DCHECK(kind() == TokenKind::RealLiteral);
      return RealId(token_payload_);
    }

    auto closing_token_index() const -> TokenIndex {
      CARBON_DCHECK(kind().is_opening_symbol());
      return TokenIndex(token_payload_);
    }
    auto set_closing_token_index(TokenIndex closing_index) -> void {
      CARBON_DCHECK(kind().is_opening_symbol());
      CARBON_DCHECK(closing_index.index < (2 << PayloadBits));
      token_payload_ = closing_index.index;
    }

    auto opening_token_index() const -> TokenIndex {
      CARBON_DCHECK(kind().is_closing_symbol());
      return TokenIndex(token_payload_);
    }
    auto set_opening_token_index(TokenIndex opening_index) -> void {
      CARBON_DCHECK(kind().is_closing_symbol());
      CARBON_DCHECK(opening_index.index < (2 << PayloadBits));
      token_payload_ = opening_index.index;
    }

    auto error_length() const -> int {
      CARBON_DCHECK(kind() == TokenKind::Error);
      return token_payload_;
    }

    auto byte_offset() const -> int32_t { return byte_offset_; }

    TokenInfo(TokenKind kind, bool has_leading_space, int32_t byte_offset)
        : kind_(kind.AsInt()),
          has_leading_space_(has_leading_space),
          byte_offset_(byte_offset) {}

    TokenInfo(TokenKind kind, bool has_leading_space, int payload,
              int32_t byte_offset)
        : kind_(kind.AsInt()),
          has_leading_space_(has_leading_space),
          token_payload_(payload),
          byte_offset_(byte_offset) {
      CARBON_DCHECK(payload >= 0 && payload < (2 << PayloadBits)) << "Payload won't fit into unsigned bit pack: " << payload;
    }

   private:
    static constexpr int PayloadBits = 23;

    unsigned kind_ : sizeof(TokenKind) * 8;

    // Whether the token has trailing whitespace.
    unsigned has_leading_space_ : 1;

    // Payload, typically representing an index into a kind-dependent array.
    unsigned token_payload_ : PayloadBits;

    // Zero-based byte offset of the token within the file.
    int32_t byte_offset_;
  };

  struct LineInfo {
    explicit LineInfo(int32_t start) : start(start), indent(0) {}

    // Zero-based byte offset of the start of the line within the source buffer
    // provided.
    int32_t start;

    // The byte offset from the start of the line of the first non-whitespace
    // character.
    int32_t indent;
  };

  // The constructor is merely responsible for trivial initialization of
  // members. A working object of this type is built with `Lex::Lex` so that its
  // return can indicate if an error was encountered while lexing.
  explicit TokenizedBuffer(SharedValueStores& value_stores,
                           SourceBuffer& source)
      : value_stores_(&value_stores), source_(&source) {}

  auto FindLineIndex(int32_t byte_offset) const -> LineIndex;
  auto GetLineInfo(LineIndex line) -> LineInfo&;
  auto GetLineInfo(LineIndex line) const -> const LineInfo&;
  auto AddLine(LineInfo info) -> LineIndex;
  auto GetTokenInfo(TokenIndex token) -> TokenInfo&;
  auto GetTokenInfo(TokenIndex token) const -> const TokenInfo&;
  auto AddToken(TokenInfo info) -> TokenIndex;
  auto GetTokenPrintWidths(TokenIndex token) const -> PrintWidths;
  auto PrintToken(llvm::raw_ostream& output_stream, TokenIndex token,
                  PrintWidths widths) const -> void;

  // Used to allocate computed string literals.
  llvm::BumpPtrAllocator allocator_;

  SharedValueStores* value_stores_;
  SourceBuffer* source_;

  llvm::SmallVector<TokenInfo> token_infos_;

  llvm::SmallVector<LineInfo> line_infos_;

  // The number of parse tree nodes that we expect to be created for the tokens
  // in this buffer.
  int expected_parse_tree_size_ = 0;

  bool has_errors_ = false;

  // A vector of flags for recovery tokens. If empty, there are none. When doing
  // token recovery, this will be extended to be indexable by token indices and
  // contain true for the tokens that were synthesized for recovery.
  llvm::BitVector recovery_tokens_;
};

// A diagnostic emitter that uses positions within a source buffer's text as
// its source of location information.
using LexerDiagnosticEmitter = DiagnosticEmitter<const char*>;

// A diagnostic emitter that uses tokens as its source of location information.
using TokenDiagnosticEmitter = DiagnosticEmitter<TokenIndex>;

}  // namespace Carbon::Lex

#endif  // CARBON_TOOLCHAIN_LEX_TOKENIZED_BUFFER_H_
