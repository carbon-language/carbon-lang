// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LEX_TOKENIZED_BUFFER_H_
#define CARBON_TOOLCHAIN_LEX_TOKENIZED_BUFFER_H_

#include <cstdint>

#include "common/ostream.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/raw_ostream.h"
#include "toolchain/base/index_base.h"
#include "toolchain/base/mem_usage.h"
#include "toolchain/base/shared_value_stores.h"
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
struct LineIndex : public IndexBase<LineIndex> {
  static constexpr llvm::StringLiteral Label = "line";
  static const LineIndex None;
  using IndexBase::IndexBase;
};

constexpr LineIndex LineIndex::None(NoneIndex);

// Indices for comments within the buffer.
struct CommentIndex : public IndexBase<CommentIndex> {
  static constexpr llvm::StringLiteral Label = "comment";
  static const CommentIndex None;
  using IndexBase::IndexBase;
};

constexpr CommentIndex CommentIndex::None(NoneIndex);

// Random-access iterator over comments within the buffer.
using CommentIterator = IndexIterator<CommentIndex>;

// Random-access iterator over tokens within the buffer.
using TokenIterator = IndexIterator<TokenIndex>;

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
  // A comment, which can be a block of lines.
  //
  // This is the API version of `CommentData`.
  struct CommentInfo {
    // The comment's full text, including `//` symbols. This may have several
    // lines for block comments.
    llvm::StringRef text;

    // The comment's indent.
    int32_t indent;

    // The first line of the comment.
    LineIndex start_line;
  };

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

  // Returns the 1-based indentation column number.
  auto GetIndentColumnNumber(LineIndex line) const -> int;

  // Returns the next line handle.
  auto GetNextLine(LineIndex line) const -> LineIndex;

  // Returns the previous line handle.
  auto GetPrevLine(LineIndex line) const -> LineIndex;

  auto GetByteOffset(TokenIndex token) const -> int32_t {
    return GetTokenInfo(token).byte_offset();
  }

  // Returns true if the token comes after the comment.
  auto IsAfterComment(TokenIndex token, CommentIndex comment_index) const
      -> bool;

  // Returns the comment's full text range.
  auto GetCommentText(CommentIndex comment_index) const -> llvm::StringRef;

  // Returns tokens as YAML. This prints the tracked token information on a
  // single line for each token. We use the single-line format so that output is
  // compact, and so that tools like `grep` are compatible.
  //
  // An example token looks like:
  //
  // - { index: 1, kind: 'Semi', line: 1, column: 1, indent: 1, spelling: ';' }
  auto Print(llvm::raw_ostream& out,
             bool omit_file_boundary_tokens = false) const -> void;

  // Prints a description of a single token.  See `Print` for details on the
  // format.
  auto PrintToken(llvm::raw_ostream& output_stream, TokenIndex token) const
      -> void;

  // Collects memory usage of members.
  auto CollectMemUsage(MemUsage& mem_usage, llvm::StringRef label) const
      -> void;

  // Converts a token to a diagnostic location.
  auto TokenToDiagnosticLoc(TokenIndex token) const -> ConvertedDiagnosticLoc;

  // Returns true if the buffer has errors that were detected at lexing time.
  auto has_errors() const -> bool { return has_errors_; }

  auto tokens() const -> llvm::iterator_range<TokenIterator> {
    return llvm::make_range(TokenIterator(TokenIndex(0)),
                            TokenIterator(TokenIndex(token_infos_.size())));
  }

  auto size() const -> int { return token_infos_.size(); }

  auto comments() const -> llvm::iterator_range<CommentIterator> {
    return llvm::make_range(CommentIterator(CommentIndex(0)),
                            CommentIterator(CommentIndex(comments_.size())));
  }

  auto comments_size() const -> size_t { return comments_.size(); }

  // This is an upper bound on the number of output parse nodes in the absence
  // of errors.
  auto expected_max_parse_tree_size() const -> int {
    return expected_max_parse_tree_size_;
  }

  auto source() const -> const SourceBuffer& { return *source_; }

 private:
  friend class Lexer;

  class SourcePointerDiagnosticEmitter : public DiagnosticEmitter<const char*> {
   public:
    explicit SourcePointerDiagnosticEmitter(DiagnosticConsumer* consumer,
                                            const TokenizedBuffer* tokens)
        : DiagnosticEmitter(consumer), tokens_(tokens) {}

   protected:
    auto ConvertLoc(const char* loc, ContextFnT /*context_fn*/) const
        -> ConvertedDiagnosticLoc override {
      return tokens_->SourcePointerToDiagnosticLoc(loc);
    }

   private:
    const TokenizedBuffer* tokens_;
  };

  class TokenDiagnosticEmitter : public DiagnosticEmitter<TokenIndex> {
   public:
    explicit TokenDiagnosticEmitter(DiagnosticConsumer* consumer,
                                    const TokenizedBuffer* tokens)
        : DiagnosticEmitter(consumer), tokens_(tokens) {}

   protected:
    auto ConvertLoc(TokenIndex token, ContextFnT /*context_fn*/) const
        -> ConvertedDiagnosticLoc override {
      return tokens_->TokenToDiagnosticLoc(token);
    }

   private:
    const TokenizedBuffer* tokens_;
  };

  // Converts a pointer into the source to a diagnostic location.
  auto SourcePointerToDiagnosticLoc(const char* loc) const
      -> ConvertedDiagnosticLoc;

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

  // Storage for the information about a specific token in the buffer.
  //
  // This provides a friendly accessor API to the carefully space-optimized
  // storage model of the information we associated with each token.
  //
  // There are four pieces of information stored here:
  // - The kind of the token.
  // - Whether that token has leading whitespace before it.
  // - A kind-specific payload that can be compressed into a small integer.
  //   - This class provides dedicated accessors for each different form of
  //     payload that check the kind and payload correspond correctly.
  // - A 32-bit byte offset of the token within the source text.
  //
  // These are compressed and stored in 8-bytes for each token.
  //
  // Note that while the class provides some limited setters for payloads and
  // mutating methods, setters on this type may be unexpectedly expensive due to
  // the bit-packed representation and should be avoided. As such, only the
  // minimal necessary setters are provided.
  //
  // TODO: It might be worth considering a struct-of-arrays data layout in order
  // to move the byte offset to a separate array from the rest as it is only hot
  // during lexing, and then cold during parsing and semantic analysis. However,
  // a trivial approach to that adds more overhead than it saves due to tracking
  // two separate vectors and their growth. Making this profitable would likely
  // at least require a highly specialized single vector that manages the growth
  // once and then provides separate storage areas for the two arrays.
  class TokenInfo {
   public:
    // The kind for this token.
    auto kind() const -> TokenKind { return kind_; }

    // Whether this token is preceded by whitespace. We only store the preceding
    // state, and look at the next token to check for trailing whitespace.
    auto has_leading_space() const -> bool { return has_leading_space_; }

    // A collection of methods to access the specific payload included with
    // particular kinds of tokens. Only the specific payload accessor below may
    // be used for an info entry of a token with a particular kind, and these
    // check that the kind is valid. Some tokens do not include a payload at all
    // and none of these methods may be called.
    auto ident_id() const -> IdentifierId {
      CARBON_DCHECK(kind() == TokenKind::Identifier);
      return IdentifierId(token_payload_);
    }
    auto set_ident_id(IdentifierId ident_id) -> void {
      CARBON_DCHECK(kind() == TokenKind::Identifier);
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
      return IntId::MakeFromTokenPayload(token_payload_);
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
      token_payload_ = closing_index.index;
    }

    auto opening_token_index() const -> TokenIndex {
      CARBON_DCHECK(kind().is_closing_symbol());
      return TokenIndex(token_payload_);
    }
    auto set_opening_token_index(TokenIndex opening_index) -> void {
      CARBON_DCHECK(kind().is_closing_symbol());
      token_payload_ = opening_index.index;
    }

    auto error_length() const -> int {
      CARBON_DCHECK(kind() == TokenKind::Error);
      return token_payload_;
    }

    // Zero-based byte offset of the token within the file. This can be combined
    // with the buffer's line information to locate the line and column of the
    // token as well.
    auto byte_offset() const -> int32_t { return byte_offset_; }

    // Transforms the token into an error token of the given length but at its
    // original position and with the same whitespace adjacency.
    auto ResetAsError(int error_length) -> void {
      // Construct a fresh token to establish any needed invariants and replace
      // this token with it.
      TokenInfo error(TokenKind::Error, has_leading_space(), error_length,
                      byte_offset());
      *this = error;
    }

   private:
    friend class Lexer;

    static constexpr int PayloadBits = 23;

    // Make sure we have enough payload bits to represent token-associated IDs.
    static_assert(PayloadBits >= IntId::TokenIdBits);
    static_assert(PayloadBits >= TokenIndex::Bits);

    // Constructor for a TokenKind that carries no payload, or where the payload
    // will be set later.
    //
    // Only used by the lexer which enforces only the correct kinds are used.
    //
    // When the payload is not being set, we leave it uninitialized. At least in
    // some cases, this will allow MSan to correctly detect erroneous attempts
    // to access the payload, as it works to track uninitialized memory
    // bit-for-bit specifically to handle complex cases like bitfields.
    TokenInfo(TokenKind kind, bool has_leading_space, int32_t byte_offset)
        : kind_(kind),
          has_leading_space_(has_leading_space),
          byte_offset_(byte_offset) {}

    // Constructor for a TokenKind that carries a payload.
    //
    // Only used by the lexer which enforces the correct kind and payload types.
    TokenInfo(TokenKind kind, bool has_leading_space, int payload,
              int32_t byte_offset)
        : kind_(kind),
          has_leading_space_(has_leading_space),
          token_payload_(payload),
          byte_offset_(byte_offset) {}

    // A bitfield that encodes the token's kind, the leading space flag, and the
    // remaining bits in a payload. These are encoded together as a bitfield for
    // density and because these are the hottest fields of tokens for consumers
    // after lexing.
    //
    // Payload values are typically ID types for which we create at most one per
    // token, so we ensure that `token_payload_` is large enough to fit any
    // token index. Stores to this field may overflow, but we produce an error
    // in `Lexer::Finalize` if the file has more than `TokenIndex::Max` tokens,
    // so this value never overflows if lexing succeeds.
    TokenKind kind_;
    static_assert(sizeof(kind_) == 1, "TokenKind must pack to 8 bits");
    bool has_leading_space_ : 1;
    unsigned token_payload_ : PayloadBits;

    // Separate storage for the byte offset, this is hot while lexing but then
    // generally cold.
    int32_t byte_offset_;
  };
  static_assert(sizeof(TokenInfo) == 8,
                "Expected `TokenInfo` to pack to an 8-byte structure.");

  // A comment, which can be a block of lines. These are tracked separately from
  // tokens because they don't affect parse; if they were part of tokens, we'd
  // need more general special-casing within token logic.
  //
  // Note that `CommentInfo` is used for an API to expose the comment.
  struct CommentData {
    // Zero-based byte offset of the start of the comment within the source
    // buffer provided.
    int32_t start;

    // The comment's length.
    int32_t length;
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
  explicit TokenizedBuffer(SharedValueStores& value_stores
                           [[clang::lifetimebound]],
                           SourceBuffer& source [[clang::lifetimebound]])
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

  // Adds a comment. This uses the indent to potentially stitch together two
  // adjacent comments.
  auto AddComment(int32_t indent, int32_t start, int32_t end) -> void;

  // Used to allocate computed string literals.
  llvm::BumpPtrAllocator allocator_;

  SharedValueStores* value_stores_;
  SourceBuffer* source_;

  llvm::SmallVector<TokenInfo> token_infos_;

  llvm::SmallVector<LineInfo> line_infos_;

  // Comments in the file.
  llvm::SmallVector<CommentData> comments_;

  // An upper bound on the number of parse tree nodes that we expect to be
  // created for the tokens in this buffer.
  int expected_max_parse_tree_size_ = 0;

  bool has_errors_ = false;

  // A vector of flags for recovery tokens. If empty, there are none. When doing
  // token recovery, this will be extended to be indexable by token indices and
  // contain true for the tokens that were synthesized for recovery.
  llvm::BitVector recovery_tokens_;
};

inline auto TokenizedBuffer::GetKind(TokenIndex token) const -> TokenKind {
  return GetTokenInfo(token).kind();
}

inline auto TokenizedBuffer::HasLeadingWhitespace(TokenIndex token) const
    -> bool {
  return GetTokenInfo(token).has_leading_space();
}

inline auto TokenizedBuffer::HasTrailingWhitespace(TokenIndex token) const
    -> bool {
  TokenIterator it(token);
  ++it;
  return it != tokens().end() && GetTokenInfo(*it).has_leading_space();
}

inline auto TokenizedBuffer::GetTokenInfo(TokenIndex token) -> TokenInfo& {
  return token_infos_[token.index];
}

inline auto TokenizedBuffer::GetTokenInfo(TokenIndex token) const
    -> const TokenInfo& {
  return token_infos_[token.index];
}

inline auto TokenizedBuffer::AddToken(TokenInfo info) -> TokenIndex {
  TokenIndex index(token_infos_.size());
  token_infos_.push_back(info);
  expected_max_parse_tree_size_ += info.kind().expected_max_parse_tree_size();
  return index;
}

}  // namespace Carbon::Lex

#endif  // CARBON_TOOLCHAIN_LEX_TOKENIZED_BUFFER_H_
