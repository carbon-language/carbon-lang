// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LEX_TOKEN_INFO_H_
#define CARBON_TOOLCHAIN_LEX_TOKEN_INFO_H_

#include "common/check.h"
#include "toolchain/base/int.h"
#include "toolchain/base/value_ids.h"
#include "toolchain/lex/token_index.h"
#include "toolchain/lex/token_kind.h"

namespace Carbon::Lex {

// A character as a unicode code point.
//
// Unicode requires 21 bits, which should fit inside `TokenInfo::PayloadBits`,
// so we store the value directly.
struct CharLiteralValue {
  int32_t value;
};

// Storage for the information about a specific token, as an implementation
// detail of `TokenizedBuffer`.
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

  auto char_literal() const -> CharLiteralValue {
    CARBON_DCHECK(kind() == TokenKind::CharLiteral);
    return CharLiteralValue(token_payload_);
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

}  // namespace Carbon::Lex

#endif  // CARBON_TOOLCHAIN_LEX_TOKEN_INFO_H_
