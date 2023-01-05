// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LEXER_TOKEN_KIND_H_
#define CARBON_TOOLCHAIN_LEXER_TOKEN_KIND_H_

#include <cstdint>

#include "common/ostream.h"
#include "toolchain/common/enum_base.h"

namespace Carbon {

CARBON_DEFINE_RAW_ENUM_CLASS(TokenKind, uint8_t) {
#define CARBON_TOKEN(TokenName) CARBON_RAW_ENUM_ENUMERATOR(TokenName)
#include "toolchain/lexer/token_kind.def"
};

class TokenKind : public CARBON_ENUM_BASE(TokenKind) {
 public:
#define CARBON_TOKEN(TokenName) CARBON_ENUM_CONSTANT_DECLARATION(TokenName)
#include "toolchain/lexer/token_kind.def"

  // Test whether this kind of token is a simple symbol sequence (punctuation,
  // not letters) that appears directly in the source text and can be
  // unambiguously lexed with `starts_with` logic. While these may appear
  // inside of other tokens, outside of the contents of other tokens they
  // don't require any specific characters before or after to distinguish them
  // in the source. Returns false otherwise.
  [[nodiscard]] auto is_symbol() const -> bool;

  // Test whether this kind of token is a grouping symbol (part of an opening
  // and closing pair that must always be matched in the token stream).
  [[nodiscard]] auto is_grouping_symbol() const -> bool;

  // Test whether this kind of token is an opening symbol for a group.
  [[nodiscard]] auto is_opening_symbol() const -> bool;

  // Returns the associated closing symbol for an opening symbol.
  //
  // The token kind must be an opening symbol.
  [[nodiscard]] auto closing_symbol() const -> TokenKind;

  // Test whether this kind of token is a closing symbol for a group.
  [[nodiscard]] auto is_closing_symbol() const -> bool;

  // Returns the associated opening symbol for a closing symbol.
  //
  // The token kind must be a closing symbol.
  [[nodiscard]] auto opening_symbol() const -> TokenKind;

  // Test whether this kind of token is a keyword.
  [[nodiscard]] auto is_keyword() const -> bool;

  // Test whether this kind of token is a sized type literal.
  [[nodiscard]] auto is_sized_type_literal() const -> bool;

  // If this token kind has a fixed spelling when in source code, returns it.
  // Otherwise returns an empty string.
  [[nodiscard]] auto fixed_spelling() const -> llvm::StringRef;

  // Test whether this token kind is in the provided list.
  [[nodiscard]] auto IsOneOf(std::initializer_list<TokenKind> kinds) const
      -> bool {
    for (TokenKind kind : kinds) {
      if (*this == kind) {
        return true;
      }
    }
    return false;
  }

  // Override the EnumBase printing to use the fixed spelling rather than the
  // name for tokens as this better corresponds to the source code the
  // represent.
  void Print(llvm::raw_ostream& out) const { out << fixed_spelling(); }
};

#define CARBON_TOKEN(TokenName) \
  CARBON_ENUM_CONSTANT_DEFINITION(TokenKind, TokenName)
#include "toolchain/lexer/token_kind.def"

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_LEXER_TOKEN_KIND_H_
