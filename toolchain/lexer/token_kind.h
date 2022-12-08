// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LEXER_TOKEN_KIND_H_
#define CARBON_TOOLCHAIN_LEXER_TOKEN_KIND_H_

#include <cstdint>
#include <initializer_list>
#include <iterator>

#include "common/ostream.h"
#include "llvm/ADT/StringRef.h"

namespace Carbon {

#define CARBON_ENUM_BASE_NAME TokenKindBase
#define CARBON_ENUM_DEF_PATH "toolchain/lexer/token_kind.def"
#include "toolchain/common/enum_base.def"

class TokenKind : public TokenKindBase<TokenKind> {
 public:
  // Test whether this kind of token is a simple symbol sequence (punctuation,
  // not letters) that appears directly in the source text and can be
  // unambiguously lexed with `starts_with` logic. While these may appear
  // inside of other tokens, outside of the contents of other tokens they
  // don't require any specific characters before or after to distinguish them
  // in the source. Returns false otherwise.
  [[nodiscard]] auto IsSymbol() const -> bool;

  // Test whether this kind of token is a grouping symbol (part of an opening
  // and closing pair that must always be matched in the token stream).
  [[nodiscard]] auto IsGroupingSymbol() const -> bool;

  // Test whether this kind of token is an opening symbol for a group.
  [[nodiscard]] auto IsOpeningSymbol() const -> bool;

  // Returns the associated closing symbol for an opening symbol.
  //
  // The token kind must be an opening symbol.
  [[nodiscard]] auto GetClosingSymbol() const -> TokenKind;

  // Test whether this kind of token is a closing symbol for a group.
  [[nodiscard]] auto IsClosingSymbol() const -> bool;

  // Returns the associated opening symbol for a closing symbol.
  //
  // The token kind must be a closing symbol.
  [[nodiscard]] auto GetOpeningSymbol() const -> TokenKind;

  // Test whether this kind of token is a keyword.
  [[nodiscard]] auto IsKeyword() const -> bool;

  // Test whether this kind of token is a sized type literal.
  [[nodiscard]] auto IsSizedTypeLiteral() const -> bool;

  // If this token kind has a fixed spelling when in source code, returns it.
  // Otherwise returns an empty string.
  [[nodiscard]] auto GetFixedSpelling() const -> llvm::StringRef;

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

  // Prints the TokenKind. This implementation is primarily for diagnostics
  // instead of tracing, which is why it uses the fixed spelling.
  void Print(llvm::raw_ostream& out) const { out << GetFixedSpelling(); }

 private:
  using TokenKindBase::TokenKindBase;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_LEXER_TOKEN_KIND_H_
