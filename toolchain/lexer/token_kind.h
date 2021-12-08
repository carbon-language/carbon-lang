// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TOOLCHAIN_LEXER_TOKEN_KIND_H_
#define TOOLCHAIN_LEXER_TOKEN_KIND_H_

#include <cstdint>
#include <initializer_list>
#include <iterator>

#include "llvm/ADT/StringRef.h"

namespace Carbon {

class TokenKind {
  // Note that this must be declared earlier in the class so that its type can
  // be used, for example in the conversion operator.
  enum class KindEnum : int8_t {
#define CARBON_TOKEN(TokenName) TokenName,
#include "toolchain/lexer/token_registry.def"
  };

 public:
  // The formatting for this macro is weird due to a `clang-format` bug. See
  // https://bugs.llvm.org/show_bug.cgi?id=48320 for details.
#define CARBON_TOKEN(TokenName)                  \
  static constexpr auto TokenName()->TokenKind { \
    return TokenKind(KindEnum::TokenName);       \
  }
#include "toolchain/lexer/token_registry.def"

  // The default constructor is deleted as objects of this type should always be
  // constructed using the above factory functions for each unique kind.
  TokenKind() = delete;

  friend auto operator==(TokenKind lhs, TokenKind rhs) -> bool {
    return lhs.kind_value_ == rhs.kind_value_;
  }
  friend auto operator!=(TokenKind lhs, TokenKind rhs) -> bool {
    return lhs.kind_value_ != rhs.kind_value_;
  }

  // Get a friendly name for the token for logging or debugging.
  [[nodiscard]] auto Name() const -> llvm::StringRef;

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

  // Enable conversion to our private enum, including in a `constexpr` context,
  // to enable usage in `switch` and `case`. The enum remains private and
  // nothing else should be using this.
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr operator KindEnum() const { return kind_value_; }

 private:
  constexpr explicit TokenKind(KindEnum kind_value) : kind_value_(kind_value) {}

  KindEnum kind_value_;
};

}  // namespace Carbon

#endif  // TOOLCHAIN_LEXER_TOKEN_KIND_H_
