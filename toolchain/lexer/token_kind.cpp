// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lexer/token_kind.h"

#include "common/check.h"
#include "llvm/ADT/StringRef.h"

namespace Carbon {

CARBON_DEFINE_ENUM_CLASS_NAMES(TokenKind) = {
#define CARBON_TOKEN(TokenName) CARBON_ENUM_CLASS_NAME_STRING(TokenName)
#include "toolchain/lexer/token_kind.def"
};

auto TokenKind::is_symbol() const -> bool {
  static constexpr bool Table[] = {
#define CARBON_TOKEN(TokenName) false,
#define CARBON_SYMBOL_TOKEN(TokenName, Spelling) true,
#include "toolchain/lexer/token_kind.def"
  };
  return Table[AsInt()];
}

auto TokenKind::is_grouping_symbol() const -> bool {
  static constexpr bool Table[] = {
#define CARBON_TOKEN(TokenName) false,
#define CARBON_OPENING_GROUP_SYMBOL_TOKEN(TokenName, Spelling, ClosingName) \
  true,
#define CARBON_CLOSING_GROUP_SYMBOL_TOKEN(TokenName, Spelling, OpeningName) \
  true,
#include "toolchain/lexer/token_kind.def"
  };
  return Table[AsInt()];
}

auto TokenKind::is_opening_symbol() const -> bool {
  static constexpr bool Table[] = {
#define CARBON_TOKEN(TokenName) false,
#define CARBON_OPENING_GROUP_SYMBOL_TOKEN(TokenName, Spelling, ClosingName) \
  true,
#include "toolchain/lexer/token_kind.def"
  };
  return Table[AsInt()];
}

auto TokenKind::closing_symbol() const -> TokenKind {
  static constexpr TokenKind Table[] = {
#define CARBON_TOKEN(TokenName) Error,
#define CARBON_OPENING_GROUP_SYMBOL_TOKEN(TokenName, Spelling, ClosingName) \
  ClosingName,
#include "toolchain/lexer/token_kind.def"
  };
  auto result = Table[AsInt()];
  CARBON_CHECK(result != Error) << "Only opening symbols are valid!";
  return result;
}

auto TokenKind::is_closing_symbol() const -> bool {
  static constexpr bool Table[] = {
#define CARBON_TOKEN(TokenName) false,
#define CARBON_CLOSING_GROUP_SYMBOL_TOKEN(TokenName, Spelling, OpeningName) \
  true,
#include "toolchain/lexer/token_kind.def"
  };
  return Table[AsInt()];
}

auto TokenKind::opening_symbol() const -> TokenKind {
  static constexpr TokenKind Table[] = {
#define CARBON_TOKEN(TokenName) Error,
#define CARBON_CLOSING_GROUP_SYMBOL_TOKEN(TokenName, Spelling, OpeningName) \
  OpeningName,
#include "toolchain/lexer/token_kind.def"
  };
  auto result = Table[AsInt()];
  CARBON_CHECK(result != Error) << "Only closing symbols are valid!";
  return result;
}

auto TokenKind::is_keyword() const -> bool {
  static constexpr bool Table[] = {
#define CARBON_TOKEN(TokenName) false,
#define CARBON_KEYWORD_TOKEN(TokenName, Spelling) true,
#include "toolchain/lexer/token_kind.def"
  };
  return Table[AsInt()];
}

auto TokenKind::is_sized_type_literal() const -> bool {
  return *this == TokenKind::IntegerTypeLiteral ||
         *this == TokenKind::UnsignedIntegerTypeLiteral ||
         *this == TokenKind::FloatingPointTypeLiteral;
}

auto TokenKind::fixed_spelling() const -> llvm::StringRef {
  static constexpr llvm::StringLiteral Table[] = {
#define CARBON_TOKEN(TokenName) "",
#define CARBON_SYMBOL_TOKEN(TokenName, Spelling) Spelling,
#define CARBON_KEYWORD_TOKEN(TokenName, Spelling) Spelling,
#include "toolchain/lexer/token_kind.def"
  };
  return Table[AsInt()];
}

}  // namespace Carbon
