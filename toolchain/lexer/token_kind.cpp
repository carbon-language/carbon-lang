// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lexer/token_kind.h"

#include "common/check.h"
#include "llvm/ADT/StringRef.h"

namespace Carbon {

auto TokenKind::Name() const -> llvm::StringRef {
  static constexpr llvm::StringLiteral Names[] = {
#define CARBON_TOKEN(TokenName) #TokenName,
#include "toolchain/lexer/token_registry.def"
  };
  return Names[static_cast<int>(kind_value_)];
}

auto TokenKind::IsSymbol() const -> bool {
  static constexpr bool Table[] = {
#define CARBON_TOKEN(TokenName) false,
#define CARBON_SYMBOL_TOKEN(TokenName, Spelling) true,
#include "toolchain/lexer/token_registry.def"
  };
  return Table[static_cast<int>(kind_value_)];
}

auto TokenKind::IsGroupingSymbol() const -> bool {
  static constexpr bool Table[] = {
#define CARBON_TOKEN(TokenName) false,
#define CARBON_OPENING_GROUP_SYMBOL_TOKEN(TokenName, Spelling, ClosingName) \
  true,
#define CARBON_CLOSING_GROUP_SYMBOL_TOKEN(TokenName, Spelling, OpeningName) \
  true,
#include "toolchain/lexer/token_registry.def"
  };
  return Table[static_cast<int>(kind_value_)];
}

auto TokenKind::IsOpeningSymbol() const -> bool {
  static constexpr bool Table[] = {
#define CARBON_TOKEN(TokenName) false,
#define CARBON_OPENING_GROUP_SYMBOL_TOKEN(TokenName, Spelling, ClosingName) \
  true,
#include "toolchain/lexer/token_registry.def"
  };
  return Table[static_cast<int>(kind_value_)];
}

auto TokenKind::GetClosingSymbol() const -> TokenKind {
  static constexpr TokenKind Table[] = {
#define CARBON_TOKEN(TokenName) Error(),
#define CARBON_OPENING_GROUP_SYMBOL_TOKEN(TokenName, Spelling, ClosingName) \
  ClosingName(),
#include "toolchain/lexer/token_registry.def"
  };
  auto result = Table[static_cast<int>(kind_value_)];
  CHECK(result != Error()) << "Only opening symbols are valid!";
  return result;
}

auto TokenKind::IsClosingSymbol() const -> bool {
  static constexpr bool Table[] = {
#define CARBON_TOKEN(TokenName) false,
#define CARBON_CLOSING_GROUP_SYMBOL_TOKEN(TokenName, Spelling, OpeningName) \
  true,
#include "toolchain/lexer/token_registry.def"
  };
  return Table[static_cast<int>(kind_value_)];
}

auto TokenKind::GetOpeningSymbol() const -> TokenKind {
  static constexpr TokenKind Table[] = {
#define CARBON_TOKEN(TokenName) Error(),
#define CARBON_CLOSING_GROUP_SYMBOL_TOKEN(TokenName, Spelling, OpeningName) \
  OpeningName(),
#include "toolchain/lexer/token_registry.def"
  };
  auto result = Table[static_cast<int>(kind_value_)];
  CHECK(result != Error()) << "Only closing symbols are valid!";
  return result;
}

auto TokenKind::IsKeyword() const -> bool {
  static constexpr bool Table[] = {
#define CARBON_TOKEN(TokenName) false,
#define CARBON_KEYWORD_TOKEN(TokenName, Spelling) true,
#include "toolchain/lexer/token_registry.def"
  };
  return Table[static_cast<int>(kind_value_)];
}

auto TokenKind::IsSizedTypeLiteral() const -> bool {
  return *this == TokenKind::IntegerTypeLiteral() ||
         *this == TokenKind::UnsignedIntegerTypeLiteral() ||
         *this == TokenKind::FloatingPointTypeLiteral();
}

auto TokenKind::GetFixedSpelling() const -> llvm::StringRef {
  static constexpr llvm::StringLiteral Table[] = {
#define CARBON_TOKEN(TokenName) "",
#define CARBON_SYMBOL_TOKEN(TokenName, Spelling) Spelling,
#define CARBON_KEYWORD_TOKEN(TokenName, Spelling) Spelling,
#include "toolchain/lexer/token_registry.def"
  };
  return Table[static_cast<int>(kind_value_)];
}

}  // namespace Carbon
