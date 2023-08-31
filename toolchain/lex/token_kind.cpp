// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lex/token_kind.h"

namespace Carbon::Lex {

CARBON_DEFINE_ENUM_CLASS_NAMES(TokenKind) = {
#define CARBON_TOKEN(TokenName) CARBON_ENUM_CLASS_NAME_STRING(TokenName)
#include "toolchain/lex/token_kind.def"
};

constexpr bool TokenKind::IsSymbol[] = {
#define CARBON_TOKEN(TokenName) false,
#define CARBON_SYMBOL_TOKEN(TokenName, Spelling) true,
#include "toolchain/lex/token_kind.def"
};

constexpr bool TokenKind::IsGroupingSymbol[] = {
#define CARBON_TOKEN(TokenName) false,
#define CARBON_OPENING_GROUP_SYMBOL_TOKEN(TokenName, Spelling, ClosingName) \
  true,
#define CARBON_CLOSING_GROUP_SYMBOL_TOKEN(TokenName, Spelling, OpeningName) \
  true,
#include "toolchain/lex/token_kind.def"
};

constexpr bool TokenKind::IsOpeningSymbol[] = {
#define CARBON_TOKEN(TokenName) false,
#define CARBON_OPENING_GROUP_SYMBOL_TOKEN(TokenName, Spelling, ClosingName) \
  true,
#include "toolchain/lex/token_kind.def"
};

constexpr TokenKind TokenKind::ClosingSymbol[] = {
#define CARBON_TOKEN(TokenName) Error,
#define CARBON_OPENING_GROUP_SYMBOL_TOKEN(TokenName, Spelling, ClosingName) \
  ClosingName,
#include "toolchain/lex/token_kind.def"
};

constexpr bool TokenKind::IsClosingSymbol[] = {
#define CARBON_TOKEN(TokenName) false,
#define CARBON_CLOSING_GROUP_SYMBOL_TOKEN(TokenName, Spelling, OpeningName) \
  true,
#include "toolchain/lex/token_kind.def"
};

constexpr TokenKind TokenKind::OpeningSymbol[] = {
#define CARBON_TOKEN(TokenName) Error,
#define CARBON_CLOSING_GROUP_SYMBOL_TOKEN(TokenName, Spelling, OpeningName) \
  OpeningName,
#include "toolchain/lex/token_kind.def"
};

constexpr bool TokenKind::IsOneCharSymbol[] = {
#define CARBON_TOKEN(TokenName) false,
#define CARBON_ONE_CHAR_SYMBOL_TOKEN(TokenName, Spelling) true,
#include "toolchain/lex/token_kind.def"
};

constexpr bool TokenKind::IsKeyword[] = {
#define CARBON_TOKEN(TokenName) false,
#define CARBON_KEYWORD_TOKEN(TokenName, Spelling) true,
#include "toolchain/lex/token_kind.def"
};

constexpr llvm::StringLiteral TokenKind::FixedSpelling[] = {
#define CARBON_TOKEN(TokenName) "",
#define CARBON_SYMBOL_TOKEN(TokenName, Spelling) Spelling,
#define CARBON_KEYWORD_TOKEN(TokenName, Spelling) Spelling,
#include "toolchain/lex/token_kind.def"
};

constexpr int8_t TokenKind::ExpectedParseTreeSize[] = {
#define CARBON_TOKEN(Name) 1,
#define CARBON_TOKEN_WITH_VIRTUAL_NODE(size) 2,
#include "toolchain/lex/token_kind.def"
};

}  // namespace Carbon::Lex
