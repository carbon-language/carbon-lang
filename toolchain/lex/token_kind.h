// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LEX_TOKEN_KIND_H_
#define CARBON_TOOLCHAIN_LEX_TOKEN_KIND_H_

#include <cstdint>

#include "common/check.h"
#include "common/enum_base.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadicDetails.h"

namespace Carbon::Lex {

CARBON_DEFINE_RAW_ENUM_CLASS(TokenKind, uint8_t) {
#define CARBON_TOKEN(TokenName) CARBON_RAW_ENUM_ENUMERATOR(TokenName)
#include "toolchain/lex/token_kind.def"
};

class TokenKind : public CARBON_ENUM_BASE(TokenKind) {
 public:
#define CARBON_TOKEN(TokenName) CARBON_ENUM_CONSTANT_DECL(TokenName)
#include "toolchain/lex/token_kind.def"

  // An array of all the keyword tokens.
  static const llvm::ArrayRef<TokenKind> KeywordTokens;

  using EnumBase::EnumBase;

  // Permit creation from RawEnumType for templates.
  using EnumBase::Make;

  // Permit conversion to integer for use as an array index.
  using EnumBase::AsInt;

  // Test whether this kind of token is a simple symbol sequence (punctuation,
  // not letters) that appears directly in the source text and can be
  // unambiguously lexed with `starts_with` logic. While these may appear
  // inside of other tokens, outside of the contents of other tokens they
  // don't require any specific characters before or after to distinguish them
  // in the source. Returns false otherwise.
  constexpr auto is_symbol() const -> bool { return IsSymbol[AsInt()]; }

  // Test whether this kind of token is a grouping symbol (part of an opening
  // and closing pair that must always be matched in the token stream).
  constexpr auto is_grouping_symbol() const -> bool {
    return IsGroupingSymbol[AsInt()];
  }

  // Test whether this kind of token is an opening symbol for a group.
  constexpr auto is_opening_symbol() const -> bool {
    return IsOpeningSymbol[AsInt()];
  }

  // Returns the associated closing symbol for an opening symbol.
  //
  // The token kind must be an opening symbol.
  constexpr auto closing_symbol() const -> TokenKind {
    auto result = ClosingSymbol[AsInt()];
    CARBON_DCHECK(result != Error, "Only opening symbols are valid!");
    return result;
  }

  // Test whether this kind of token is a closing symbol for a group.
  constexpr auto is_closing_symbol() const -> bool {
    return IsClosingSymbol[AsInt()];
  }

  // Returns the associated opening symbol for a closing symbol.
  //
  // The token kind must be a closing symbol.
  constexpr auto opening_symbol() const -> TokenKind {
    auto result = OpeningSymbol[AsInt()];
    CARBON_DCHECK(result != Error, "Only closing symbols are valid!");
    return result;
  }

  // Test whether this kind of token is a one-character symbol whose character
  // is not part of any other symbol.
  constexpr auto is_one_char_symbol() const -> bool {
    return IsOneCharSymbol[AsInt()];
  }

  // Test whether this kind of token is a keyword.
  constexpr auto is_keyword() const -> bool { return IsKeyword[AsInt()]; }

  // Test whether this kind of token is a sized type literal.
  constexpr auto is_sized_type_literal() const -> bool {
    return *this == TokenKind::IntTypeLiteral ||
           *this == TokenKind::UnsignedIntTypeLiteral ||
           *this == TokenKind::FloatTypeLiteral;
  }

  // Test whether this kind of token is a dollar int literal.
  constexpr auto is_dollar_int_literal() const -> bool {
    return *this == TokenKind::DollarIntLiteral;
  }

  // Test whether this kind of token is a word.
  constexpr auto is_word() const -> bool {
    return *this == TokenKind::Identifier || *this == TokenKind::Underscore ||
           is_keyword() || is_sized_type_literal();
  }

  // Test whether this kind of token is a binding pattern operator.
  constexpr auto is_binding_pattern_operator() const -> bool {
    return *this == TokenKind::Colon || *this == TokenKind::ColonQuestion;
  }

  // If this token kind has a fixed spelling when in source code, returns it.
  // Otherwise returns an empty string.
  constexpr auto fixed_spelling() const -> llvm::StringLiteral {
    return FixedSpelling[AsInt()];
  }

  // Get the expected number of parse tree nodes that will be created for this
  // token.
  constexpr auto expected_max_parse_tree_size() const -> int {
    return ExpectedParseTreeSize[AsInt()];
  }

  // Test whether this token kind is in the provided list.
  constexpr auto IsOneOf(std::initializer_list<TokenKind> kinds) const -> bool {
    for (TokenKind kind : kinds) {
      if (*this == kind) {
        return true;
      }
    }
    return false;
  }

 private:
  static const TokenKind KeywordTokensStorage[];

  static const bool IsSymbol[];
  static const bool IsGroupingSymbol[];
  static const bool IsOpeningSymbol[];
  static const TokenKind ClosingSymbol[];
  static const bool IsClosingSymbol[];
  static const TokenKind OpeningSymbol[];
  static const bool IsOneCharSymbol[];

  static const bool IsKeyword[];

  static const llvm::StringLiteral FixedSpelling[];

  static const int8_t ExpectedParseTreeSize[];
};

#define CARBON_TOKEN(TokenName) \
  CARBON_ENUM_CONSTANT_DEFINITION(TokenKind, TokenName)
#include "toolchain/lex/token_kind.def"

inline constexpr TokenKind TokenKind::KeywordTokensStorage[] = {
#define CARBON_KEYWORD_TOKEN(TokenName, Spelling) TokenKind::TokenName,
#include "toolchain/lex/token_kind.def"
};
inline constexpr llvm::ArrayRef<TokenKind> TokenKind::KeywordTokens =
    KeywordTokensStorage;

inline constexpr bool TokenKind::IsSymbol[] = {
#define CARBON_TOKEN(TokenName) false,
#define CARBON_SYMBOL_TOKEN(TokenName, Spelling) true,
#include "toolchain/lex/token_kind.def"
};

inline constexpr bool TokenKind::IsGroupingSymbol[] = {
#define CARBON_TOKEN(TokenName) false,
#define CARBON_OPENING_GROUP_SYMBOL_TOKEN(TokenName, Spelling, ClosingName) \
  true,
#define CARBON_CLOSING_GROUP_SYMBOL_TOKEN(TokenName, Spelling, OpeningName) \
  true,
#include "toolchain/lex/token_kind.def"
};

inline constexpr bool TokenKind::IsOpeningSymbol[] = {
#define CARBON_TOKEN(TokenName) false,
#define CARBON_OPENING_GROUP_SYMBOL_TOKEN(TokenName, Spelling, ClosingName) \
  true,
#include "toolchain/lex/token_kind.def"
};

inline constexpr TokenKind TokenKind::ClosingSymbol[] = {
#define CARBON_TOKEN(TokenName) Error,
#define CARBON_OPENING_GROUP_SYMBOL_TOKEN(TokenName, Spelling, ClosingName) \
  ClosingName,
#include "toolchain/lex/token_kind.def"
};

inline constexpr bool TokenKind::IsClosingSymbol[] = {
#define CARBON_TOKEN(TokenName) false,
#define CARBON_CLOSING_GROUP_SYMBOL_TOKEN(TokenName, Spelling, OpeningName) \
  true,
#include "toolchain/lex/token_kind.def"
};

inline constexpr TokenKind TokenKind::OpeningSymbol[] = {
#define CARBON_TOKEN(TokenName) Error,
#define CARBON_CLOSING_GROUP_SYMBOL_TOKEN(TokenName, Spelling, OpeningName) \
  OpeningName,
#include "toolchain/lex/token_kind.def"
};

inline constexpr bool TokenKind::IsOneCharSymbol[] = {
#define CARBON_TOKEN(TokenName) false,
#define CARBON_ONE_CHAR_SYMBOL_TOKEN(TokenName, Spelling) true,
#include "toolchain/lex/token_kind.def"
};

inline constexpr bool TokenKind::IsKeyword[] = {
#define CARBON_TOKEN(TokenName) false,
#define CARBON_KEYWORD_TOKEN(TokenName, Spelling) true,
#include "toolchain/lex/token_kind.def"
};

inline constexpr llvm::StringLiteral TokenKind::FixedSpelling[] = {
#define CARBON_TOKEN(TokenName) "",
#define CARBON_SYMBOL_TOKEN(TokenName, Spelling) Spelling,
#define CARBON_KEYWORD_TOKEN(TokenName, Spelling) Spelling,
#include "toolchain/lex/token_kind.def"
};

inline constexpr int8_t TokenKind::ExpectedParseTreeSize[] = {
#define CARBON_TOKEN(Name) 1,
#define CARBON_TOKEN_WITH_VIRTUAL_NODE(size) 2,
#include "toolchain/lex/token_kind.def"
};

}  // namespace Carbon::Lex

// We use formatv primarily for diagnostics. In these cases, it's expected that
// the spelling in source code should be used.
template <>
struct llvm::format_provider<Carbon::Lex::TokenKind> {
  static void format(const Carbon::Lex::TokenKind& kind, raw_ostream& out,
                     StringRef /*style*/) {
    auto spelling = kind.fixed_spelling();
    if (!spelling.empty()) {
      out << spelling;
    } else {
      // Default to the name if there's no fixed spelling.
      out << kind;
    }
  }
};

#endif  // CARBON_TOOLCHAIN_LEX_TOKEN_KIND_H_
